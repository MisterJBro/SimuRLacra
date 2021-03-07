import numpy as np
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
import torch as to
import multiprocessing as mp
from multiprocessing import freeze_support
import os
import pyrado
from pyrado.sampling.envs import Envs
from pyrado.algorithms.step_based.ppo_gae import PPOGAE
from pyrado.policies.base import Policy, TwoHeadedPolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.input_output import print_cbt
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.experiments import load_experiment
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.utils.data_types import RenderMode
from pyrado.algorithms.policy_distillation.utils.load import load_teachers

import argparse
from datetime import datetime

def check_net_performance(env, nets, names, max_len=8000, reps=1000, path=''):
    start = datetime.now()
    print('Started checking net performance.')
    envs = Envs(cpu_num=min(mp.cpu_count(),len(nets)), env_num=len(nets), env=env, game_len=max_len, gamma=0.99, lam=0.97)
    su = []
    hidden = []
    for i, net in enumerate(nets):
        if isinstance(net, Policy):
            # Reset the policy / the exploration strategy
            net.reset()

            # Set dropout and batch normalization layers to the right mode
            net.eval()

            # Check for recurrent policy, which requires special handling
            if net.is_recurrent:
                # Initialize hidden state var
                hidden[i] = net.init_hidden()

    for rep in range(reps):
        obss = envs.reset()
        obs_to = to.from_numpy(obss).type(to.get_default_dtype())  # policy operates on PyTorch tensors
        i = 0
        while i < max_len:
            acts = [] # = np.concatenate([t.get_action(obss[i])[0] for i, t in enumerate(nets)], 0)
            for i, net in enumerate(nets):
                with to.no_grad():
                    if isinstance(net, Policy):
                        if net.is_recurrent:
                            if isinstance(getattr(net, "policy", net), TwoHeadedPolicy):
                                act_to, _, _ = net(obs_to, hidden[i])
                            else:
                                act_to, _ = net(obs_to, hidden[i])
                        else:
                            if isinstance(getattr(net, "policy", net), TwoHeadedPolicy):
                                act_to, _ = net(obs_to)
                            else:
                                act_to = net(obs_to)
                    else:
                        # If the policy ist not of type Policy, it should still operate on PyTorch tensors
                        act_to = net(obs_to)
                acts.append(act_to.detach().cpu().numpy())
            act = np.concatenate(acts)
            obss = envs.step(act, np.zeros(len(nets)))
            i+=1
        lens = np.array([len(s) for s in envs.buf.sections])
        su.append(envs.buf.rew_buf.sum(1)/lens)
        print('rep', rep)

    envs.close()

    su = np.stack(su, 1)
    for idx, sums in enumerate(su):
        print('Endsumme (', names[idx], 'from', reps, 'reps ): MEAN =', np.mean(sums), 'STD =', np.std(sums),
            'MIN =', np.min(sums), 'MAX =', np.max(sums), 'MEDIAN =', np.median(sums))

    save_performance(start, su, names, env.name, path)
    return su
    """
        Traceback (most recent call last):
            File "eval.py", line 234, in <module>
                check_old_teacher_performance(teacher_count=args.teacher_count, frequency=args.frequency, reps=args.reps)
            File "eval.py", line 159, in check_old_teacher_performance
                check_net_performance(env=env_sim, nets=teachers[:], names=names, reps=reps)
            File "eval.py", line 70, in check_net_performance
                obss = envs.step(act, np.zeros(len(nets)))
            File "/home/benedikt/UNI/SimuRLacra/Pyrado/pyrado/sampling/envs.py", line 90, in step
                self.buf.store(self.obss, acts, rews, vals)
            File "/home/benedikt/UNI/SimuRLacra/Pyrado/pyrado/sampling/buffer.py", line 70, in store
                self.act_buf[:, self.ptr] = act
            ValueError: could not broadcast input array from shape (4,1) into shape (2,1)
    """


def check_performance(env, policy, name, n=1000, max_len=8000, path=''):
    print('Started checking performance.')
    start=datetime.now()
    su = []
    # Test the policy in the environment
    done, param, state = False, None, None
    for i in range(n):
        print('rollout', i, '/', n-1)
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=False, video=False),
            eval=True,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        # print_domain_params(env.domain_param)
        su.append(ro.undiscounted_return())
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, param, state = False, None, None #done, state, param = after_rollout_query(env, policy, ro)
    sums = np.array(su)
    print('Endsumme (' + name + ' from', n, 'reps ): MEAN =', np.mean(sums), 'STD =', np.std(sums),
          'MIN =', np.min(sums), 'MAX =', np.max(sums), 'MEDIAN =', np.median(sums))
    save_performance(start, sums, np.array([name]), env.name, path)
    return np.mean(sums)-np.std(sums)


def save_performance(start, sums, names, env_name='', path=''):
    env_str = f'{env_name}_' if env_name!='' else ''

    if path == '':
        np.save( f'{pyrado.TEMP_DIR}/eval/sums_{env_str}{start.strftime("%Y-%m-%d_%H:%M:%S")}', sums)
        np.save( f'{pyrado.TEMP_DIR}/eval/names_{env_str}{start.strftime("%Y-%m-%d_%H:%M:%S")}', names)
    else:
        eval_path = f'{path}eval/'
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        np.save( f'{eval_path}sums_{env_str}{start.strftime("%Y-%m-%d_%H:%M:%S")}', sums)
        np.save( f'{eval_path}names_{env_str}{start.strftime("%Y-%m-%d_%H:%M:%S")}', names)


def check_old_teacher_performance(teacher_count:int=8, frequency:int=250, reps:int=1000):
    # Teachers
    teachers, _, teacher_expl_strat, hidden, ex_dirs, env_name = load_teachers(teacher_count)

    env_hparams = dict(dt=1 / frequency, max_steps=reps)
    # Environment
    if (env_name == 'qq-su'):
        env_sim = ActNormWrapper(QQubeSwingUpSim(**env_hparams))
        dp_nom = QQubeSwingUpSim.get_nominal_domain_param()
    elif (env_name == 'qcp-su'):
        env_sim = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
        dp_nom = QCartPoleSwingUpSim.get_nominal_domain_param()
    elif (env_name == 'qbb'):
        env_sim = ActNormWrapper(QBallBalancerSim(**env_hparams))
        dp_nom = QBallBalancerSim.get_nominal_domain_param()
    else:
        raise pyrado.TypeErr(msg="No matching environment found!")
    env_sim.domain_param = dp_nom

    names=[ f'teacher {t}' for t in range(len(teachers)) ]
    check_net_performance(env=env_sim, nets=teachers[:], names=names, reps=reps)
    env_sim.close


if __name__ == "__main__":
    freeze_support()
    to.set_default_dtype(to.float32)
    device = to.device('cpu')

    # Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--simulate', type=bool, default=False)
    parser.add_argument('--singlePolicy', type=bool, default=False)
    parser.add_argument('--teacher_count', type=int, default=8)
    parser.add_argument('--frequency', type=int, default=250)
    parser.add_argument('--reps', type=int, default=8)
    

    # Parse command line arguments
    args = parser.parse_args()

    if args.singlePolicy:

        ex_dir = ask_for_experiment()
        env_sim, _, extra = load_experiment(ex_dir)
        expl_strat = extra["expl_strat"]

        #env_sim will not be used here, because we want to evaluate the policy on a different environment
        #we can use it, by changing the parameters to the default ones:
        if (env_sim.name == 'qq-su'):
            env_sim.domain_param = QQubeSwingUpSim.get_nominal_domain_param()
        elif (env_sim.name == 'qcp-su'):
            env_sim.domain_param = QCartPoleSwingUpSim.get_nominal_domain_param()
        else:
            raise pyrado.TypeErr(msg="No matching environment found!")

        if args.simulate:
            # Test the policy in the environment
            done, param, state = False, None, None
            while not done:
                ro = rollout(
                    env_sim,
                    expl_strat,
                    render_mode=RenderMode(text=True, video=True),
                    eval=True,
                    reset_kwargs=dict(domain_param=param, init_state=state),
                )
                # print_domain_params(env.domain_param)
                print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
                done, state, param = after_rollout_query(env_sim, expl_strat, ro)

        else:
            # Evaluate
            check_performance(env_sim, expl_strat, 'student_after', n=1000)

        env_sim.close()

    else:
        check_old_teacher_performance(teacher_count=args.teacher_count, frequency=args.frequency, reps=args.reps)