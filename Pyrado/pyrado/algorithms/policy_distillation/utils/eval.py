import multiprocessing
import numpy as np
from numpy.lib.function_base import iterable
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
from pyrado.algorithms.policy_distillation.utils.load import load_student, load_teachers
from copy import deepcopy

import argparse
from datetime import datetime

def check_net_performance(env, nets, names, max_len=8000, reps=1000, path=''):
    start = datetime.now()
    print(f'Started checking net performance: {len(nets)} networks on {env.name} ({start.strftime("%Y-%m-%d_%H:%M:%S")})')
    envs = Envs(cpu_num=min(mp.cpu_count(), len(nets)), env_num=len(nets), env=env, game_len=max_len, gamma=0.99, lam=0.97)
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
        iter = 0
        while iter < max_len:
            acts = [] # = np.concatenate([t.get_action(obss[i])[0] for i, t in enumerate(nets)], 0)
            for i, net in enumerate(nets):
                with to.no_grad():
                    if isinstance(net, Policy):
                        if net.is_recurrent:
                            if isinstance(getattr(net, "policy", net), TwoHeadedPolicy):
                                act_to, _, _ = net(obs_to[i].reshape(1,-1), hidden[i])
                            else:
                                act_to, _ = net(obs_to[i].reshape(1,-1), hidden[i])
                        else:
                            if isinstance(getattr(net, "policy", net), TwoHeadedPolicy):
                                act_to, _ = net(obs_to[i].reshape(1,-1))
                            else:
                                act_to = net(obs_to[i].reshape(1,-1))
                    else:
                        # If the policy ist not of type Policy, it should still operate on PyTorch tensors
                        act_to = net(obs_to[i].reshape(1,-1))
                acts.append(act_to.detach().cpu().numpy())
            act = np.concatenate(acts)
            obss = envs.step(act, np.zeros(len(nets)))
            iter+=1
        lens = np.array([len(s) for s in envs.buf.sections])
        su.append(envs.buf.rew_buf.sum(1)/lens)
        print('finished rep', rep, 'at', datetime.now().strftime("%H:%M:%S"))

    envs.close()

    su = np.stack(su, 1)
    for idx, sums in enumerate(su):
        print('Endsumme (', names[idx], 'from', reps, 'reps ): MEAN =', np.mean(sums), 'STD =', np.std(sums),
            'MIN =', np.min(sums), 'MAX =', np.max(sums), 'MEDIAN =', np.median(sums))

    save_performance(start, su, names, env.name, path)
    return su


def check_performance(env, policy, name, n=1000, path='', verbose=True):
    print('Started checking performance.')
    start=datetime.now()
    su = []
    # Test the policy in the environment
    done, param, state = False, None, None
    for i in range(n):
        if verbose:
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
        if verbose:
            print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, param, state = False, None, None #done, state, param = after_rollout_query(env, policy, ro)
    sums = np.array(su)
    print('Endsumme (' + name + ' from', n, 'reps ): MEAN =', np.mean(sums), 'STD =', np.std(sums),
          'MIN =', np.min(sums), 'MAX =', np.max(sums), 'MEDIAN =', np.median(sums))
    save_performance(start, sums, np.array([name]), env.name, path)
    return (name, np.mean(sums), np.std(sums), sums)

"""
def check_performance(env, policy, name:str, n:int=1000, path=''):
    print('Started checking performance.')
    start=datetime.now()
    a_pool = multiprocessing.Pool()
    su = a_pool.starmap(rollout_wrapper, [(deepcopy(env), deepcopy(policy), n, i) for i in range(n)])
    sums = np.array(su)
    print('Endsumme (' + name + ' from', n, 'reps ): MEAN =', np.mean(sums), 'STD =', np.std(sums),
          'MIN =', np.min(sums), 'MAX =', np.max(sums), 'MEDIAN =', np.median(sums))
    save_performance(start, sums, np.array([name]), env.name, path)
    return np.mean(sums)-np.std(sums)

def rollout_wrapper(env, policy, n, i):
    param, state = None, None 
    print('rollout', i, '/', n-1)
    ro = rollout(
        env,
        policy,
        render_mode=RenderMode(text=False, video=False),
        eval=True,
        reset_kwargs=dict(domain_param=param, init_state=state),
    )
    env.close()
    print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
    return ro.undiscounted_return()
"""

def save_performance(start, sums, names, env_name='', path=''):
    env_str = f'{env_name}_{names[0]}' if env_name!='' else ''

    if path == '':
        np.save( f'{pyrado.TEMP_DIR}/eval/sums_{env_str}_{start.strftime("%Y-%m-%d_%H:%M:%S")}', sums)
        np.save( f'{pyrado.TEMP_DIR}/eval/names_{env_str}_{start.strftime("%Y-%m-%d_%H:%M:%S")}', names)
    else:
        eval_path = os.path.join(pyrado.TEMP_DIR,path,'eval') #f'{path}eval/'
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        np.save( os.path.join(eval_path,f'sums_{env_str,start.strftime("%Y-%m-%d_%H:%M:%S")}'), sums)  #f'{eval_path}sums_{env_str}{start.strftime("%Y-%m-%d_%H:%M:%S")}',
        np.save( os.path.join(eval_path,f'names_{env_str,start.strftime("%Y-%m-%d_%H:%M:%S")}'), names)


def check_pack_performance(teacher_envs, teacher_expl_strat, ex_dirs, reps):
    names=[ f'teacher {t}' for t in range(len(teacher_expl_strat)) ]
    a_pool = multiprocessing.Pool(processes=4)
    su = a_pool.starmap_async(check_performance, [(teacher_envs[idx], teacher_expl_strat[idx], names[idx], reps, ex_dirs[idx]) for idx in range(len(teacher_expl_strat))]).get()
    a_pool.close()
    a_pool.join()
    return su[3], names


def check_old_teacher_performance(env_name:str, teacher_count:int=8, frequency:int=250, max_steps:int=600, reps:int=1000, packs:bool=None):
    # Teachers
    teachers, _, teacher_expl_strat, _, _, ex_dirs = load_teachers(teacher_count, env_name, packs)

    env_hparams = dict(dt=1 / frequency, max_steps=max_steps)
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

    a_pool = multiprocessing.Pool(processes=4)
    su = a_pool.starmap_async(check_performance, [(deepcopy(env_sim), policy, names[idx], 1000, ex_dirs[idx]) for idx, policy in enumerate(teacher_expl_strat)]).get()
    a_pool.close()
    a_pool.join()

    print(su[:2])

    #check_net_performance(env=env_sim, nets=teachers[:], names=names, reps=reps)
    env_sim.close


def check_performance_on_random_envs(policy, env, count:int, ex_dir, iters):
    from pyrado.algorithms.policy_distillation.train_teachers import get_random_envs
    test_envs = get_random_envs(env_count = count, env = env)

    a_pool = multiprocessing.Pool(processes=4)
    su = a_pool.starmap_async(check_performance, [(env, deepcopy(policy), f'student_on_random_env_{idx}', iters, ex_dir) for idx, env in enumerate(test_envs)]).get()
    a_pool.close()
    a_pool.join()

    # Check student performance on teacher envs:
    for idx, env in enumerate(test_envs):
        env.close()
    
    return su


if __name__ == "__main__":
    freeze_support()
    to.set_default_dtype(to.float32)
    device = to.device('cpu')

    # Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--singlePolicy', action='store_true', default=False)
    parser.add_argument('--random_envs', action='store_true', default=False)
    parser.add_argument('--student', action='store_true', default=False)
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--simulate', action='store_true', default=False)
    parser.add_argument('--animation', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--teacher_count', type=int, default=8)
    parser.add_argument('--frequency', type=int, default=250)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--env_name', type=str, default='qq-su')
    parser.add_argument('--packs', action='store_true', default=False)

    # Parse command line arguments
    args = parser.parse_args()

    if args.singlePolicy:
        if not args.student:
            ex_dir = ask_for_experiment(max_display=150, env_name=args.env_name, base_dir=pyrado.TEMP_DIR)
            env_sim, _, extra = load_experiment(ex_dir)
            expl_strat = extra["expl_strat"]
            print(f'Std variance {expl_strat.std.item()}')
        else:
            student, env_sim, expl_strat, ex_dir_stud = load_student(1.0/args.frequency, args.env_name, args.folder, args.max_steps)
            if args.random_envs:
                print(ex_dir_stud)
                check_performance_on_random_envs(expl_strat, env_sim, args.teacher_count, ex_dir_stud, 100)
                exit()


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
            check_performance(env_sim, expl_strat, 'student_after', n=args.reps)

        env_sim.close()

    else:
        check_old_teacher_performance(args.env_name, teacher_count=args.teacher_count, frequency=args.frequency, max_steps=args.max_steps, reps=args.reps, packs=args.packs)