from multiprocessing.spawn import freeze_support
import multiprocessing
import torch as to
from torch.utils.tensorboard import SummaryWriter, writer
import argparse
import os
import pyrado
from pyrado.algorithms.policy_distillation.utils.eval import check_net_performance, check_performance, check_performance_on_random_envs
from pyrado.algorithms.policy_distillation.utils.plot import plot_distillation_performance
from pyrado.algorithms.policy_distillation.utils.load import load_teachers
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.sampling.envs import Envs
#from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.feed_forward.fnn import FNNPolicy
#, TwoHeadedPolicy
#from pyrado.utils.experiments import load_experiment
from copy import deepcopy
import numpy as np

from datetime import datetime


START = datetime.now()

# Parameters
parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--frequency', type=int, default=250)
parser.add_argument('--max_steps', type=int, default=1500)
parser.add_argument('--max_eval_steps', type=int, default=1500)
parser.add_argument('--teacher_count', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--num_iters', type=int, default=10)
parser.add_argument('--env_name', type=str, default='qq-su')
parser.add_argument('--packs', action='store_true', default=False)
parser.add_argument('--eval_reps', type=int, default=100)
parser.add_argument('--eval_policy', type=str, default='student')


def on_policy_distill(args, env, student, expl_strat, optimizer, teachers, teacher_expl_strat, teacher_weights, writer, teacher_envs):
    # Criterion
    criterion = to.nn.KLDivLoss(log_target=True, reduction='batchmean')
    save_dict = {
                    "policy": student.state_dict(),
                    "expl_strat": expl_strat.state_dict(),
                }
    
    # Environments
    envs = Envs(len(teachers), len(teachers), env, args.max_steps, 0.99, 0.97, env_list=teacher_envs)
    max_avg_ret = 0

    # Student sampling
    for epoch in range(args.num_epochs):
        losses = []

        # Sample observations
        obss = envs.reset()

        for _ in range(args.max_steps):
            obss = to.as_tensor(obss)   
            with to.no_grad():
                acts = expl_strat(obss).cpu().numpy()
            obss = envs.step(acts, np.zeros_like(acts).reshape(-1))

        avg_ret = np.mean(envs.ret_and_adv())

        # Save best performing student
        if avg_ret > max_avg_ret:
            max_avg_ret = avg_ret
            save_dict = {
                "policy": deepcopy(student.state_dict()),
                "expl_strat": deepcopy(expl_strat.state_dict()),
            }

        obss, _, _, _, _ = envs.get_data(to.device('cpu'))
        obss = to.as_tensor(obss).float().reshape(len(teachers), -1, obss.shape[-1])

        # Train student
        for _ in range(args.num_iters):
            optimizer.zero_grad()

            loss = 0
            for t_idx, teacher in enumerate(teachers):
                s_dist = expl_strat.action_dist_at(student(obss[t_idx]))
                s_act = s_dist.sample()
                t_dist = teacher_expl_strat[t_idx].action_dist_at(teacher(obss[t_idx]))

                l = teacher_weights[t_idx] * criterion(t_dist.log_prob(s_act), s_dist.log_prob(s_act)) #mse(student(obss[t_idx]), teacher(obss[t_idx]))
                loss += l
                losses.append([t_idx, l.item()])
            print(f'Epoch {epoch} Loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        #print(f'Std var: {expl_strat.std}')

        writer.add_scalars(f'Teachers', {f'Teacher {i}': l for i, l in losses}, epoch)

    # Load and save best student
    student.load_state_dict(save_dict["policy"])
    expl_strat.load_state_dict(save_dict["expl_strat"])

    to.save(
        save_dict,
        temp_path + "student.pt",
    )
    print('Finished training the student!')
    envs.close()

if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)
    device = to.device('cpu')

    # Parse arguments
    args = parser.parse_args()
    env_name = args.env_name

    # Teachers
    teachers, teacher_envs, teacher_expl_strat, teacher_critics, hidden, ex_dirs = load_teachers(args.teacher_count, env_name, args.packs)

    # Environment
    env_hparams = dict(dt=1 / args.frequency, max_steps=args.max_steps)
    if (env_name == 'qq-su'):
        env_real = QQubeSwingUpSim(**env_hparams)
    elif (env_name == 'qcp-su'):
        env_real = QCartPoleSwingUpSim(**env_hparams)
    elif (env_name == 'qbb'):
        env_real = QBallBalancerSim(**env_hparams)
    else:
        raise pyrado.TypeErr(msg="No matching environment found!")

    # Wrapper
    randomizer = create_default_randomizer(env_real)
    env_real = DomainRandWrapperLive(env_real, randomizer)
    env_real = ActNormWrapper(env_real)

    timestamp = START.strftime("%Y-%m-%d_%H:%M:%S")
    temp_path = f'{pyrado.TEMP_DIR}/../runs/distillation/{env_name}/{timestamp}/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    obs_dim = env_real.obs_space.flat_dim
    act_dim = env_real.act_space.flat_dim

    # Student
    student_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
    student = FNNPolicy(spec=env_real.spec, **student_hparam)

    expl_strat = NormalActNoiseExplStrat(student, std_init=0.15)
    optimizer = to.optim.Adam(
        [
            {"params": student.parameters()},
            #{"params": expl_strat.noise.parameters()},
            #{"params": self.critic.parameters()},
        ],
        lr=5e-4,
    )

    teacher_weights = np.ones(len(teachers))

    # Writer
    writer = SummaryWriter(temp_path)

    # Train
    on_policy_distill(args, env_real, student, expl_strat, optimizer, teachers, teacher_expl_strat, teacher_weights, writer, teacher_envs)

    # Check student performance:
    iters = args.eval_reps
    policy = student if args.eval_policy=="student" else expl_strat
    check_performance(env=env_real, policy=policy, name='student_after', n=iters, path=temp_path, verbose=False, max_eval_steps=args.max_eval_steps)

    processes = 4
    a_pool = multiprocessing.Pool(processes=processes)
    su = a_pool.starmap_async(check_performance, [(env, deepcopy(policy), f'student_on_teacher_env_{idx}', iters, temp_path, False, args.max_eval_steps) for idx, env in enumerate(teacher_envs)]).get()
    a_pool.close()
    a_pool.join()

    # Check student performance on teacher envs:
    for idx, env in enumerate(teacher_envs):
        env.close()

    random_env_count = 8
    check_performance_on_random_envs(policy, env_real, random_env_count, temp_path, iters, args.max_eval_steps)

    plot_distillation_performance(env_name, timestamp, goalReward=args.max_eval_steps*.7, showPlot=True)

    env_real.close()
    writer.flush()
    writer.close()

    print('Finished evaluating the student!')
