from multiprocessing.spawn import freeze_support
import multiprocessing
import torch as to
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import pyrado
from pyrado.algorithms.policy_distillation.utils.eval import check_net_performance, check_performance
from pyrado.algorithms.policy_distillation.utils.plot import plot_distillation_performance
from pyrado.algorithms.policy_distillation.utils.load import load_teachers
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
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
parser.add_argument('--max_steps', type=int, default=8_000)
parser.add_argument('--teacher_count', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--num_iters', type=int, default=20)
parser.add_argument('--goal_reward', type=int, default=7000)
parser.add_argument('--env_name', type=str, default='qq-su')
parser.add_argument('--packs', action='store_true', default=False)


if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)
    device = to.device('cpu')

    # Parse arguments
    args = parser.parse_args()
    env_name = args.env_name

    # Teachers
    teachers, teacher_envs, teacher_expl_strat, hidden, ex_dirs = load_teachers(args.teacher_count, env_name, args.packs)

    # Environment
    if (env_name == 'qq-su'):
        env_hparams = dict(dt=1 / args.frequency, max_steps=args.max_steps)
        env_real = ActNormWrapper(QQubeSwingUpSim(**env_hparams))
        env_sim = ActNormWrapper(QQubeSwingUpSim(**env_hparams))
        dp_nom = QQubeSwingUpSim.get_nominal_domain_param()
        # künstliche gap einfügen
    elif (env_name == 'qcp-su'):
        env_hparams = dict(dt=1 / args.frequency, max_steps=args.max_steps)
        env_real = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
        env_sim = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
        dp_nom = QCartPoleSwingUpSim.get_nominal_domain_param()
        #dp_nom["B_pole"] = 0.0
    elif (env_name == 'qbb'):
        env_hparams = dict(dt=1 / args.frequency, max_steps=args.max_steps)
        env_real = ActNormWrapper(QBallBalancerSim(**env_hparams))
        env_sim = ActNormWrapper(QBallBalancerSim(**env_hparams))
        dp_nom = QBallBalancerSim.get_nominal_domain_param()
        # künstliche gap einfügen
    else:
        raise pyrado.TypeErr(msg="No matching environment found!")

    env_sim.domain_param = dp_nom

    timestamp = START.strftime("%Y-%m-%d_%H:%M:%S")
    temp_path = f'{pyrado.TEMP_DIR}/../runs/distillation/{env_name}/{timestamp}/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    pyrado.save(env_sim, "env", "pkl", temp_path)

    obs_dim = env_sim.obs_space.flat_dim
    act_dim = env_sim.act_space.flat_dim

    # Student
    student_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
    student = FNNPolicy(spec=env_sim.spec, **student_hparam)
    expl_strat = NormalActNoiseExplStrat(student, std_init=0.6)
    optimizer = to.optim.Adam(
                [
                    {"params": student.parameters()},
                    {"params": expl_strat.noise.parameters()},
                    #{"params": self.critic.parameters()},
                ],
                lr=1e-4,
            )

    teacher_weights = np.ones(len(teachers))
    

    # Criterion
    criterion = to.nn.KLDivLoss(log_target=True, reduction='batchmean')
    #criterion = torch.nn.MSELoss()
    writer = SummaryWriter(temp_path)

    # Student sampling
    for epoch in range(args.num_epochs):
        act_student = []
        obss = []
        obs = env_sim.reset()
        losses = []

        for i in range(args.max_steps):
            obs = to.as_tensor(obs).float()
            obss.append(obs)

            s_dist = expl_strat.action_dist_at(student(obs)) ##student.get_dist(obs)
            s_act = s_dist.sample()
            act_student.append(s_dist.log_prob(s_act))      #s_dist.mean()
            obs, rew, done, _ = env_sim.step(s_act.numpy().reshape(-1))

            if done:
                obs = env_sim.reset()

        obss = to.stack(obss, 0)

        for _ in range(args.num_iters):
            optimizer.zero_grad()

            s_dist = expl_strat.action_dist_at(student(obss)) ##student.get_dist(obss)
            s_act = s_dist.sample()

            loss = 0
            for t_idx, teacher in enumerate(teachers):
                #act_teacher = []

                t_dist = teacher_expl_strat[t_idx].action_dist_at(teacher(obss)) ##teacher.get_dist(obss) #oder student(obss)
                t_act = t_dist.sample()
                #act_teacher.append(t_dist.log_prob(s_act))  #t_dist.mean()

                l = teacher_weights[t_idx] * criterion(t_dist.log_prob(s_act), s_dist.log_prob(s_act))
                loss += l
                losses.append([t_idx, l.item()])
            print(f'Epoch {epoch} Loss: {loss.item()}')
            loss.backward()
            optimizer.step()

        writer.add_scalars(f'Teachers', {f'Teacher {i}': l for i, l in losses}, epoch)

        to.save(
                {
                    "policy": student.state_dict(),#self.policy.state_dict(),
                    #"critic": self.critic.state_dict(),
                    "expl_strat": expl_strat.state_dict(),
                },
                temp_path + "student.pt",
            )

    print('Finished training the student!')

    # Check student performance:
    check_performance(env=env_real, policy=student, name='student_after', path=temp_path)


    a_pool = multiprocessing.Pool(processes=4)
    su = a_pool.starmap(check_performance, [(env, deepcopy(student), f'student_on_teacher_env_{idx}', 1000, temp_path) for idx, env in enumerate(teacher_envs)])

    # Check student performance on teacher envs:
    #for idx, env in enumerate(teacher_envs):
    #    check_performance(env=env, policy=student, name=f'student_on_teacher_env_{idx}', path=temp_path)
    #    env.close()

    plot_distillation_performance(env_name, timestamp, goalReward=args.goal_reward, showPlot=False)

    env_sim.close()
    env_real.close()
    writer.flush()
    writer.close()

    print('Finished evaluating the student!')
