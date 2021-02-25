import torch as to
from torch.utils.tensorboard import SummaryWriter

import pyrado
from pyrado.algorithms.policy_distillation.utils.eval import check_net_performance, check_performance
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.policies.base import Policy, TwoHeadedPolicy
from pyrado.utils.experiments import load_experiment

import numpy as np

from datetime import datetime

START = datetime.now()
temp_path = f'{pyrado.TEMP_DIR}/../runs/distillation/' + START.strftime("%Y-%m-%d_%H:%M:%S")


if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)
    device = to.device('cpu')

    # Teachers
    hidden = []
    teachers = []
    env_name = ''
    for idx in range(8):
        # Get the experiment's directory to load from
        ex_dir = ask_for_experiment(max_display = 50) # if args.dir is None else args.dir

        # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
        env_teacher, policy, _ = load_experiment(ex_dir) #, args)
        if (env_name == ''):
            env_name = env_teacher.name
        elif (env_teacher.name != env_name):
            raise pyrado.TypeErr(msg="The teacher environment does not match the previous one(s)!")
        teachers.append(policy)

        # auf doppelte teacher testen

    for i, t in enumerate(teachers):
        if isinstance(t, Policy):
            # Reset the policy / the exploration strategy
            t.reset()

            # Set dropout and batch normalization layers to the right mode
            t.eval()

            # Check for recurrent policy, which requires special handling
            if t.is_recurrent:
                # Initialize hidden state var
                hidden[i] = t.init_hidden()

    teacher_expl_strat = [NormalActNoiseExplStrat(teacher, std_init=0.6) for teacher in teachers]

    # Environment
    if (env_name == 'qq-su'):
        env_hparams = dict(dt=1 / 250.0, max_steps=600)
        env_real = ActNormWrapper(QQubeSwingUpSim(**env_hparams))
        env_sim = ActNormWrapper(QQubeSwingUpSim(**env_hparams))
        dp_nom = QQubeSwingUpSim.get_nominal_domain_param()
        # künstliche gap einfügen
    elif (env_name == 'qcp-su'):
        env_hparams = dict(dt=1 / 250.0, max_steps=600)
        env_real = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
        env_sim = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
        dp_nom = QCartPoleSwingUpSim.get_nominal_domain_param()
        dp_nom["B_pole"] = 0.0
    else:
        raise pyrado.TypeErr(msg="No matching environment found!")

    env_sim.domain_param = dp_nom

    obs_dim = env_sim.obs_space.flat_dim
    act_dim = env_sim.act_space.flat_dim

    #log_std = sum([t.log_std.cpu().detach().item()/len(teachers) for t in teachers])
    #print('log_std', log_std)

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
    #student.load(f'{TEACHER_PATH}/trained/student.pt')


    # Check teacher performance:
    teacher_weights = np.ones(len(teachers))
    """
    nets = teachers[:]
    nets.append(student)
    names=[ f'teacher {t}' for t in range(len(teachers)) ]
    names.append('student_before_sim')
    performances = check_net_performance(env=env_sim, nets=nets, names=names, reps=1000)
    for idx, sums in enumerate(performances[:-1]):
        teacher_weights[idx] = np.mean(sums)-np.std(sums)
    teacher_weights = teacher_weights / sum(teacher_weights) * len(teachers)
    """

    #input('Press any key to continue...')

    #teacher_weights = np.array([t.log_std.cpu().detach().exp().item() for t in teachers])
    #teacher_weights = (teacher_weights / teacher_weights.sum()) * len(teachers)

    #print('teacher_weight',teacher_weights)
    #[ 0.8618  1.2897  1.1004  1.3799  0.6525  0.1363  1.0253  1.554 ]


    # Student performance before learning:
    #check_performance(env_real, student, 'student_before_real')

    #exit()

    # Criterion
    criterion = to.nn.KLDivLoss(log_target=True, reduction='batchmean')
    #criterion = torch.nn.MSELoss()
    writer = SummaryWriter(temp_path)

    # Student sampling
    for epoch in range(500):
        act_student = []
        obss = []
        obs = env_sim.reset()
        losses = []

        for i in range(8000):
            obs = to.as_tensor(obs).float()
            obss.append(obs)

            s_dist = expl_strat.action_dist_at(student(obs)) ##student.get_dist(obs)
            s_act = s_dist.sample()
            act_student.append(s_dist.log_prob(s_act))      #s_dist.mean()
            obs, rew, done, _ = env_sim.step(s_act.numpy().reshape(-1))

            if done:
                obs = env_sim.reset()

        obss = to.stack(obss, 0)

        for _ in range(20):
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
                temp_path + "student.pt",#f"{self._save_name}.pt",
            )


    # Check student performance:
    check_performance(env_real, student, 'student_after')

    env_sim.close()
    env_real.close()
    writer.flush()
    writer.close()
