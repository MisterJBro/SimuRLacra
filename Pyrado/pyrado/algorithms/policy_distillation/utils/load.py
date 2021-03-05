
import torch as to

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.experiments import load_experiment
from pyrado.policies.base import Policy
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.feed_forward.fnn import FNNPolicy
    
def load_teachers(teacher_count:int):
    # Teachers
    hidden = []
    teachers = []
    teacher_envs = []
    teacher_expl_strat = []
    ex_dirs = []
    env_name = ''
    for _ in range(teacher_count):
        # Get the experiment's directory to load from
        ex_dir = ask_for_experiment(max_display = 100) # if args.dir is None else args.dir

        # Check if this teacher was already selected before
        while ex_dir in ex_dirs:
            print('This teacher environment was already used. Choose a new one!')
            ex_dir = ask_for_experiment(max_display = 50)
        ex_dirs.append(ex_dir)

        print(ex_dir)
        # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
        env_teacher, policy, extra = load_experiment(ex_dir) #, args)
        if (env_name == ''):
            env_name = env_teacher.name
        elif (env_teacher.name != env_name):
            raise pyrado.TypeErr(msg="The teacher environment does not match the previous one(s)!")
        teachers.append(policy)
        teacher_envs.append(env_teacher)
        teacher_expl_strat.append(extra["expl_strat"])

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


    return teachers, teacher_expl_strat, hidden, ex_dirs, env_name

def load_student(dt:float, env_type:str, folder:str, max_steps:int):
    # Get the experiment's directory to load from
    #ask_for_experiment(max_display=50) if args.dir is None else args.dir
    ex_dir = f'{pyrado.TEMP_DIR}/../runs/distillation/{env_type}/{folder}/'

    # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
    #env_sim, policy, _ = load_experiment(ex_dir, args)
    checkpoint = to.load(f'{ex_dir}/student.pt')

    # Environment
    if (env_type == 'qq-su'):
        env_hparams = dict(dt=dt, max_steps=max_steps)
        env_sim = ActNormWrapper(QQubeSwingUpSim(**env_hparams))
    elif (env_type == 'qcp-su'):
        env_hparams = dict(dt=dt, max_steps=max_steps)
        env_sim = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
    elif (env_type == 'qbb'):
        env_hparams = dict(dt=dt, max_steps=max_steps)
        env_sim = ActNormWrapper(QBallBalancerSim(**env_hparams))
    else:
        raise pyrado.TypeErr(msg="No matching environment found!")


    student_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
    student = FNNPolicy(spec=env_sim.spec, **student_hparam)

    student.load_state_dict(checkpoint['policy'])

    return student, env_sim
