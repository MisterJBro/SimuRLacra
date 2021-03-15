
import torch as to
import argparse
import os
import sys
import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.experiments import load_experiment
from pyrado.utils.saving_loading import save, load
from pyrado.policies.base import Policy
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.policies.feed_forward.fnn import FNNPolicy

def load_teachers(teacher_count:int, env_name:str, packs:bool=False):
    if not packs:
        base_dir = pyrado.TEMP_DIR
        # Teachers
        hidden = []
        teachers = []
        teacher_envs = []
        teacher_expl_strat = []
        teacher_critic = []
        ex_dirs = []
        for _ in range(teacher_count):
            # Get the experiment's directory to load from
            ex_dir = ask_for_experiment(max_display = 150, env_name=env_name, base_dir=base_dir)

            # Check if this teacher was already selected before
            while ex_dir in ex_dirs:
                print('This teacher environment was already used. Choose a new one!')
                ex_dir = ask_for_experiment(max_display = 150, env_name=env_name, base_dir=base_dir)
            ex_dirs.append(ex_dir)

            print(ex_dir)
            # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
            env_teacher, policy, extra = load_experiment(ex_dir)
            if (env_teacher.name != env_name):
                raise pyrado.TypeErr(msg="The teacher environment does not match the previous one(s)!")
            teachers.append(policy)
            teacher_envs.append(env_teacher)
            teacher_expl_strat.append(extra["expl_strat"])
            teacher_critic.append(extra["vfcn"])

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
    else:
        packlist = ask_for_packlist()
        teachers, teacher_envs, teacher_expl_strat, teacher_critic, hidden, ex_dirs, _, _ = load_packed_teachers(env_name, packlist, teacher_count)
    
    return teachers, teacher_envs, teacher_expl_strat, teacher_critic, hidden, ex_dirs

def load_student(dt:float, env_type:str, folder:str, max_steps:int):
    # Get the experiment's directory to load from
    ex_dir = f'{pyrado.TEMP_DIR}/../runs/distillation/{env_type}/{folder}/'

    # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
    checkpoint = to.load(f'{ex_dir}student.pt')

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

    expl_strat = NormalActNoiseExplStrat(student, std_init=0.6)
    expl_strat.load_state_dict(checkpoint['expl_strat'])

    return student, env_sim, expl_strat, ex_dir


def pack_teachers(teacher_count:int, env_name:str, suffix:str, packs:bool=False, evalReps:int=0):
    teachers, teacher_envs, teacher_expl_strat, teacher_critics, hidden, ex_dirs = load_teachers(teacher_count, env_name, packs)

    pack_loaded_teachers(teachers, teacher_envs, teacher_expl_strat, teacher_critics, hidden, ex_dirs, evalReps, suffix)

    for env in teacher_envs:
        env.close()

    print('Finished packing teachers.')


def load_teachers_from_dir(env_name, ex_dirs):
    # Teachers
    hidden = []
    teachers = []
    teacher_envs = []
    teacher_expl_strat = []
    teacher_critic = []
    for ex_dir in ex_dirs:
        # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
        env_teacher, policy, extra = load_experiment(ex_dir)
        if (env_teacher.name != env_name):
            raise pyrado.TypeErr(msg="The teacher environment does not match the previous one(s)!")
        teachers.append(policy)
        teacher_envs.append(env_teacher)
        teacher_expl_strat.append(extra["expl_strat"])
        teacher_critic.append(extra["vfcn"])

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

    return teachers, teacher_envs, teacher_expl_strat, teacher_critic, hidden

    

def pack_loaded_teachers(teachers, teacher_envs, teacher_expl_strat, teacher_critics, hidden, ex_dirs, evalReps, suffix, max_eval_steps):
    if evalReps>0:
        from pyrado.algorithms.policy_distillation.utils.eval import check_pack_performance
        sums, names, rollouts = check_pack_performance(teacher_envs, teachers, ex_dirs, evalReps, max_eval_steps)
        suffix+="_eval"
    else:
        sums, names, rollouts = [], [], []

    pack = {
        "teachers": teachers,
        "teacher_envs": teacher_envs,
        "teacher_expl_strat": teacher_expl_strat,
        "teacher_critics": teacher_critics,
        "hidden": hidden,
        "ex_dirs": ex_dirs,
        "sums": sums, 
        "names": names,
        "rollouts": rollouts
        }

    env_name = teacher_envs[0].name
    teacher_count = len(teachers)
    pack_path = f'{pyrado.TEMP_DIR}/packs/{env_name}'
    if not os.path.exists(pack_path):
        os.makedirs(pack_path)
    save(pack, "teachers", "pkl", pack_path, {"prefix":teacher_count, "suffix":suffix})


def load_packed_teachers(env_name:str, packs:list, teacher_count:int):
    teacher_counts = packs[0]
    suffixes = [ f'{p}' for p in packs[1]]
    all_teachers, all_teacher_envs, all_teacher_expl_strat, all_teacher_critic, all_hidden, all_ex_dirs, all_sums, all_names, all_rollouts = [], [], [], [], [], [], [], [], []
    for i in range(len(teacher_counts)):
        teachers, teacher_envs, teacher_expl_strat, teacher_critics, hidden, ex_dirs, sums, names, rollouts = load_specific_packed_teachers(teacher_counts[i], env_name, suffixes[i])
        all_teachers += teachers
        all_teacher_envs += teacher_envs
        all_teacher_expl_strat += teacher_expl_strat
        all_teacher_critic += teacher_critics
        all_hidden += hidden
        all_ex_dirs += ex_dirs
        all_sums += sums
        all_names += names
        all_rollouts += rollouts
    print(f'Loaded {len(all_teachers)} teachers.')
    if teacher_count==None:
        teacher_count = len(all_teachers)
    elif len(all_teachers) > teacher_count:
        print(f'Using only the first {teacher_count} of them.')
    elif len(all_teachers) < teacher_count:
        raise pyrado.ValueErr(given=teacher_count, given_name='teacher_count', l_constraint=len(all_teachers))
    return all_teachers[:teacher_count], all_teacher_envs[:teacher_count], all_teacher_expl_strat[:teacher_count], all_teacher_critic[:teacher_count], all_hidden[:teacher_count], all_ex_dirs[:teacher_count], all_sums[:teacher_count], all_names[:teacher_count], all_rollouts[:teacher_count]


def load_specific_packed_teachers(teacher_count:int, env_name:str, suffix:str):
    pack_path = f'{pyrado.TEMP_DIR}/packs/{env_name}'
    pack = load(obj=None, name="teachers", file_ext="pkl", load_dir=pack_path, meta_info={"prefix":teacher_count, "suffix":suffix})
    teachers, teacher_envs, teacher_expl_strat, teacher_critics, hidden, ex_dirs = pack["teachers"], pack["teacher_envs"], pack["teacher_expl_strat"], pack["teacher_critics"], pack["hidden"], pack["ex_dirs"]
    # not every pack has evaluation results:
    if "sums" in pack and "names" in pack:
        sums, names = pack["sums"], pack["names"]
    else:
        sums, names = [], []
    if "rollouts" in pack:
        rollouts = pack["rollouts"]
    else:
        rollouts = []
    return teachers, teacher_envs, teacher_expl_strat, teacher_critics, hidden, ex_dirs, sums, names, rollouts


def ask_for_packlist():
    packlist = [[],[]]
    while(True):
        print('Enter teacher_count (prefix) or "e":')
        input = sys.stdin.readline().strip()
        if input=='e':
            break
        else:
            packlist[0].append(int(input))

        print('Enter suffix:')
        input = sys.stdin.readline().strip()
        packlist[1].append(input)
    print(f'You selected {len(packlist[0])} pack(s).')
    return packlist



if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--teacher_count', type=int, default=4)
    parser.add_argument('--counter', type=int, default=0)
    parser.add_argument('--evalReps', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='qq-su')
    parser.add_argument('--descr', type=str, default='')
    parser.add_argument('--packs', action='store_true', default=False)

    args = parser.parse_args()

    pack_teachers(args.teacher_count, args.env_name, f'{args.descr}{args.counter}', args.packs, args.evalReps)
    #load_teachers(args.teacher_count, args.env_name, args.packs)