import multiprocessing
import numpy as np
from numpy.lib.function_base import iterable
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
import torch as to
import multiprocessing as mp
from multiprocessing import freeze_support
from pyrado.utils.saving_loading import save
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
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from copy import deepcopy

import argparse
from datetime import datetime

def check_performance_raw(env, policy, name, n=1000, path='', verbose=True, max_steps=None):
    print('Started checking performance.')
    start=datetime.now()
    su = []
    ros = []
    # Test the policy in the environment
    done, param, state = False, None, None
    for i in range(n):
        if verbose:
            print('rollout', i, '/', n-1)
        ro = rollout(
            env,
            policy,
            max_steps=max_steps,
            render_mode=RenderMode(text=False, video=False),
            eval=True,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        su.append(ro.undiscounted_return())
        ros.append(ro)
        if verbose:
            print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, param, state = False, None, None
    sums = np.array(su)
    print('Endsumme (' + name + ' from', n, 'reps ): MEAN =', np.mean(sums), 'STD =', np.std(sums),
          'MIN =', np.min(sums), 'MAX =', np.max(sums), 'MEDIAN =', np.median(sums))
    save_performance(start, sums, ros, name, path)
    return sums, ros


def save_performance(start, sums, rollouts, name, path):
    save(obj=rollouts, name=f"{name}_rollouts.pkl", save_dir=path, suffix=start.strftime("%Y-%m-%d_%H:%M:%S"))
    save(obj=np.array([name]), name=f"{name}_names.pkl", save_dir=path, suffix=start.strftime("%Y-%m-%d_%H:%M:%S"))
    save(obj=sums, name=f"{name}_sums.pkl", save_dir=path, suffix=start.strftime("%Y-%m-%d_%H:%M:%S"))


def eval_student_on_random_envs(policy, env, count:int, eval_dir, iters, max_eval_steps=None, verbose=False):
    randomizer = create_default_randomizer(env)
    randomizer.randomize(num_samples=count)
    params = randomizer.get_params(fmt="dict", dtype="numpy")
    random_envs = []
    for e in range(count):
        random_envs.append(deepcopy(env))
        print({key: value[e] for key, value in params.items()})
        random_envs[e].domain_param = {key: value[e] for key, value in params.items()}

    names=[ f'student_on_random_env_{idx}' for idx in range(count) ]
    save(obj=[e.domain_param for e in random_envs], name="random_env_params.pkl", save_dir=eval_dir, suffix=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    a_pool = multiprocessing.Pool(processes=4)
    su = a_pool.starmap_async(check_performance_raw, [(env, deepcopy(policy), names[idx], iters, eval_dir, verbose, max_eval_steps) for idx, env in enumerate(random_envs)]).get()
    a_pool.close()
    a_pool.join()
    print('Finished evaluating the student on random environments!')

    # Check student performance on teacher envs:
    for env in random_envs:
        env.close()
    
    sums = su[0]
    rollouts = su[1]
    return sums, names, rollouts 


def eval_teachers_on_student_env(env, policies, ex_dirs, iters, max_steps, verbose):
    print("Started evaluating all teachers.")
    names=[ f'teacher_on_student {t}' for t in range(len(policies)) ]
    eval_dirs = []
    for dir in ex_dirs:
        eval_dir = os.path.join(dir,'eval')
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        eval_dirs.append(eval_dir)

    a_pool = mp.pool.ThreadPool(processes=mp.cpu_count())
    print(policies[0])
    print(names[0])
    print(eval_dirs[0])
    check_performance_raw(deepcopy(env), policies[0], names[0], iters, eval_dirs[0], verbose, max_steps)
    #su = a_pool.starmap_async(check_performance_raw, [(deepcopy(env), policies[idx], names[idx], iters, eval_dirs[idx], verbose, max_steps) for idx in range(len(policies))]).get()
    a_pool.close()
    a_pool.join()
    print('Finished evaluating all teachers!')
    sums = su[0]
    rollouts = su[1]
    return sums, names, rollouts 


def eval_teachers_on_teacher_envs(envs, policies, ex_dirs, iters, max_steps, verbose):
    print("Started evaluating all teachers.")
    names=[ f'teacher {t}' for t in range(len(policies)) ]
    eval_dirs = []
    for dir in ex_dirs:
        eval_dir = os.path.join(dir,'eval')
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        eval_dirs.append(eval_dir)

    a_pool = mp.pool.ThreadPool(processes=mp.cpu_count())
    su = a_pool.starmap_async(check_performance_raw, [(envs[idx], policies[idx], names[idx], iters, eval_dirs[idx], verbose, max_steps) for idx in range(len(policies))]).get()
    a_pool.close()
    a_pool.join()
    print('Finished evaluating all teachers!')
    sums = su[0]
    rollouts = su[1]
    return sums, names, rollouts 

if __name__ == "__main__":
    freeze_support()
    to.set_default_dtype(to.float32)
    device = to.device('cuda')

    # Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher_eval', action='store_true', default=False)
    parser.add_argument('--student_random_envs', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--env_name', type=str, default='qq-su')

    # Parse command line arguments
    args = parser.parse_args()

    ex_dir = ask_for_experiment(max_display=150, env_name=args.env_name, perma=False)
    env_sim, student_policy, extra = load_experiment(ex_dir)
    eval_dir = os.path.join(ex_dir,'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    if args.student_random_envs > 0:
        eval_student_on_random_envs(student_policy, env_sim, args.student_random_envs, eval_dir, args.iters, args.max_steps, args.verbose)
        exit()
    if args.teacher_eval:
        #teacher_policies, teacher_envs, teacher_expl_strats, teacher_critics, teacher_ex_dirs = extra["teacher_policies"], extra["teacher_envs"], extra["teacher_expl_strats"], extra["teacher_critics"], extra["teacher_ex_dirs"]
        teacher_ex_dirs = extra["teacher_ex_dirs"]
        teacher_envs, teacher_policies = [], []
        for dir in teacher_ex_dirs:
            teacher_env, teacher_policy, _ = load_experiment(dir)
            teacher_envs.append(teacher_env)
            teacher_policies.append(teacher_policy)

        eval_teachers_on_student_env(env_sim, teacher_policies, teacher_ex_dirs, args.iters, args.max_steps, args.verbose)
        #eval_teachers_on_teacher_envs(teacher_envs, teacher_policies, teacher_ex_dirs, args.iters, args.max_steps, args.verbose)
    
    