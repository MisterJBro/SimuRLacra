import torch as to
import argparse

from copy import deepcopy
import pyrado
from torch.optim import lr_scheduler
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
#from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.domain_randomization.default_randomizers import create_default_randomizer_qcp, create_default_randomizer_qq
from pyrado.domain_randomization.domain_parameter import NormalDomainParam, UniformDomainParam, ConstantDomainParam
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper

from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO

from pyrado.algorithms.step_based.ppo_gae import PPOGAE
from pyrado.utils.argparser import get_argparser
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.environment_specific import QCartPoleSwingUpAndBalanceCtrl
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.spaces import ValueFunctionSpace

from multiprocessing import freeze_support
import multiprocessing as mp

def get_random_envs(env_count = 10, env = ActNormWrapper(QCartPoleSwingUpSim(dt=1/250)), env_type='qcp-su'):
    """Creates random environments of the given type."""

    envs = []

    # Randomizer
    if env_type == 'qcp-su':
        dp_nom = QCartPoleSwingUpSim.get_nominal_domain_param()
        randomizer = DomainRandomizer(
        NormalDomainParam(name="g", mean=dp_nom["g"], std=dp_nom["g"] / 10, clip_lo=1e-4),
        NormalDomainParam(name="m_cart", mean=dp_nom["m_cart"], std=dp_nom["m_cart"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="m_pole", mean=dp_nom["m_pole"], std=dp_nom["m_pole"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="l_rail", mean=dp_nom["l_rail"], std=dp_nom["l_rail"] / 5, clip_lo=1e-2),
        NormalDomainParam(name="l_pole", mean=dp_nom["l_pole"], std=dp_nom["l_pole"] / 5, clip_lo=1e-2),
        UniformDomainParam(name="eta_m", mean=dp_nom["eta_m"], halfspan=dp_nom["eta_m"] / 4, clip_lo=1e-4, clip_up=1),
        UniformDomainParam(name="eta_g", mean=dp_nom["eta_g"], halfspan=dp_nom["eta_g"] / 4, clip_lo=1e-4, clip_up=1),
        NormalDomainParam(name="K_g", mean=dp_nom["K_g"], std=dp_nom["K_g"] / 4, clip_lo=1e-4),
        NormalDomainParam(name="J_m", mean=dp_nom["J_m"], std=dp_nom["J_m"] / 4, clip_lo=1e-9),
        NormalDomainParam(name="r_mp", mean=dp_nom["r_mp"], std=dp_nom["r_mp"] / 5, clip_lo=1e-4),
        NormalDomainParam(name="R_m", mean=dp_nom["R_m"], std=dp_nom["R_m"] / 4, clip_lo=1e-4),
        NormalDomainParam(name="k_m", mean=dp_nom["k_m"], std=dp_nom["k_m"] / 4, clip_lo=1e-4),
        UniformDomainParam(name="B_eq", mean=dp_nom["B_eq"], halfspan=dp_nom["B_eq"] / 4, clip_lo=1e-4),
        #UniformDomainParam(name="B_pole", mean=dp_nom["B_pole"], halfspan=dp_nom["B_pole"] / 4, clip_lo=1e-4),
        # set to 0 for simpler simulation:
        ConstantDomainParam(name="B_pole", value=0.0),
    )
    elif env_type == 'qq-su':
        randomizer = create_default_randomizer_qq()

    randomizer.randomize(num_samples=env_count)
    params = randomizer.get_params(fmt="dict", dtype="numpy")

    for e in range(env_count):
        envs.append(deepcopy(env))
        print ({ key: value[e] for key, value in params.items() })
        envs[e].domain_param = { key: value[e] for key, value in params.items() }

    return envs

# Parameters
parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--hertz', type=int, default=250)
parser.add_argument('--env_type', type=str, default='qcp-su')
parser.add_argument('--env_count', type=int, default=8)
parser.add_argument('--max_steps', type=int, default=8_000)
parser.add_argument('--seed', type=int, default=None)

if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = parser.parse_args()

    # Environment
    env_hparams = dict(dt=1 / args.hertz, max_steps=args.max_steps)
    if args.env_type == 'qcp-su':
        env = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
    elif args.env_type == 'qq-su':
        env = ActNormWrapper(QQubeSwingUpSim(**env_hparams))
    else:
        print(f'Environment {args.env} does not exist!')
        quit()

    # Train all teachers
    for idx, env in enumerate(get_random_envs(env_count = args.env_count, env = env, env_type=args.env_type)):
        print(f'Training teacher: {idx}')

        # Experiment (set seed before creating the modules)
        ex_dir = setup_experiment(QCartPoleSwingUpSim.name, f"{PPOGAE.name}_{QCartPoleSwingUpAndBalanceCtrl.name}_teacher_{idx}")

        # Set seed if desired
        pyrado.set_seed(args.seed, verbose=True)

        # Policy
        policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
        policy = FNNPolicy(spec=env.spec, **policy_hparam)

        # Reduce weights of last layer, recommended by paper
        for p in policy.net.output_layer.parameters():
            with to.no_grad():
                p /= 100

        # Critic
        critic_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu)
        critic = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **critic_hparam)

        # Subroutine
        algo_hparam = dict(
            max_iter=50,
            tb_name="ppo",
            traj_len=args.max_steps,
            gamma=0.99,
            lam=0.97,
            env_num=9,
            cpu_num=min(9, mp.cpu_count()-1),
            epoch_num=40,
            device="cpu",
            max_kl=0.05,
            std_init=0.6,
            clip_ratio=0.25,
            lr=3e-3,
        )
        algo = PPOGAE(ex_dir, env, policy, critic, **algo_hparam)

        # Save the hyper-parameters
        save_dicts_to_yaml(
            dict(env=env_hparams, seed=args.seed),
            dict(policy=policy_hparam),
            dict(critic=critic_hparam),
            dict(algo=algo_hparam, algo_name=algo.name),
            save_dir=ex_dir,
        )

        # Jeeeha
        algo.train(snapshot_mode="best", seed=args.seed)

print('Finished training all teachers!')