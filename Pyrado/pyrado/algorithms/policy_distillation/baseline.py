"""
Baseline for policy distillation. Baseline model using domain randomization and ppo.
"""
import torch as to
from multiprocessing import freeze_support

import pyrado
from pyrado.algorithms.step_based.ppo_gae import PPOGAE
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.environment_specific import QCartPoleSwingUpAndBalanceCtrl
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from pyrado.spaces import ValueFunctionSpace
import multiprocessing as mp
import argparse

# Parameters
parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--frequency', type=int, default=250)
parser.add_argument('--env_type', type=str, default='qcp-su')
parser.add_argument('--max_steps', type=int, default=8_000)
parser.add_argument('--seed', type=int, default=None)

if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = parser.parse_args()

    # Environment
    env_hparams = dict(dt=1 / args.frequency, max_steps=args.max_steps)
    
    # Experiment (set seed before creating the modules)
    if args.env_type == 'qcp-su':
        ex_dir = setup_experiment(QCartPoleSwingUpSim.name, f"{PPOGAE.name}_baseline_{QCartPoleSwingUpAndBalanceCtrl.name}_{args.frequency}Hz")
        env = QCartPoleSwingUpSim(**env_hparams)
    elif args.env_type == 'qq-su':
        ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{PPOGAE.name}_baseline_{QQubeSwingUpSim.name}_{args.frequency}Hz")
        env = QQubeSwingUpSim(**env_hparams)
    elif args.env_type == 'qbb':
        ex_dir = setup_experiment(QBallBalancerSim.name, f"{PPOGAE.name}_baseline_{QBallBalancerSim.name}_{args.frequency}Hz")
        env = QBallBalancerSim(**env_hparams)

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Domain Randomization
    #randomizer = create_default_randomizer(env)
    #env = DomainRandWrapperLive(env, randomizer)
    env = ActNormWrapper(env)
    print(env)

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.nn.ReLU(), output_nonlin=to.tanh, output_scale=0.8)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Reduce weights of last layer, recommended by paper
    for p in policy.net.output_layer.parameters():
        with to.no_grad():
            p /= 100

    # Critic
    critic_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.nn.ReLU())
    critic = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **critic_hparam)

    # Subroutine
    algo_hparam = dict(
        max_iter=100,
        tb_name="ppo",
        traj_len=16_000,
        gamma=0.99,
        lam=0.97,
        env_num=9,
        cpu_num=min(9,mp.cpu_count()-1),
        epoch_num=40,
        device="cpu",
        max_kl=0.05,
        std_init=0.6,
        clip_ratio=0.2,
        lr=2e-3,
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
