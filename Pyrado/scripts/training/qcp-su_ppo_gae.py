"""
Train an agent to solve the Quanser Cart-Pole swing-up task using Proximal Policy Optimization.
"""
import torch as to
from multiprocessing import freeze_support

import pyrado
from pyrado.algorithms.step_based.ppo_gae import PPOGAE
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.environment_specific import QCartPoleSwingUpAndBalanceCtrl
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from pyrado.spaces import ValueFunctionSpace
import multiprocessing as mp


if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QCartPoleSwingUpSim.name, f"{PPOGAE.name}_{QCartPoleSwingUpAndBalanceCtrl.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 250.0, max_steps=1500)
    env = ActNormWrapper(QCartPoleSwingUpSim(**env_hparams))
    print(env)

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Reduce weights of last layer, recommended by paper
    for p in policy.net.output_layer.parameters():
        with to.no_grad():
            p /= 100

    # Critic
    critic_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.exp)
    critic = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **critic_hparam)

    # Subroutine
    algo_hparam = dict(
        max_iter=100,
        tb_name="ppo",
        traj_len=1500,
        gamma=0.99,
        lam=0.97,
        env_num=30,
        cpu_num=min(9,mp.cpu_count()-1),
        epoch_num=40,
        device="cpu",
        max_kl=0.05,
        std_init=1.0,
        clip_ratio=0.1,
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
    