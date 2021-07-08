"""
Train an agent to solve the Quanser Qube swing-up task using Proximal Policy Optimization with Generalized Advantage Estimation.
"""
import torch as to
from multiprocessing import freeze_support

import pyrado
from pyrado.algorithms.step_based.ppo_gae import PPOGAE
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.utils.data_types import RenderMode
from pyrado.policies.recurrent.rnn import LSTMPolicy
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.experiments import load_experiment
from pyrado.logger.experiment import Experiment, ask_for_experiment
import multiprocessing as mp
import numpy as np

if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = get_argparser().parse_args()

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)
    use_cuda = args.device == "cuda"

    freq = 500.0

    for _ in range(1000):
        max_iter = np.random.choice([100, 200, 300])  # 150 200 300
        traj_len = np.random.choice([4000, 8000, 16000])  #  4000, 8000, 16000
        gamma = np.random.choice(np.arange(0.95, 1, 0.001))  #   0.95 .. < 1
        lam = np.random.choice(np.arange(0.8, 1, 0.001))  #   0.8 .. < 1
        env_num = (np.random.choice([32, 64, 128]),)  #   64
        epoch_num = np.random.choice([10, 20, 40, 80])  #   10, 20, 40, 80
        max_kl = np.random.choice(np.arange(0.01, 0.4, 0.01))  #   0.01 .. 0.4
        std_init = np.random.choice(np.arange(0.1, 2.0, 0.1))  #   0.1 .. 2.0
        clip_ratio = np.random.choice(np.arange(0.05, 0.3, 0.02))  #   0.05 .. 0.3
        lr = np.random.choice(np.arange(1e-4, 2e-3, 1e-4))  #   1e-4 .. 2e-3

        # Experiment (set seed before creating the modules)
        ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{PPOGAE.name}_{FNNPolicy.name}", f"{freq}Hz_seed_{args.seed}")

        # Environment
        env_hparams = dict(dt=1 / freq, max_steps=args.max_steps)
        env = ActNormWrapper(QQubeSwingUpSim(**env_hparams))

        # Policy
        # policy_hparam = dict(hidden_size=64, num_recurrent_layers=1, output_nonlin=to.tanh, use_cuda=use_cuda)
        # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
        policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
        policy = FNNPolicy(spec=env.spec, **policy_hparam)

        # Reduce weights of last layer, recommended by paper
        for p in policy.net.output_layer.parameters():
            with to.no_grad():
                p /= 100

        # Critic
        # critic_hparam = dict(hidden_size=64, num_recurrent_layers=1, output_nonlin=to.exp, use_cuda=use_cuda)
        # critic = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **critic_hparam)
        critic_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu)
        critic = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **critic_hparam)

        # Subroutine
        algo_hparam = dict(
            max_iter=max_iter,  # 150 200 300                    
            tb_name="ppo",
            traj_len=traj_len,  #  4000, 8000, 16000
            gamma=gamma,  #   0.95 .. < 1
            lam=lam,  #   0.8 .. < 1
            env_num=env_num,  #   64
            cpu_num=args.num_cpus,  #   8
            epoch_num=epoch_num,  #   10, 20, 40, 80
            device=args.device,  #   cpu
            max_kl=max_kl,  #   0.01 .. 0.4
            std_init=std_init,  #   0.1 .. 2.0
            clip_ratio=clip_ratio,  #   0.05 .. 0.3
            lr=lr,  #   1e-4 .. 2e-3
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

        # Test policy
        while True:
            input("Press some key to continue:")
            ro = rollout(env, algo.policy, render_mode=RenderMode(text=True, video=True))

            print(f"Return: {ro.undiscounted_return()}")
