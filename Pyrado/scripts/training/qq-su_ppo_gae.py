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


if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{PPOGAE.name}_{FNNPolicy.name}", f"250Hz_seed_{args.seed}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True) 
    use_cuda = args.device == "cuda"
 
    # Environment
    env_hparams = dict(dt=1 / 250.0, max_steps=4000)
    env = ActNormWrapper(QQubeSwingUpSim(**env_hparams))

    # Policy
    policy_hparam = dict(hidden_size=64, num_recurrent_layers=1, output_nonlin=to.tanh, use_cuda=use_cuda)
    policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    #policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
    #policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Reduce weights of last layer, recommended by paper
    for p in policy.output_layer.parameters():
        with to.no_grad():
            p /= 100

    # Critic
    critic_hparam = dict(hidden_size=64, num_recurrent_layers=1, output_nonlin=to.exp, use_cuda=use_cuda)
    critic = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **critic_hparam)
    #critic_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.exp)
    #critic = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **critic_hparam)

    #ex_dir2 = ask_for_experiment()
    #env, policy, extra = load_experiment(ex_dir2)
    #critic = extra["vfcn"]

    # Subroutine
    algo_hparam = dict(
        max_iter=100,
        tb_name="ppo",
        traj_len=4000,
        gamma=0.99,
        lam=0.97,
        env_num=30,
        cpu_num=3,
        epoch_num=40,
        device=args.device,
        max_kl=0.05,
        std_init=0.2,
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
    
    # Test policy
    while True:
        input('Press some key to continue:')
        ro = rollout(
            env,
            algo.policy,
            render_mode=RenderMode(text=True, video=True)
        )

        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)