# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Train an agent to solve the Quanser Qube swing-up task using Online Peer-to-Peer Distillation Reinforcement Learning (P2PDRL).

.. note::
    Call with defining --max_steps.
"""
import torch as to
import multiprocessing as mp

import pyrado
from pyrado.algorithms.step_based.p2pdrl import P2PDRL
from pyrado.algorithms.step_based.ppo_gae import PPOGAE
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.recurrent.rnn import LSTMPolicy
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from multiprocessing import freeze_support

parser = get_argparser()
parser.add_argument("--freq", type=int, default=500)
parser.add_argument("--max_iter", type=int, default=200)
parser.add_argument("--num_epochs", type=int, default=250)

if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = parser.parse_args()

     # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{P2PDRL.name}")
    
    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)
    use_cuda = args.device == "cuda"
    descr = f"_{args.max_steps}st_{args.freq}Hz_{args.num_workers}t_{P2PDRL.name}"
    
    # Environment
    env_hparams = dict(dt=1 / args.freq, max_steps=args.max_steps)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

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

    # Worker subroutine
    algo_hparam = dict(
        max_iter=args.max_iter,
        tb_name="p2pdrl",
        traj_len=args.max_steps,
        gamma=0.99,
        lam=0.97,
        env_num=32,
        cpu_num=args.num_cpus,
        epoch_num=40,
        device=args.device,
        max_kl=0.05,
        std_init=1.0,
        clip_ratio=0.1,
        lr=2e-3,
        alpha= 0.1,
        num_workers=args.num_workers,
    )
    algo = P2PDRL(ex_dir, env, policy, critic, **algo_hparam)
    
    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="best", seed=args.seed)
