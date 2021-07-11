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
Train an agent to solve the Quanser Qube swing-up task using Policy Distillation with Domain Randomization.

.. note::
    Call with defining --max_steps.
"""
import torch as to
import multiprocessing as mp

import pyrado
from pyrado.algorithms.meta.pddr import PDDR
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
parser.add_argument("--max_iter_teacher", type=int, default=200)
parser.add_argument("--train_teachers", action="store_true", default=False)
parser.add_argument("--num_teachers", type=int, default=2)
parser.add_argument("--max_iter", type=int, default=500)
parser.add_argument("--num_epochs", type=int, default=250)

if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = parser.parse_args()

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)
    use_cuda = args.device == "cuda"
    descr = f"_{args.max_steps}st_{args.freq}Hz_{args.num_teachers}t_{LSTMPolicy.name}"

    # Environment
    env_hparams = dict(dt=1 / args.freq, max_steps=args.max_steps)
    env_real = QQubeSwingUpSim(**env_hparams)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{PDDR.name}{descr}")

    if args.train_teachers:
        # Teacher Policy
        #teacher_policy_hparam = dict(hidden_size=64, num_recurrent_layers=1, output_nonlin=to.tanh, use_cuda=use_cuda)
        #teacher_policy = LSTMPolicy(spec=env_real.spec, **teacher_policy_hparam)
        teacher_policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu, output_nonlin=to.tanh)
        teacher_policy = FNNPolicy(spec=env_real.spec, **teacher_policy_hparam)

        # Reduce weights of last layer, recommended by paper
        for p in teacher_policy.net.output_layer.parameters():
            with to.no_grad():
                p /= 100

        # Teacher Critic
        #critic_hparam = dict(hidden_size=64, num_recurrent_layers=1, use_cuda=use_cuda)
        #critic = LSTMPolicy(spec=EnvSpec(env_real.obs_space, ValueFunctionSpace), **critic_hparam)
        critic_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu)
        critic = FNNPolicy(spec=EnvSpec(env_real.obs_space, ValueFunctionSpace), **critic_hparam)

        # Teacher subroutine
        teacher_algo_hparam = dict(
            max_iter=args.max_iter_teacher,
            tb_name="ppo",
            traj_len=args.max_steps,
            gamma=0.99,
            lam=0.97,
            env_num=30,
            cpu_num=12,  # int(mp.cpu_count()*2),
            epoch_num=40,
            device=args.device,
            max_kl=0.05,
            std_init=1.0,
            clip_ratio=0.1,
            lr=2e-3,
            critic=critic,
        )

        teacher_algo = PPOGAE

    else:
        teacher_policy = None
        teacher_algo = None
        teacher_algo_hparam = None
    print(env_real)

    # Wrapper
    randomizer = create_default_randomizer(env_real)
    env_real = DomainRandWrapperLive(env_real, randomizer)
    env_real = ActNormWrapper(env_real)

    # Student policy
    policy_hparam = dict(hidden_size=64, num_recurrent_layers=1, output_nonlin=to.tanh, use_cuda=use_cuda)
    policy = LSTMPolicy(spec=env_real.spec, **policy_hparam)

    # Subroutine
    algo_hparam = dict(
        max_iter=args.max_iter,
        min_steps=args.max_steps,
        num_cpu=20,  # int(mp.cpu_count()/2),
        std_init=0.1,
        num_epochs=args.num_epochs,
        num_teachers=args.num_teachers,
        device=args.device,
        teacher_policy=teacher_policy,
        teacher_algo=teacher_algo,
        teacher_algo_hparam=teacher_algo_hparam,
    )

    algo = PDDR(ex_dir, env_real, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="best", seed=args.seed)
