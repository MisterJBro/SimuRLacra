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
Load and run a policy on the associated real-world Quanser environment
"""
import torch as to
import pyrado
import argparse
import os
from datetime import datetime

from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleReal
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim, QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.quanser.quanser_cartpole import QCartPoleSwingUpReal, QCartPoleStabReal
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.utils import inner_env
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser
from pyrado.algorithms.policy_distillation.utils.load import load_student
from pyrado.utils.saving_loading import save


# Parameters
parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--frequency', type=int, default=250)
parser.add_argument('--env_type', type=str, default='qcp-su')
parser.add_argument('--max_steps', type=int, default=8_000)
parser.add_argument('--folder', type=str, default=None)
parser.add_argument('--animation', type=bool, default=False)
parser.add_argument('--verbose', action='store_true', default=False)

if __name__ == "__main__":

    # Parse command line arguments
    args = parser.parse_args()
    dt = 1.0/args.frequency

    student, env_sim, expl_strat, ex_dir = load_student(dt, args.env_type, args.folder, args.max_steps)
    #student = expl_strat
    print(student)
    student.net.output_scale = 0.87
    print(student.net.output_scale)

    eval_path = os.path.join(ex_dir,"eval")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # Detect the correct real-world counterpart and create it
    if isinstance(inner_env(env_sim), QBallBalancerSim):
        env_real = QBallBalancerReal(dt=dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QCartPoleSim):
        env_real = QCartPoleSwingUpReal(dt=dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QQubeSim):
        env_real = QQubeReal(dt=dt, max_steps=args.max_steps)
    else:
        raise pyrado.TypeErr(given=env_sim, expected_type=[QBallBalancerSim, QCartPoleSim, QQubeSim])

    # Wrap the real environment in the same way as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run on device
    done = False
    print_cbt("Running loaded policy ...", "c", bright=True)
    while not done:
        ro = rollout(
            env_real, student, eval=True, record_dts=True, render_mode=RenderMode(text=args.verbose, video=args.animation)
        )
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        save(ro, "deploy", "pkl", eval_path, {"suffix":datetime.now().strftime("%Y-%m-%d_%H:%M:%S")})

        done, _, _ = after_rollout_query(env_real, student, ro)
