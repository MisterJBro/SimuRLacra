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

from typing import List

import os
import torch as to

import pyrado
from pyrado.policies.base import Policy
from pyrado.logger.experiment import Experiment, ask_for_experiment
from pyrado.utils.experiments import load_experiment
from pyrado.utils.data_types import EnvSpec


class EnsemblePolicy(Policy):
    """Ensemble of policy forming a new policy."""

    name: str = "ens"

    def __init__(
        self,
        spec: EnvSpec,
        num_policies: int,
        policies: List = [],
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param hidden_sizes: sizes of hidden layer outputs. Every entry creates one hidden layer.
        :param hidden_nonlin: nonlinearity for hidden layers
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param output_nonlin: nonlinearity for output layer
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)
        
        # Init parameters
        self.policies = []
        self.ex_dirs = []
        self.num_policies = num_policies
        self.hidden = [None for _ in range(self.num_policies)]

        # If policies are already present, use them
        if len(policies) == self.num_policies:
            self.policies = policies
        else:
            # or else load them
            self.load_policies()
        
    def load_policies(self):
        """Recursively load all polciies that can be found in the current experiment's directory."""
        # Get the experiment's directory to load from
        ex_dir = ask_for_experiment(max_display=75, perma=False)
        self.load_policy_experiment(ex_dir)
        if len(self.policies) < self.num_policies:
            print(
                f"You have loaded {len(self.policies)} teachers - load at least {self.num_policies - len(self.policies)} more!"
            )
            self.load_policies()

    def load_policy_experiment(self, exp: Experiment):
        """
        Load policy from PDDRPolicy experiment.
        
        :param exp: the policy's experiment object
        """
        _, _, extra = load_experiment(exp)
        
        self.unpack_policies(extra)

    def unpack_policies(self, extra: dict):
        """
        Unpack policy from PDDRPolicy experiment.

        :param extra: dict with policies data
        """
        num_policy_to_load = self.num_policies - len(self.policies)
        self.ex_dirs.extend(extra["teacher_ex_dirs"][: num_policy_to_load])
        for dir in self.ex_dirs:
            dir = os.path.join(pyrado.TEMP_DIR, dir[dir.find("/data/temp/")+len("/data/temp/"):])
            env, policy, extra = load_experiment(dir)
            self.policies.append(policy)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor) -> to.Tensor:
        # Get action from each policies
        acts = []

        for i, p in enumerate(self.policies):
            if p.is_recurrent:
                act, self.hidden[i] = p(obs, self.hidden[i])
            else:
                act = p(obs)
            acts.append(act)

        # Combine all actions
        acts = to.stack(acts, 0)
        act = acts.mean(0)

        return act
