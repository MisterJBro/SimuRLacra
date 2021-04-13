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
Simulate (with animation) a rollout in an environment for all policies generated by Bayesian Domain Randomization.
"""
import joblib
import numpy as np
import os
import os.path as osp
import torch as to
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import pyrado
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.logger.experiment import ask_for_experiment, load_dict_from_yaml
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.ordering import natural_sort
from pyrado.utils.input_output import print_cbt
from pyrado.utils.data_types import RenderMode


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the environment randomizer
    env_sim = joblib.load(osp.join(ex_dir, "env_sim.pkl"))
    hparam = load_dict_from_yaml(osp.join(ex_dir, "hyperparams.yaml"))

    # Override the time step size if specified
    if args.dt is not None:
        env_sim.dt = args.dt

    # Crawl through the given directory and check how many init policies and candidates there are
    found_policies, found_cands = None, None
    for root, dirs, files in os.walk(ex_dir):
        dirs.clear()  # prevents walk() from going into subdirectories
        found_policies = [p for p in files if p.endswith("_policy.pt")]
        found_cands = [c for c in files if c.endswith("_candidate.pt")]

    # Remove unwanted entries from the lists
    found_policies = [p for p in found_policies if not p.startswith("ddp_")]
    if not args.load_all:
        found_policies = [p for p in found_policies if not p.startswith("init_") and p.endswith("_policy.pt")]
        found_cands = [c for c in found_cands if not c.startswith("init_") and c.endswith("_candidate.pt")]

    # Check
    if not found_policies:
        raise pyrado.ShapeErr(msg="No policies found!")
    if not found_cands:
        raise pyrado.ShapeErr(msg="No candidates found!")
    if len(found_policies) != len(found_cands):  # don't count the final policy
        raise pyrado.ShapeErr(msg=f"Found {len(found_policies)} initial policies but {len(found_cands)} candidates!")

    # Sort
    found_policies = natural_sort(found_policies)
    found_cands = natural_sort(found_cands)

    # Plot the candidate values
    fig, ax = plt.subplots(1)
    for i in range(len(found_cands)):
        cand = to.load(osp.join(ex_dir, found_cands[i])).numpy()
        ax.scatter(np.arange(cand.size), cand, label=r"$\phi_{" + str(i) + "}$", c=f"C{i%10}", s=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("parameter value")
    ax.set_xlabel("parameter index")
    plt.legend()
    plt.show()

    # Simulate
    for i in range(len(found_policies)):
        # Load current
        policy = to.load(osp.join(ex_dir, found_policies[i]))
        cand = to.load(osp.join(ex_dir, found_cands[i]))

        # Set the domain randomizer
        if isinstance(env_sim, MetaDomainRandWrapper):
            env_sim.adapt_randomizer(cand)
            print_cbt(f"Set the domain randomizer to\n{env_sim.randomizer}", "c")
        else:
            raise pyrado.TypeErr(given=env_sim, expected_type=MetaDomainRandWrapper)

        done, state, param = False, None, None
        while not done:
            print_cbt(f"Simulating {found_policies[i]} with associated domain parameter distribution.", "g")
            ro = rollout(
                env_sim,
                policy,
                render_mode=RenderMode(video=True),
                eval=True,
                reset_kwargs=dict(domain_param=param, init_state=state),
            )  # calls env.reset()
            print_domain_params(env_sim.domain_param)
            print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
            done, state, param = after_rollout_query(env_sim, policy, ro)
