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

import os
from copy import deepcopy

import numpy as np
import torch as to

import pyrado
from pyrado.algorithms.base import Algorithm, InterruptableAlgorithm
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.experiment import Experiment, ask_for_experiment
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.utils.experiments import load_experiment
from pyrado.sampling.envs import Envs
from pyrado.algorithms.step_based.ppo_gae import PPOGAE


class P2PDRL(PPOGAE):
    """Online Peer-to-Peer Distillation Reinforcement Learning (P2PDRL). https://arxiv.org/pdf/2012.04839v1.pdf."""

    name: str = "p2pdrl"

    def __init__(
        self,
        save_dir: str,
        env: Env,
        policy: Policy,
        critic: Policy,
        max_iter: int,
        tb_name: str = "p2pdrl",
        traj_len: int = 8_000,
        gamma: float = 0.99,
        lam: float = 0.97,
        env_num: int = 9,
        cpu_num: int = 3,
        epoch_num: int = 40,
        device: str = "cpu",
        max_kl: float = 0.05,
        std_init: float = 0.6,
        clip_ratio: float = 0.25,
        lr: float = 3e-3,
        logger: StepLogger = None,
        early_stopping: bool = False,
        alpha: float = 0.1,
        num_workers: int = 8,
        weight_decay: float = 1e-5,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        :param device: device to use for updating the policy (cpu or gpu)
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                    By default, the learning rate is constant.
        :param std_init: initial standard deviation on the actions for the exploration noise
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_epochs: number of epochs (how often we iterate over the same batch)
        :param max_iter: number of iterations (policy updates)
        :param num_teachers: number of teachers that are used for distillation
        :param num_cpu: number of cpu cores to use
        :param teacher_extra: extra dict from PDDRTeachers algo. If provided, teachers are loaded from there
        :param teacher_policy: policy to be updated (is duplicated for each teacher)
        :param teacher_algo: algorithm class to be used for training the teachers
        :param teacher_algo_hparam: hyperparams to be used for teacher_algo
        :param randomizer: randomizer for sampling the teacher domain parameters. If None, the default one for env is used
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(policy, Policy):
            raise pyrado.TypeErr(given=policy, expected_type=Policy)

        # Call Algorithm's constructor.
        super().__init__(
            save_dir=save_dir,
            env=env,
            policy=policy,
            critic=critic,
            max_iter=max_iter,
            tb_name=tb_name,
            traj_len=traj_len,
            gamma=gamma,
            lam=lam,
            env_num=env_num,
            cpu_num=cpu_num,
            epoch_num=epoch_num,
            device=device,
            max_kl=max_kl,
            std_init=std_init,
            clip_ratio=clip_ratio,
            lr=lr,
            logger=logger,
            early_stopping=early_stopping,
        )

        # Store the inputs
        self.num_workers = num_workers
        self.device = device
        self.alpha = alpha

        self.workers_policies = []
        self.worker_envs = []
        self.worker_expl_strats = []
        self.worker_critics = []
        self.worker_ex_dirs = []
        self.worker_optimizer = []

        # Add Worker policies, critics, optimizer and expl strats
        for _ in range(self.num_workers):
            p = deepcopy(policy)
            expl = NormalActNoiseExplStrat(p, std_init=std_init)
            c = deepcopy(critic)
            o = to.optim.Adam(
            [
                {"params": p.parameters()},
                {"params": expl.parameters()},
                {"params": c.parameters()},
            ],
            lr=lr,
            weight_decay=weight_decay,
            )

            self.worker_policies.append(p)
            self.worker_expl_strats.append(expl)
            self.worker_critics.append(c)
            self.worker_optimizer.append(o)

        # Environments
        self.envs = None

        # Distillation loss criterion
        self.criterion = to.nn.KLDivLoss(log_target=True, reduction="batchmean")

        print('LIEF DURCH BRUDAH')

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self.worker_expl_strats[0]

    @property
    def policy(self) -> Policy:
        return self.worker_policies[0]

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Performs a single iteration of the algorithm. This includes collecting the data, updating the parameters, and
        adding the metrics of interest to the logger. Does not update the `curr_iter` attribute.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

        # Set envs
        self.set_envs()

        # Sample batch
        rets = self.sample_batch()

        # Log current progress
        self.logger.add_value("max return", np.max(rets), 4)
        self.logger.add_value("median return", np.median(rets), 4)
        self.logger.add_value("avg return", np.mean(rets), 4)
        self.logger.add_value("min return", np.min(rets), 4)
        self.logger.add_value("std return", np.std(rets), 4)
        self.logger.add_value("std var", self.expl_strat.std.item(), 4)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, np.mean(rets), meta_info)

        # Update policy and value function
        for i in range(self.num_workers):
            self.update(i)

    def update(self, idx):
        """Update one policy using PPO."""
        obs, act, rew, ret, adv, dones = self.envs.get_data(self.device)
        
        print(obs.shape, act.shape, rew.shape, ret.shape)
        # Use only the respective observations 
        obs = obs.reshape(self.num_teachers, self.min_steps, -1)
        obs = obs[idx]

        # For recurrent pack observations
        if self.workers_policies[idx].is_recurrent:
            obs = obs.reshape(-1, self.traj_len, obs.shape[-1])
            obs_list = []
            lengths = []
            for idx, section in enumerate(dones):
                start = 0
                for end in section[1:]:
                    obs_list.append(obs[idx, start:end])
                    lengths.append(end - start)
                    start = end
                if start != self.traj_len:
                    obs_list.append(obs[idx, start:])
                    lengths.append(self.traj_len - start)
            obs = to.nn.utils.rnn.pad_sequence(obs_list)
            obs = to.nn.utils.rnn.pack_padded_sequence(obs, lengths=lengths, enforce_sorted=False)

        with to.no_grad():
            if self.workers_policies[idx].is_recurrent:
                mean, _ = self.workers_policies[idx].rnn_layers(obs)
                mean, lens = to.nn.utils.rnn.pad_packed_sequence(mean)
                mean = to.cat([mean[:l, i] for i, l in enumerate(lens)], 0)

                mean = self.workers_policies[idx].output_layer(mean)
                if self.workers_policies[idx].output_nonlin is not None:
                    mean = self.workers_policies[idx].output_nonlin(mean)
            else:
                mean = self.workers_policies[idx](obs)
            old_logp = self.worker_expl_strats[idx].action_dist_at(mean).log_prob(act).sum(-1)

        for i in range(self.epoch_num):
            self.worker_optimizer[idx].zero_grad()

            # Policy
            if self.workers_policies[idx].is_recurrent:
                mean, _ = self.workers_policies[idx].rnn_layers(obs)
                mean, lens = to.nn.utils.rnn.pad_packed_sequence(mean)
                mean = to.cat([mean[:l, i] for i, l in enumerate(lens)], 0)

                mean = self.workers_policies[idx].output_layer(mean)
                if self.workers_policies[idx].output_nonlin is not None:
                    mean = self.workers_policies[idx].output_nonlin(mean)
            else:
                mean = self.workers_policies[idx](obs)
            dist = self.worker_expl_strats[idx].action_dist_at(mean)

            # Critic
            if self.workers_critics[idx].is_recurrent:
                val, _ = self.workers_critics[idx].rnn_layers(obs)
                val, lens = to.nn.utils.rnn.pad_packed_sequence(val)
                val = to.cat([val[:l, i] for i, l in enumerate(lens)], 0)

                val = self.workers_critics[idx].output_layer(val)
                if self.workers_critics[idx].output_nonlin is not None:
                    val = self.workers_critics[idx].output_nonlin(val)
            else:
                val = self.workers_critics[idx](obs)
            val = val.reshape(-1)

            logp = dist.log_prob(act).sum(-1)
            loss_policy, kl = self.loss_fcn(logp, old_logp, adv)

            # Early stopping if kl divergence too high
            if kl > self.max_kl:
                return
            loss_value = self.criterion(val, ret)

            # Distillation loss
            loss_dis = 0
            for i in range(self.num_workers):
                if i is not idx:
                    loss_dis += self.criterion(self.workers_policies[i], self.workers_policies[idx])
            loss_dis /= (self.num_workers-1)

            loss = loss_policy + loss_value + self.alpha * loss_dis
            loss.backward()

            self.optimizer.step()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        pyrado.save(self.policy, "policy.pt", self.save_dir)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self.env_real, "env.pkl", self.save_dir)

    def __getstate__(self):
        # Remove the unpickleable elements from this algorithm instance
        tmp_teacher_policies = self.__dict__.pop("teacher_policies")
        tmp_teacher_algo = self.__dict__.pop("teacher_algo")
        tmp_envs = self.__dict__.pop("envs")

        # Call Algorithm's __getstate__() without the unpickleable elements
        state_dict = super(P2PDRL, self).__getstate__()

        # Make a deep copy of the state dict such that we can return the pickleable version
        state_dict_copy = deepcopy(state_dict)

        # Insert them back
        self.__dict__["teacher_policies"] = tmp_teacher_policies
        self.__dict__["teacher_algo"] = tmp_teacher_algo
        self.__dict__["envs"] = tmp_envs

        return state_dict_copy

    def __setstate__(self, state):
        # Call Algorithm's __setstate__()
        super().__setstate__(state)

        # Recover settings of environment
        self.envs = Envs(
            state["cpu_num"], state["env_num"], state["env"], state["traj_len"], state["gamma"], state["lam"], state["worker_envs"]
        )

    def set_envs(self):
        """Creates random environments of the given type."""
        self.randomizer.randomize(num_samples=self.num_workers)
        params = self.randomizer.get_params(fmt="dict", dtype="numpy")
        self.worker_envs = []
        for e in range(self.num_teachers):
            self.worker_envs.append(deepcopy(self.env))
            print({key: value[e] for key, value in params.items()})
            self.worker_envs[e].domain_param = {key: value[e] for key, value in params.items()}

        self.envs = Envs(
            min(self.num_cpu, self.num_workers),
            self.num_teachers,
            self.env,
            self.min_steps,
            self.algo.gamma,
            self.algo.lam,
            env_list=self.worker_envs,
        )
