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
from typing import Any, List, Tuple

import numpy as np
import torch as to
import multiprocessing as mp

import pyrado
from pyrado.algorithms.base import Algorithm, InterruptableAlgorithm
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.experiment import Experiment, ask_for_experiment
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.sampling.envs import Envs


class PDDR(InterruptableAlgorithm):
    """Policy Distillation with Domain Randomization (PDDR)"""

    name: str = "pddr"

    def __init__(
        self,
        save_dir: str,
        env: Env,
        policy: Policy,
        logger: StepLogger = None,
        device: str = "cpu",
        lr: float = 5e-4,
        std_init: float = 0.1,
        min_steps: int = 4000,
        num_epochs: int = 10,
        max_iter: int = 500,
        num_teachers: int = 8,
        num_cpu: int = 3,
        teacher_extra: dict = None,
        teacher_policy: Policy = None,
        teacher_algo: callable = None,
        teacher_algo_hparam: dict() = None,
        randomizer: DomainRandomizer = None,
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
            num_checkpoints=1, init_checkpoint=-1, save_dir=save_dir, max_iter=max_iter, policy=policy, logger=logger
        )

        # Store the inputs
        self.min_steps = min_steps
        self.num_epochs = num_epochs
        self.num_teachers = num_teachers
        self.num_cpu = num_cpu
        self.device = device
        self.env_real = env
        self.max_iter = max_iter

        self.teacher_algo = teacher_algo
        self.teacher_algo_hparam = teacher_algo_hparam
        self.teacher_policy = teacher_policy
        self.teacher_policies = []
        self.teacher_envs = []
        self.teacher_expl_strats = []
        self.teacher_critics = []
        self.teacher_ex_dirs = []

        # Teachers
        if teacher_policy is not None and teacher_algo is not None and teacher_algo_hparam is not None:
            if not isinstance(teacher_policy, Policy):
                raise pyrado.TypeErr(given=teacher_policy, expected_type=Policy)
            if not issubclass(teacher_algo, Algorithm):
                raise pyrado.TypeErr(given=teacher_algo, expected_type=Algorithm)

            if randomizer is None:
                self.randomizer = create_default_randomizer(env)
            else:
                assert isinstance(randomizer, DomainRandomizer)
                self.randomizer = randomizer

            self.set_random_envs()

            # Prepare folders
            self.teacher_ex_dirs = [os.path.join(self.save_dir, f"teacher_{idx}") for idx in range(self.num_teachers)]
            for idx in range(self.num_teachers):
                os.makedirs(self.teacher_ex_dirs[idx], exist_ok=True)
        elif teacher_extra is not None:
            self.unpack_teachers(teacher_extra)
            assert self.num_teachers == len(self.teacher_policies)
            self.reset_checkpoint()
        else:
            self.load_teachers()
            if self.num_teachers < len(self.teacher_policies):
                print(
                    f"You have loaded {len(self.teacher_policies)} teachers. Only the first {self.num_teachers} will be used!"
                )
                # self.prune_teachers()
            assert self.num_teachers == len(self.teacher_policies)
            self.reset_checkpoint()

        # Student
        self._expl_strat = NormalActNoiseExplStrat(self._policy, std_init=std_init)
        self._policy = self._policy.to(self.device)
        self.optimizer = to.optim.Adam(
            [
                {"params": self.policy.parameters()},
                {"params": self._expl_strat.noise.parameters()},
            ],
            lr=lr,
            weight_decay=1e-5,
        )

        # Environments
        self.envs = Envs(self.num_cpu, self.num_teachers, env, min_steps, 0.99, 0.97, env_list=self.teacher_envs)

        # Distillation loss criterion
        self.criterion = to.nn.KLDivLoss(log_target=True, reduction="batchmean")

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self._expl_strat

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

        if self.curr_checkpoint == -1:
            self.train_teachers(snapshot_mode, None)
            self.reached_checkpoint()  # setting counter to 0

        if self.curr_checkpoint == 0:
            # Sample batch
            rets, all_lengths = self.sample()

            # Log current progress
            self.logger.add_value("max return", np.max(rets), 4)
            self.logger.add_value("median return", np.median(rets), 4)
            self.logger.add_value("avg return", np.mean(rets), 4)
            self.logger.add_value("min return", np.min(rets), 4)
            self.logger.add_value("std return", np.std(rets), 4)
            self.logger.add_value("std var", self.expl_strat.std.item(), 4)
            self.logger.add_value("avg rollout len", np.mean(all_lengths), 4)
            self.logger.add_value("num total samples", np.sum(all_lengths))

            # Save snapshot data
            self.make_snapshot(snapshot_mode, np.mean(rets), meta_info)

            # Update policy and value function
            self.update()

    def reset_states(self, env_indices=[]):
        """
        Resets the hidden states.
        :param env_indices: indices of the environment hidden states to reset. If empty, reset all.
        """
        num_env_ind = len(env_indices)
        if self.policy.is_recurrent:
            if num_env_ind == 0:
                self.hidden_policy = to.zeros(
                    self.num_teachers, self.policy.hidden_size, device=self.device
                ).contiguous()
            else:
                self.hidden_policy[env_indices] = to.zeros(
                    num_env_ind, self.policy.hidden_size, device=self.device
                ).contiguous()

    def sample(self) -> np.ndarray:
        """Sample batch of trajectories for training."""
        obss = self.envs.reset()
        self.reset_states()

        for _ in range(self.min_steps):
            obss = to.as_tensor(obss).to(self.device)
            with to.no_grad():
                if self.expl_strat.is_recurrent:
                    acts, self.hidden_policy = self.expl_strat(obss, self.hidden_policy.contiguous())
                else:
                    acts = self.expl_strat(obss)
                acts = acts.cpu().numpy()
            obss, done_ind = self.envs.step(acts, np.zeros_like(acts).reshape(-1))
            if len(done_ind) != 0:
                self.reset_states(done_ind)

        rets = self.envs.ret_and_adv()
        return rets

    def update(self):
        """Update the policy's (and value functions') parameters based on the collected rollout data."""
        obs, act, rew, ret, adv, dones = self.envs.get_data(self.device)
        obs = obs.reshape(self.num_teachers, self.min_steps, -1)

        # Teacher observation
        teacher_obs = []
        for t_idx, teacher in enumerate(self.teacher_policies):
            if teacher.is_recurrent:
                obs_list = []
                lengths = []
                start = 0
                for end in dones[t_idx][1:]:
                    obs_list.append(obs[t_idx, start:end].clone())
                    lengths.append(end - start)
                    start = end
                if start != self.min_steps:
                    obs_list.append(obs[t_idx, start:].clone())
                    lengths.append(self.min_steps - start)
                obs_pad = to.nn.utils.rnn.pad_sequence(obs_list)
                obs_pack = to.nn.utils.rnn.pack_padded_sequence(obs_pad, lengths=lengths, enforce_sorted=False)
                teacher_obs.append(obs_pack)
            else:
                teacher_obs.append(obs[t_idx].clone())

        # For recurrent pack observations
        if self.policy.is_recurrent:
            obs = obs.reshape(-1, self.min_steps, obs.shape[-1])
            obs_list = []
            lengths = []
            for idx, section in enumerate(dones):
                start = 0
                for end in section[1:]:
                    obs_list.append(obs[idx, start:end])
                    lengths.append(end - start)
                    start = end
                if start != self.min_steps:
                    obs_list.append(obs[idx, start:])
                    lengths.append(self.min_steps - start)
            obs = to.nn.utils.rnn.pad_sequence(obs_list)
            obs = to.nn.utils.rnn.pack_padded_sequence(obs, lengths=lengths, enforce_sorted=False)

        # Train student
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            # Student actions
            if self.policy.is_recurrent:
                mean, _ = self.policy.rnn_layers(obs)
                mean, lens = to.nn.utils.rnn.pad_packed_sequence(mean)
                mean = to.cat([mean[:l, i] for i, l in enumerate(lens)], 0)

                mean = self.policy.output_layer(mean)
                if self.policy.output_nonlin is not None:
                    mean = self.policy.output_nonlin(mean)
            else:
                mean = self.policy(obs)
            mean = mean.reshape(self.num_teachers, self.min_steps)

            # Iterate over all teachers
            loss = 0
            for t_idx, teacher in enumerate(self.teacher_policies):
                # Teacher actions
                if teacher.is_recurrent:
                    t_out, _ = teacher.rnn_layers(teacher_obs[t_idx])
                    t_out, lens = to.nn.utils.rnn.pad_packed_sequence(t_out)
                    t_out = to.cat([t_out[:l, i] for i, l in enumerate(lens)], 0)

                    t_out = teacher.output_layer(t_out)
                    if teacher.output_nonlin is not None:
                        t_out = teacher.output_nonlin(t_out)
                else:
                    t_out = teacher(obs)
                t_out = t_out.reshape(-1)

                # Get distributions
                s_dist = self.expl_strat.action_dist_at(mean[t_idx])
                s_act = s_dist.sample()
                t_dist = self.teacher_expl_strats[t_idx].action_dist_at(t_out)

                l = self.criterion(t_dist.log_prob(s_act), s_dist.log_prob(s_act))
                loss += l
            if epoch % 50 == 0:
                print(f"Epoch {epoch} Loss: {loss.item()}")
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
        state_dict = super(PDDR, self).__getstate__()

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
        # self.teacher_policies = []
        # self.teacher_expl_strats = []
        # for dir in state["teacher_ex_dirs"][:16]:
        #    dir = os.path.join(pyrado.TEMP_DIR, dir[dir.find("/data/temp/")+len("/data/temp/"):])

        #    _, teacher_policy, teacher_extra = load_experiment(dir)
        #    self.teacher_policies.append(teacher_policy)
        #    self.teacher_expl_strats.append(teacher_extra["expl_strat"])

        self.envs = Envs(
            state["num_cpu"],
            state["num_teachers"],
            state["env_real"],
            state["min_steps"],
            0.99,
            0.97,
            env_list=state["teacher_envs"],
        )

    def set_random_envs(self):
        """Creates random environments of the given type."""
        self.randomizer.randomize(num_samples=self.num_teachers)
        params = self.randomizer.get_params(fmt="dict", dtype="numpy")

        for e in range(self.num_teachers):
            self.teacher_envs.append(deepcopy(self.env_real))
            print({key: value[e] for key, value in params.items()})
            self.teacher_envs[e].domain_param = {key: value[e] for key, value in params.items()}

    def train_teachers(self, snapshot_mode: str = "latest", seed: int = None):
        """
        Trains all teachers.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new high-score)
        :param seed: seed value for the random number generators, pass `None` for no seeding
        """

        self.teacher_policies = []
        self.teacher_expl_strats = []
        self.teacher_critics = []

        for idx in range(self.num_teachers):
            algo = self.teacher_algo(
                save_dir=self.teacher_ex_dirs[idx],
                env=self.teacher_envs[idx],
                policy=deepcopy(self.teacher_policy),
                logger=None,
                **deepcopy(self.teacher_algo_hparam),
            )

            algo.train(snapshot_mode, seed)
            self.teacher_policies.append(deepcopy(algo.policy))
            self.teacher_expl_strats.append(deepcopy(algo.expl_strat))
            self.teacher_critics.append(deepcopy(algo.critic))

            del algo

    def load_teachers(self):
        """Recursively load all teachers that can be found in the current experiment's directory."""
        # Get the experiment's directory to load from
        ex_dir = ask_for_experiment(max_display=75, env_name=self.env_real.name, perma=False)
        self.load_teacher_experiment(ex_dir)
        if len(self.teacher_policies) < self.num_teachers:
            print(
                f"You have loaded {len(self.teacher_policies)} teachers - load at least {self.num_teachers - len(self.teacher_policies)} more!"
            )
            self.load_teachers()

    def load_teacher_experiment(self, exp: Experiment):
        """
        Load teachers from PDDRTeachers experiment.

        :param exp: the teacher's experiment object
        """
        _, _, extra = load_experiment(exp)

        self.unpack_teachers(extra)

    def unpack_teachers(self, extra: dict):
        """
        Unpack teachers from PDDRTeachers experiment.

        :param extra: dict with teacher data
        """
        num_teachers_to_load = self.num_teachers - len(self.teacher_policies)
        self.teacher_ex_dirs.extend(extra["teacher_ex_dirs"][:num_teachers_to_load])
        for dir in self.teacher_ex_dirs:
            dir = os.path.join(pyrado.TEMP_DIR, dir[dir.find("/data/temp/") + len("/data/temp/") :])
            teacher_env, teacher_policy, teacher_extra = load_experiment(dir)
            self.teacher_envs.append(teacher_env)
            self.teacher_policies.append(teacher_policy)
            self.teacher_expl_strats.append(teacher_extra["expl_strat"])

    def prune_teachers(self):
        """Prune teachers to only use the first num_teachers of them."""
        self.teacher_policies = self.teacher_policies[: self.num_teachers]
        self.teacher_envs = self.teacher_envs[: self.num_teachers]
        self.teacher_expl_strats = self.teacher_expl_strats[: self.num_teachers]
        self.teacher_critics = self.teacher_critics[: self.num_teachers]
        self.teacher_ex_dirs = self.teacher_ex_dirs[: self.num_teachers]
