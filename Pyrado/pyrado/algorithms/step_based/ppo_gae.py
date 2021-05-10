from typing import Sequence
import time
import os

import numpy as np
import pyrado
import torch as to
from copy import deepcopy

from pyrado.algorithms.base import Algorithm
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.envs import Envs
from torch.utils.tensorboard import SummaryWriter
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat


class PPOGAE(Algorithm):
    """
    Implementation of Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
    that differs from the pyrado PPO implementation.
    .. seealso::
        [1] J. Schulmann,  F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms",
        arXiv, 2017
    """

    name: str = "ppo_gae"

    def __init__(
        self,
        save_dir: str,
        env: Env,
        policy: Policy,
        critic: Policy,
        max_iter: int,
        tb_name: str = "ppo",
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
        early_stopping: bool = True,
        std_loss: float = 0.1,
    ):
        """
        Constructor
        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param critic: advantage estimation function $A(s,a) = Q(s,a) - V(s)$
        :param max_iter: number of iterations (policy updates)
        :param tb_name: name for tensorboard
        :param traj_len: trajectorie length for one batch
        :param gamma: discount factor
        :param lam: lambda factor for GAE
        :param env_num: number of environments for parallel sampling
        :param cpu_num: number of cpu cores to use
        :param epoch_num: number of epochs (how often we iterate over the same batch)
        :param device: device to use for updating the policy (cpu or gpu)
        :param max_kl: Maximum KL divergence between two updates
        :param std_init: initial standard deviation on the actions for the exploration noise
        :param clip_ratio: max/min probability ratio, see [1]
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        assert isinstance(policy, Policy)

        # Call Algorithm's constructor.
        super().__init__(save_dir, max_iter, policy, logger)

        # Environment
        self.env = env
        self.env_num = env_num
        self.envs = Envs(cpu_num, env_num, env, traj_len, gamma, lam)
        self.obs_dim = self.env.obs_space.flat_dim
        self.act_dim = self.env.act_space.flat_dim

        # Other
        self.gamma = gamma
        self.lam = lam
        self.traj_len = traj_len
        self.cpu_num = cpu_num
        self.epoch_num = epoch_num
        self.max_kl = max_kl
        self.clip_ratio = clip_ratio
        self.end = False
        self.early_stopping = early_stopping

        # Policy
        self.device = to.device(device)
        self.critic = critic
        self.critic = self.critic.to(self.device)
        self.std_loss = std_loss
        self._expl_strat = NormalActNoiseExplStrat(self._policy, std_init=std_init, std_min=0.1)
        self._policy = self._policy.to(self.device)
        self._expl_strat = self._expl_strat.to(self.device)
        self.optimizer = to.optim.Adam(
            [
                {"params": self.policy.parameters()},
                {"params": self._expl_strat.noise.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=lr,
        )
        self.criterion = to.nn.SmoothL1Loss()
        self.reset_states()

        print("Environment:        ", self.env.name)
        print("Observation shape:  ", self.obs_dim)
        print("Action number:      ", self.act_dim)
        print("Algorithm:          ", self.name)
        print("CPU count:          ", self.cpu_num)

    def reset_states(self, env_indices = []):
        """
        Resets the hidden states.
        :param env_indices: indices of the environment hidden states to reset. If empty, reset all. 
        """
        num_env_ind = len(env_indices)
        if num_env_ind == 0:
            if self.policy.is_recurrent:
                self.hidden_policy = to.zeros(self.env_num, self.policy.hidden_size,
                                device=self.device).contiguous()
            if self.critic.is_recurrent:
                self.hidden_critic = to.zeros(self.env_num, self.critic.hidden_size,
                                device=self.device).contiguous()
        else:
            if self.policy.is_recurrent:
                self.hidden_policy[env_indices] = to.zeros(num_env_ind, self.policy.hidden_size,
                                device=self.device).contiguous()
            if self.critic.is_recurrent:
                self.hidden_critic[env_indices] = to.zeros(num_env_ind, self.critic.hidden_size,
                                device=self.device).contiguous()

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self._expl_strat

    def loss_fcn(self, log_probs: to.Tensor, log_probs_old: to.Tensor, adv: to.Tensor) -> [to.Tensor, to.Tensor]:
        """
        PPO loss function.
        :param log_probs: logarithm of the probabilities of the taken actions using the updated policy
        :param log_probs_old: logarithm of the probabilities of the taken actions using the old policy
        :param adv: advantage values
        :return: loss value, kl_approximation
        """

        ratio = to.exp(log_probs - log_probs_old)
        clipped = to.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss = -(to.min(ratio * adv, clipped)).mean()
        kl_approx = (log_probs_old - log_probs).mean().item()
        return loss, kl_approx

    def stopping_criterion_met(self) -> bool:
        return self.end

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Sample batch
        rets, all_lengths = self.sample_batch()

        # Log current progress
        self.logger.add_value("max return", np.max(rets), 4)
        self.logger.add_value("median return", np.median(rets), 4)
        self.logger.add_value("avg return", np.mean(rets), 4)
        self.logger.add_value("min return", np.min(rets), 4)
        self.logger.add_value("std return", np.std(rets), 4)
        self.logger.add_value("std var", self.expl_strat.std.item(), 4)
        self.logger.add_value("avg rollout len", np.mean(all_lengths), 4)
        self.logger.add_value("num total samples", np.sum(all_lengths))

        # Early stoping
        if self.early_stopping and self._curr_iter > 50 and np.mean(rets) > 0.8 * self.traj_len and self.expl_strat.std.item() < 0.3:
            print('Reached optimal policy! Early stop!' )
            self.end = True
            return

        # Update policy and value function
        self.update()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, np.mean(rets), meta_info)

    def sample_batch(self) -> np.ndarray:
        """ Sample batch of trajectories for training. """
        obss = self.envs.reset()
        self.reset_states()

        for _ in range(self.traj_len):
            obss = to.as_tensor(obss).to(self.device)
            with to.no_grad():
                if self.expl_strat.is_recurrent:
                    acts, self.hidden_policy = self.expl_strat(obss, self.hidden_policy.contiguous())
                else:
                    acts = self.expl_strat(obss)
                acts = acts.cpu().numpy()
                if self.critic.is_recurrent:
                    vals, self.hidden_critic = self.critic(obss, self.hidden_critic.contiguous())
                else:
                    vals = self.critic(obss)
                vals = vals.reshape(-1).cpu().numpy()
            obss, done_ind = self.envs.step(acts, vals)
            if len(done_ind) != 0:
                self.reset_states(done_ind)

        rets = self.envs.ret_and_adv()
        return rets

    def update(self):
        """ Update the policy using PPO. """
        obs, act, rew, ret, adv, dones = self.envs.get_data(self.device)
        
        # For recurrent pack observations
        if self.policy.is_recurrent:
            obs = obs.reshape(-1, self.traj_len, obs.shape[-1])
            obs_list = []
            lengths = []
            for idx, section in enumerate(dones):
                start = 0
                for end in section[1:]:
                    obs_list.append(obs[idx, start:end])
                    lengths.append(end-start)
                    start = end
                if start != self.traj_len:
                    obs_list.append(obs[idx, start:])
                    lengths.append(self.traj_len-start)
            obs = to.nn.utils.rnn.pad_sequence(obs_list)
            obs = to.nn.utils.rnn.pack_padded_sequence(obs, lengths=lengths, enforce_sorted=False)

        with to.no_grad():
            if self.policy.is_recurrent:
                mean, _ = self.policy.rnn_layers(obs)   
                mean, lens = to.nn.utils.rnn.pad_packed_sequence(mean)
                mean = to.cat([mean[:l, i] for i, l in enumerate(lens)], 0)

                mean = self.policy.output_layer(mean)
                if self.policy.output_nonlin is not None:
                    mean = self.policy.output_nonlin(mean)
            else:
                mean = self.policy(obs)
            old_logp = self.expl_strat.action_dist_at(mean).log_prob(act).sum(-1)

        for i in range(self.epoch_num):
            self.optimizer.zero_grad()

            # Policy
            if self.policy.is_recurrent:
                mean, _ = self.policy.rnn_layers(obs)   
                mean, lens = to.nn.utils.rnn.pad_packed_sequence(mean)
                mean = to.cat([mean[:l, i] for i, l in enumerate(lens)], 0)

                mean = self.policy.output_layer(mean)
                if self.policy.output_nonlin is not None:
                    mean = self.policy.output_nonlin(mean)
            else:
                mean = self.policy(obs)
            dist = self.expl_strat.action_dist_at(mean)

            # Critic
            if self.critic.is_recurrent:
                val, _ = self.critic.rnn_layers(obs)   
                val, lens = to.nn.utils.rnn.pad_packed_sequence(val)
                val = to.cat([val[:l, i] for i, l in enumerate(lens)], 0)

                val = self.critic.output_layer(val)
                if self.critic.output_nonlin is not None:
                    val = self.critic.output_nonlin(val)
            else:
                val = self.critic(obs)
            val = val.reshape(-1)
            
            logp = dist.log_prob(act).sum(-1)
            loss_policy, kl = self.loss_fcn(logp, old_logp, adv)
            loss_policy += self.expl_strat.std.mean() * self.std_loss

            # Early stopping if kl divergence too high
            if kl > self.max_kl:
                return
            loss_value = self.criterion(val, ret)

            loss = loss_policy + loss_value
            loss.backward()

            self.optimizer.step()

    def train(self, snapshot_mode: str = "latest", seed: int = None, meta_info: dict = None):
        super().train(snapshot_mode, seed, meta_info)
        
        # Close environments
        self.envs.close()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        pyrado.save(self._expl_strat.policy, "policy.pt", self.save_dir, use_state_dict=True)
        pyrado.save(self._expl_strat, "expl_strat.pt", self.save_dir, use_state_dict=True)
        pyrado.save(self.critic, "vfcn.pt", self.save_dir, use_state_dict=True)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self.env, "env.pkl", self.save_dir)

    def __getstate__(self):
        self.envs
        # Remove the unpickleable elements from this algorithm instance
        tmp_envs = self.__dict__.pop("envs")

        # Call Algorithm's __getstate__() without the unpickleable elements
        state_dict = super(PPOGAE, self).__getstate__()

        # Make a deep copy of the state dict such that we can return the pickleable version
        state_dict_copy = deepcopy(state_dict)

        # Insert them back
        self.__dict__["envs"] = tmp_envs

        return state_dict_copy

    def __setstate__(self, state):
        # Call Algorithm's __setstate__()
        super().__setstate__(state)

        # Recover settings of environment
        self.envs = Envs(state["cpu_num"], state["env_num"], state["env"], state["traj_len"], state["gamma"], state["lam"])