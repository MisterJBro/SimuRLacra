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

        # Policy
        self.device = to.device(device)
        self.critic = critic
        self._expl_strat = NormalActNoiseExplStrat(self._policy, std_init=std_init)
        self.optimizer = to.optim.Adam(
            [
                {"params": self.policy.parameters()},
                {"params": self._expl_strat.noise.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=lr,
        )
        self.criterion = to.nn.SmoothL1Loss()

        print("Environment:        ", self.env.name)
        print("Observation shape:  ", self.obs_dim)
        print("Action number:      ", self.act_dim)
        print("Algorithm:          ", self.name)
        print("CPU count:          ", self.cpu_num)

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

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Sample batch
        rets, all_lengths = self.sample_batch()

        # Log current progress
        self.logger.add_value("max return", np.max(rets), 4)
        self.logger.add_value("median return", np.median(rets), 4)
        self.logger.add_value("avg return", np.mean(rets), 4)
        self.logger.add_value("min return", np.min(rets), 4)
        self.logger.add_value("std return", np.std(rets), 4)
        self.logger.add_value("std var", self.expl_strat.std, 4)
        self.logger.add_value("avg rollout len", np.mean(all_lengths), 4)
        self.logger.add_value("num total samples", np.sum(all_lengths))

        # Update policy and value function
        self.update()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, np.mean(rets), meta_info)

    def sample_batch(self) -> np.ndarray:
        """ Sample batch of trajectories for training. """
        obss = self.envs.reset()
        scale = self.policy.net.output_scale

        for _ in range(self.traj_len):
            obss = to.as_tensor(obss).to(self.device)   
            with to.no_grad():
                acts = self.expl_strat(obss).cpu().numpy()#.clip(-scale, scale)
                vals = self.critic(obss).reshape(-1).cpu().numpy()
            obss = self.envs.step(acts, vals)

        rets = self.envs.ret_and_adv()
        return rets

    def update(self):
        """ Update the policy using PPO. """
        obs, act, rew, ret, adv = self.envs.get_data(self.device)

        with to.no_grad():
            mean = self.policy(obs)
            old_logp = self.expl_strat.action_dist_at(mean).log_prob(act).sum(-1)

        for i in range(self.epoch_num):
            self.optimizer.zero_grad()

            mean = self.policy(obs)
            dist = self.expl_strat.action_dist_at(mean)
            val = self.critic(obs).reshape(-1)

            logp = dist.log_prob(act).sum(-1)
            loss_policy, kl = self.loss_fcn(logp, old_logp, adv)
            loss_policy += 0.01*dist.entropy().mean()

            # Early stopping if kl divergence too high
            if kl > self.max_kl:
                return
            loss_value = self.criterion(val, ret)

            loss = loss_policy + loss_value
            loss.backward()

            self.optimizer.step()
        
        self.expl_strat.std -= 0.000#7

    def train(self, snapshot_mode: str = "latest", seed: int = None, meta_info: dict = None):
        super().train(snapshot_mode, seed, meta_info)
        
        # Close environments
        self.envs.close()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        pyrado.save(self._expl_strat.policy, "policy", "pt", self.save_dir, meta_info)
        pyrado.save(self._expl_strat, "expl_strat", "pt", self.save_dir, meta_info)
        pyrado.save(self.critic, "vfcn", "pt", self.save_dir, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self.env, "env", "pkl", self.save_dir, meta_info)

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