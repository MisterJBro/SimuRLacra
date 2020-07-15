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

import joblib
import numpy as np
import torch as to
from typing import Sequence
from init_args_serializer import Serializable

import pyrado
from pyrado.algorithms.adr_discriminator import RewardGenerator
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.svpg import SVPG, SVPGParticle
from pyrado.domain_randomization.domain_parameter import DomainParam
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_evaluation import eval_domain_params
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces.box import BoxSpace
from os import path as osp


class ADR(Algorithm):
    """
    Active Domain Randomization (ADR)

    .. seealso::
        [1] B. Mehta, M. Diaz, F. Golemo, C.J. Pal, L. Paull, "Active Domain Randomization", arXiv, 2019
    """

    name: str = 'adr'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 subroutine: Algorithm,
                 max_iter: int,
                 svpg_particle_hparam: dict,
                 num_svpg_particles: int,
                 num_discriminator_epoch: int,
                 batch_size: int,
                 svpg_learning_rate: float = 3e-4,
                 svpg_temperature: float = 10,
                 svpg_evaluation_steps: int = 10,
                 svpg_horizon: int = 50,
                 svpg_kl_factor: float = 0.03,
                 svpg_warmup: int = 0,
                 svpg_serial: bool = False,
                 num_sampler_envs: int = 4,
                 num_trajs_per_config: int = 8,
                 max_step_length: float = 0.05,
                 randomized_params: Sequence[str] = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment to train in
        :param subroutine: algorithm which performs the policy / value-function optimization
        :param max_iter: maximum number of iterations
        :param svpg_particle_hparam: SVPG particle hyperparameters
        :param num_svpg_particles: number of SVPG particles
        :param num_discriminator_epoch: epochs in discriminator training
        :param batch_size: batch size for training
        :param svpg_learning_rate: SVPG particle optimizers' learning rate
        :param svpg_temperature: SVPG temperature coefficient (how strong is the influence of the particles on each other)
        :param svpg_evaluation_steps: how many configurations to sample between training
        :param svpg_horizon: how many steps until the particles are reset
        :param svpg_kl_factor: kl reward coefficient
        :param svpg_warmup: number of iterations without SVPG training in the beginning
        :param svpg_serial: serial mode (see SVPG)
        :param num_sampler_envs: number of parallel sampling environments for the physics configs
        :param num_trajs_per_config: number of trajectories to sample from each config
        :param max_step_length: maximum change of physics parameters per step
        :param randomized_params: which parameters to randomize
        :param logger: see Logger
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(subroutine, Algorithm):
            raise pyrado.TypeErr(given=subroutine, expected_type=Algorithm)
        if not isinstance(subroutine.policy, Policy):
            raise pyrado.TypeErr(given=subroutine.policy, expected_type=Policy)

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, subroutine.policy, logger)
        self.log_loss = True

        # Store the inputs
        self.env = env
        self.subroutine = subroutine
        self.num_particles = num_svpg_particles
        self.num_discriminator_epoch = num_discriminator_epoch
        self.batch_size = batch_size
        self.num_trajs_per_config = num_trajs_per_config
        self.warm_up_time = svpg_warmup
        self.svpg_evaluation_steps = svpg_evaluation_steps
        self.svpg_temperature = svpg_temperature
        self.svpg_lr = svpg_learning_rate
        self.svpg_max_step_length = max_step_length
        self.svpg_horizon = svpg_horizon
        self.svpg_kl_factor = svpg_kl_factor

        # Get the number of params
        if(isinstance(randomized_params, list) and len(randomized_params) == 0):
            randomized_params = inner_env(self.env).get_nominal_domain_param().keys()
        self.params = [DomainParam(param, 1) for param in randomized_params]
        self.num_params = len(self.params)

        # Initialize the sampler
        self.pool = SamplerPool(num_sampler_envs)

        # Initialize reward generator
        self.reward_generator = RewardGenerator(
            env.spec,
            self.batch_size,
            reward_multiplier=1,
            lr=1e-3,
            logger=self.logger
        )

        # Initialize step counter
        self.curr_time_step = 0

        # Initialize logbook
        self.sim_instances_full_horizon = np.random.random_sample(
            (self.num_particles, self.svpg_horizon, self.svpg_evaluation_steps, self.num_params)
        )

        self.svpg_wrapper = SVPGAdapter(
            env,
            self.params,
            subroutine.expl_strat,
            self.reward_generator,
            horizon=self.svpg_horizon,
            num_trajs_per_config=self.num_trajs_per_config,
            num_sampler_envs=num_sampler_envs,
        )

        # Initialize SVPG
        self.svpg = SVPG(
            save_dir,
            self.svpg_wrapper,
            svpg_particle_hparam,
            max_iter,
            self.num_particles,
            self.svpg_temperature,
            self.svpg_lr,
            self.svpg_horizon,
            serial=svpg_serial,
            num_sampler_envs=num_sampler_envs,
            logger=logger
        )

    def compute_params(self, sim_instances: to.Tensor, t: int):
        """
        Computes the parameters

        :param sim_instances: Physics configurations trajectory
        :param t: time step to chose
        :return: parameters at the time
        """
        nominal = self.svpg_wrapper.nominal_dict()
        keys = nominal.keys()
        assert (len(keys) == sim_instances[t][0].shape[0])

        params = []
        for sim_instance in sim_instances[t]:
            d = dict()
            for i, k in enumerate(keys):
                d[k] = (sim_instance[i] + 0.5)*(nominal[k])
            params.append(d)

        return params

    def step(self, snapshot_mode: str, meta_info: dict = None, parallel: bool = True):
        rand_trajs = []
        ref_trajs = []
        ros = []
        visited = []
        for i in range(self.svpg.num_particles):
            done = False
            svpg_env = self.svpg_wrapper
            state = svpg_env.reset()
            states = []
            actions = []
            rewards = []
            infos = []
            rand_trajs_now = []
            if parallel:
                with to.no_grad():
                    for t in range(10):
                        action = self.svpg.expl_strats[i](
                            to.as_tensor(state, dtype=to.get_default_dtype())).detach().numpy()
                        state = svpg_env.lite_step(action)
                        states.append(state)
                        actions.append(action)
                    visited.append(states)
                    rewards, rand_trajs_now, ref_trajs_now = svpg_env.eval_states(states)
                    rand_trajs += rand_trajs_now
                    ref_trajs += ref_trajs_now
                    ros.append(StepSequence(observations=states, actions=actions, rewards=rewards))
            else:
                with to.no_grad():
                    while not done:
                        action = self.svpg.expl_strats[i](
                            to.as_tensor(state, dtype=to.get_default_dtype())).detach().numpy()
                        state, reward, done, info = svpg_env.step(action)
                        print(self.params.array_to_dict(state), ' => ', reward)
                        states.append(state)
                        rewards.append(reward)
                        actions.append(action)
                        infos.append(info)
                        rand_trajs += info['rand']
                        ref_trajs += info['ref']
                    ros.append(StepSequence(observations=states, actions=actions, rewards=rewards))
            self.logger.add_value(f'SVPG_agent_{i}_mean_reward', np.mean(rewards))
            ros[i].torch(data_type=to.DoubleTensor)
            for rt in rand_trajs_now:
                rt.torch(data_type=to.double)
                rt.observations = rt.observations.double().detach()
                rt.actions = rt.actions.double().detach()
            self.subroutine.update(rand_trajs_now)

        # Logging
        rets = [ro.undiscounted_return() for ro in rand_trajs]
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_std = np.std(rets)
        self.logger.add_value('num rollouts', len(rand_trajs))
        self.logger.add_value('avg rollout len', np.mean([ro.length for ro in rand_trajs]))
        self.logger.add_value('avg return', ret_avg)
        self.logger.add_value('median return', ret_med)
        self.logger.add_value('std return', ret_std)

        # Flatten and combine all randomized and reference trajectories for discriminator
        flattened_randomized = StepSequence.concat(rand_trajs)
        flattened_randomized.torch(data_type=to.double)
        flattened_reference = StepSequence.concat(ref_trajs)
        flattened_reference.torch(data_type=to.double)
        self.reward_generator.train(flattened_reference, flattened_randomized, self.num_discriminator_epoch)

        if self.curr_time_step > self.warm_up_time:
            # Update the particles
            # List of lists to comply with interface
            self.svpg.update(list(map(lambda x: [x], ros)))
        flattened_randomized.torch(data_type=to.double)
        flattened_randomized.observations = flattened_randomized.observations.double().detach()
        flattened_randomized.actions = flattened_randomized.actions.double().detach()

        # np.save(f'{self.save_dir}actions{self.curr_iter}', flattened_randomized.actions)
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)
        self.subroutine.make_snapshot(snapshot_mode='best', curr_avg_ret=float(ret_avg))
        self.curr_time_step += 1

    def save_snapshot(self, meta_info: dict = None):
        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            joblib.dump(self.env, osp.join(self._save_dir, 'env.pkl'))
            to.save(self.reward_generator.discriminator, osp.join(self._save_dir, 'discriminator.pt'))
            self.svpg.save_snapshot(meta_info=[])
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            raise NotImplementedError

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self.reward_generator.discriminator.load_state_dict(
                to.load(osp.join(ld, 'discriminator.pt')).state_dict()
            )
            self.svpg.load_snapshot(ld)
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            raise NotImplementedError


class SVPGAdapter(EnvWrapper, Serializable):
    """ Wrapper to encapsulate the domain parameter search as a reinforcement learning problem """

    def __init__(self,
                 wrapped_env: Env,
                 parameters: Sequence[DomainParam],
                 inner_policy: Policy,
                 discriminator: RewardGenerator,
                 step_length: float = 0.01,
                 horizon: int = 50,
                 num_trajs_per_config: int = 8,
                 num_sampler_envs: int = 4):
        """
        Constructor

        :param wrapped_env: the environment to wrap
        :param parameters: which physics parameters should be randomized
        :param inner_policy: the policy to train the subroutine on
        :param discriminator: the discriminator to distinguish reference envs from randomized ones
        :param step_length: the step size
        :param horizon: an svpg horizon
        :param num_trajs_per_config: number of trajectories to sample per physics configuration
        :param num_sampler_envs: the number of samplers operating in parallel
        """
        Serializable._init(self, locals())

        EnvWrapper.__init__(self, wrapped_env)

        self.parameters: Sequence[DomainParam] = parameters
        self.pool = SamplerPool(num_sampler_envs)
        self.inner_policy = inner_policy
        self.state = None
        self.count = 0
        self.num_trajs = num_trajs_per_config
        self.svpg_max_step_length = step_length
        self.discriminator = discriminator
        self.max_steps = 8
        self._adapter_obs_space = BoxSpace(-np.ones(len(parameters)), np.ones(len(parameters)))
        self._adapter_act_space = BoxSpace(-np.ones(len(parameters)), np.ones(len(parameters)))
        self.horizon = horizon
        self.horizon_count = 0

    @property
    def obs_space(self):
        return self._adapter_obs_space

    @property
    def act_space(self):
        return self._adapter_act_space

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None):
        assert domain_param is None
        self.count = 0
        if init_state is None:
            self.state = np.random.random_sample(len(self.parameters))
        return self.state

    def step(self, act: np.ndarray):
        # Clip the action according to the maximum step length
        action = np.clip(act, -1, 1)*self.svpg_max_step_length

        # Perform step by moving into direction of action
        self.state = np.clip(self.state + action, 0, 1)
        param_norm = self.state + 0.5
        rand_eval_params = [self.array_to_dict(param_norm*self.nominal())]*self.num_trajs
        norm_eval_params = [self.nominal_dict()]*self.num_trajs
        rand = eval_domain_params(self.pool, self.wrapped_env, self.inner_policy, rand_eval_params)
        ref = eval_domain_params(self.pool, self.wrapped_env, self.inner_policy, norm_eval_params)
        rewards = [self.discriminator.get_reward(traj) for traj in rand]
        reward = np.mean(rewards)
        info = dict(rand=rand, ref=ref)
        if self.count >= self.max_steps - 1:
            done = True
        else:
            done = False
        self.count += 1
        self.horizon_count += 1
        if self.horizon_count >= self.horizon:
            self.horizon_count = 0
            self.state = np.random.random_sample(self.parameters.length)

        return self.state, reward, done, info

    def lite_step(self, act: np.ndarray):
        """
        Performs a step without the step interface.
        This allows for parallel computation of prior steps.

        :param act: the action to perform
        :return: the observation after the step
        """
        action = np.clip(act, -1, 1)*self.svpg_max_step_length
        self.state = np.clip(self.state + action, 0, 1)
        return self.state

    def eval_states(self, states: Sequence[np.ndarray]):
        """
        Evaluate the states.

        :param states: the states to evaluate
        :return: respective rewards and according trajectories
        """
        flatten = lambda l: [item for sublist in l for item in sublist]
        sstates = flatten([
            [self.array_to_dict((state + 0.5)*self.nominal())]*self.num_trajs
            for state in states]
        )
        rand = eval_domain_params(self.pool, self.wrapped_env, self.inner_policy, sstates)
        ref = eval_domain_params(self.pool, self.wrapped_env, self.inner_policy,
                                 [self.nominal_dict()]*(self.num_trajs*len(states)))
        rewards = [self.discriminator.get_reward(traj) for traj in rand]
        rewards = [np.mean(rewards[i*self.num_trajs:(i + 1)*self.num_trajs]) for i in range(len(states))]
        return rewards, rand, ref

    def params(self):
        return [param.name for param in self.parameters]

    def nominal(self):
        return [inner_env(self.wrapped_env).get_nominal_domain_param()[k] for k in self.params()]

    def nominal_dict(self):
        return {k: inner_env(self.wrapped_env).get_nominal_domain_param()[k] for k in self.params()}

    def array_to_dict(self, arr):
        return {k: a for k, a in zip(self.params(), arr)}
