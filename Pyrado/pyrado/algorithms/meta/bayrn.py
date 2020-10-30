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
import os
import os.path as osp
import torch as to
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement, PosteriorMean
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from tabulate import tabulate
from typing import Optional

import pyrado
from pyrado.algorithms.base import Algorithm, InterruptableAlgorithm
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.logger.step import StepLogger
from pyrado.utils.saving_loading import save_prefix_suffix, load_prefix_suffix
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.environment_wrappers.utils import inner_env, typed_env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.rollout import rollout
from pyrado.utils.order import natural_sort
from pyrado.utils.input_output import print_cbt
from pyrado.utils.math import UnitCubeProjector
from pyrado.utils.data_processing import standardize


class BayRn(InterruptableAlgorithm):
    """
    Bayesian Domain Randomization (BayRn)

    .. note::
        A candidate is a set of parameter values for the domain parameter distribution and its value is the
        (estimated) real-world return.

    .. seealso::
        F. Muratore, C. Eilers, M. Gienger, J. Peters, "Bayesian Domain Randomization for Sim-to-Real Transfer",
        arXiv, 2020
    """

    name: str = 'bayrn'
    iteration_key: str = 'bayrn_iteration'  # logger's iteration key

    def __init__(self,
                 save_dir: str,
                 env_sim: MetaDomainRandWrapper,
                 env_real: [RealEnv, EnvWrapper],
                 subrtn: Algorithm,
                 bounds: to.Tensor,
                 max_iter: int,
                 acq_fc: str,
                 acq_restarts: int,
                 acq_samples: int,
                 acq_param: dict = None,
                 num_init_cand: int = 5,
                 mc_estimator: bool = True,
                 num_eval_rollouts_real: int = 5,
                 num_eval_rollouts_sim: int = 50,
                 thold_succ: float = pyrado.inf,
                 thold_succ_subrtn: float = -pyrado.inf,
                 warmstart: bool = True,
                 policy_param_init: Optional[to.Tensor] = None,
                 valuefcn_param_init: Optional[to.Tensor] = None,
                 subrtn_snapshot_mode: str = 'best',
                 logger: Optional[StepLogger] = None):
        """
        Constructor

        .. note::
            If you want to continue an experiment, use the `load_dir` argument for the `train` call. If you want to
            initialize every of the policies with a pre-trained policy parameters use `policy_param_init`.

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain
        :param subrtn: algorithm which performs the policy / value-function optimization
        :param bounds: boundaries for inputs of randomization function, format: [lower, upper]
        :param max_iter: maximum number of iterations
        :param acq_fc: Acquisition Function
                       'UCB': Upper Confidence Bound (default $\beta = 0.1$)
                       'EI': Expected Improvement
                       'PI': Probability of Improvement
        :param acq_restarts: number of restarts for optimizing the acquisition function
        :param acq_samples: number of initial samples for optimizing the acquisition function
        :param acq_param: hyper-parameter for the acquisition function, e.g. $\beta$ for UCB
        :param num_init_cand: number of initial policies to train, ignored if `init_dir` is provided
        :param mc_estimator: estimate the return with a sample average (`True`) or a lower confidence
                                     bound (`False`) obtained from bootstrapping
        :param num_eval_rollouts_real: number of rollouts in the target domain to estimate the return
        :param num_eval_rollouts_sim: number of rollouts in simulation to estimate the return after training
        :param thold_succ: success threshold on the real system's return for BayRn, stop the algorithm if exceeded
        :param thold_succ_subrtn: success threshold on the simulated system's return for the subroutine, repeat the
                                      subroutine until the threshold is exceeded or the for a given number of iterations
        :param warmstart: initialize the policy parameters with the one of the previous iteration. This option has no
                          effect for initial policies and can be overruled by passing init policy params explicitly.
        :param policy_param_init: initial policy parameter values for the subroutine, set `None` to be random
        :param valuefcn_param_init: initial value function parameter values for the subroutine, set `None` to be random
        :param subrtn_snapshot_mode: snapshot mode for saving during training of the subroutine
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if typed_env(env_sim, MetaDomainRandWrapper) is None:
            raise pyrado.TypeErr(given=env_sim, expected_type=MetaDomainRandWrapper)
        if not isinstance(subrtn, Algorithm):
            raise pyrado.TypeErr(given=subrtn, expected_type=Algorithm)
        assert bounds.shape[0] == 2
        assert all(bounds[1] > bounds[0])
        if num_init_cand < 1:
            raise pyrado.ValueErr(given=num_init_cand, ge_constraint='1')

        # Call InterruptableAlgorithm's constructor without specifying the policy
        super().__init__(num_checkpoints=2, init_checkpoint=-2, save_dir=save_dir, max_iter=max_iter,
                         policy=subrtn.policy, logger=logger)

        self._env_sim = env_sim
        self._env_real = env_real
        self._subrtn = subrtn
        self._subrtn.save_name = 'subrtn'
        self.bounds = bounds
        self.cand_dim = bounds.shape[1]
        self.cands = None  # called x in the context of GPs
        self.cands_values = None  # called y in the context of GPs
        self.argmax_cand = to.Tensor()
        self.acq_fcn_type = acq_fc.upper()
        self.acq_restarts = acq_restarts
        self.acq_samples = acq_samples
        self.acq_param = acq_param
        self.num_init_cand = num_init_cand
        self.mc_estimator = mc_estimator
        self.policy_param_init = policy_param_init
        self.valuefcn_param_init = valuefcn_param_init.detach() if valuefcn_param_init is not None else None
        self.warmstart = warmstart
        self.num_eval_rollouts_real = num_eval_rollouts_real
        self.num_eval_rollouts_sim = num_eval_rollouts_sim
        self.subrtn_snapshot_mode = subrtn_snapshot_mode
        self.thold_succ = to.tensor([thold_succ])
        self.thold_succ_subrtn = to.tensor([thold_succ_subrtn])
        self.max_subrtn_rep = 3  # number of tries to exceed thold_succ_subrtn during training in simulation
        self.curr_cand_value = -pyrado.inf  # for the stopping criterion
        self.uc_normalizer = UnitCubeProjector(bounds[0, :], bounds[1, :])

        if self.policy_param_init is not None:
            if to.is_tensor(self.policy_param_init):
                self.policy_param_init.detach()
            else:
                self.policy_param_init = to.tensor(self.policy_param_init)

        # Save initial environments and bounds
        self.save_snapshot(meta_info=None)
        to.save(self.bounds, osp.join(self.save_dir, 'bounds.pt'))

    @property
    def subroutine(self) -> Algorithm:
        """ Get the policy optimization subroutine. """
        return self._subrtn

    def stopping_criterion_met(self) -> bool:
        return self.curr_cand_value > self.thold_succ

    def train_policy_sim(self, cand: to.Tensor, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param cand: hyper-parameters for the domain parameter distribution (need be compatible with the randomizer)
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: estimated return of the trained policy in the target domain
        """
        # Save the current candidate
        to.save(cand.view(-1), osp.join(self.save_dir, f'{prefix}_candidate.pt'))

        # Set the domain randomizer
        self._env_sim.adapt_randomizer(cand.detach().cpu().numpy())

        # Reset the subroutine's algorithm which includes resetting the exploration
        self._subrtn.reset()

        # Do a warm start if desired
        self._subrtn.init_modules(
            self.warmstart, policy_param_init=self.policy_param_init, valuefcn_param_init=self.valuefcn_param_init
        )

        # Train a policy in simulation using the subroutine
        self._subrtn.train(snapshot_mode=self.subrtn_snapshot_mode, meta_info=dict(prefix=prefix))

        # Return the estimated return of the trained policy in simulation
        avg_ret_sim = self.eval_policy(
            None, self._env_sim, self._subrtn.policy, self.mc_estimator, prefix, self.num_eval_rollouts_sim
        )
        return float(avg_ret_sim)

    def train_init_policies(self):
        """
        Initialize the algorithm with a number of random distribution parameter sets a.k.a. candidates specified by
        the user. Train a policy for every candidate. Finally, store the policies and candidates.
        """
        cands = to.empty(self.num_init_cand, self.cand_dim)
        for i in range(self.num_init_cand):
            print_cbt(f'Generating initial domain instance and policy {i + 1} of {self.num_init_cand} ...',
                      'g', bright=True)
            # Generate random samples within bounds
            cands[i, :] = (self.bounds[1, :] - self.bounds[0, :])*to.rand(self.bounds.shape[1]) + self.bounds[0, :]

            # Train a policy for each candidate, repeat if the resulting policy did not exceed the success threshold
            print_cbt(f'Randomly sampled the next candidate: {cands[i].numpy()}', 'g')
            wrapped_trn_fcn = until_thold_exceeded(
                self.thold_succ_subrtn.item(), self.max_subrtn_rep
            )(self.train_policy_sim)
            wrapped_trn_fcn(cands[i], prefix=f'init_{i}')

        # Save candidates into a single tensor (policy is saved during training or exists already)
        save_prefix_suffix(cands, 'candidates', 'pt', self.save_dir, meta_info=None)
        self.cands = cands

    def eval_init_policies(self):
        """
        Execute the trained initial policies on the target device and store the estimated return per candidate.
        The number of initial policies to evaluate is the number of found policies.
        """
        # Crawl through the experiment's directory
        for root, dirs, files in os.walk(self.save_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            found_policies = [p for p in files if p.startswith('init_') and p.endswith('_policy.pt')]
            found_cands = [c for c in files if c.startswith('init_') and c.endswith('_candidate.pt')]
        if not len(found_policies) == len(found_cands):
            raise pyrado.ValueErr(msg='Found a different number of initial policies than candidates!')
        elif len(found_policies) == 0:
            raise pyrado.ValueErr(msg='No policies or candidates found!')

        num_init_cand = len(found_cands)
        cands_values = to.empty(num_init_cand)

        # Load all found candidates to save them into a single tensor
        found_cands = natural_sort(found_cands)  # the order is important since it determines the rows of the tensor
        cands = to.stack([to.load(osp.join(self.save_dir, c)) for c in found_cands])

        # Evaluate learned policies from random candidates on the target environment (real-world) system
        for i in range(num_init_cand):
            policy = load_prefix_suffix(self.policy, 'policy', 'pt', self.save_dir, meta_info=dict(prefix=f'init_{i}'))
            cands_values[i] = self.eval_policy(self.save_dir, self._env_real, policy, self.mc_estimator,
                                               prefix=f'init_{i}', num_rollouts=self.num_eval_rollouts_real)

        # Save candidates's and their returns into tensors (policy is saved during training or exists already)
        # save_prefix_suffix(cands, 'candidates', 'pt', self._save_dir, meta_info)
        save_prefix_suffix(cands_values, 'candidates_values', 'pt', self.save_dir, meta_info=None)
        self.cands, self.cands_values = cands, cands_values

    @staticmethod
    def eval_policy(save_dir: [str, None],
                    env: [RealEnv, SimEnv, MetaDomainRandWrapper],
                    policy: Policy,
                    mc_estimator: bool,
                    prefix: str,
                    num_rollouts: int,
                    num_parallel_envs: int = 1) -> to.Tensor:
        """
        Evaluate a policy on the target system (real-world platform).
        This method is static to facilitate evaluation of specific policies in hindsight.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: target environment for evaluation, in the sim-2-sim case this is another simulation instance
        :param policy: policy to evaluate
        :param mc_estimator: estimate the return with a sample average (`True`) or a lower confidence
                                     bound (`False`) obtained from bootrapping
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_parallel_envs: number of environments for the parallel sampler (only used for SimEnv)
        :return: estimated return in the target domain
        """
        if save_dir is not None:
            print_cbt(f'Executing {prefix}_policy ...', 'c', bright=True)

        rets_real = to.zeros(num_rollouts)
        if isinstance(inner_env(env), RealEnv):
            # Evaluate sequentially when conducting a sim-to-real experiment
            for i in range(num_rollouts):
                rets_real[i] = rollout(env, policy, eval=True).undiscounted_return()
                # If a reward of -1 is given, skip evaluation ahead and set all returns to zero
                if rets_real[i] == -1:
                    print_cbt('Set all returns for this policy to zero.', color='c')
                    rets_real = to.zeros(num_rollouts)
                    break
        elif isinstance(inner_env(env), SimEnv):
            # Create a parallel sampler when conducting a sim-to-sim experiment
            sampler = ParallelRolloutSampler(env, policy, num_workers=num_parallel_envs, min_rollouts=num_rollouts)
            ros = sampler.sample()
            for i in range(num_rollouts):
                rets_real[i] = ros[i].undiscounted_return()
        else:
            raise pyrado.TypeErr(given=inner_env(env), expected_type=[RealEnv, SimEnv])

        if save_dir is not None:
            # Save the evaluation results
            to.save(rets_real, osp.join(save_dir, f'{prefix}_returns_real.pt'))

            print_cbt('Target domain performance', bright=True)
            print(tabulate([['mean return', to.mean(rets_real).item()],
                            ['std return', to.std(rets_real)],
                            ['min return', to.min(rets_real)],
                            ['max return', to.max(rets_real)]]))

        if mc_estimator:
            return to.mean(rets_real)
        else:
            return to.from_numpy(bootstrap_ci(rets_real.numpy(), np.mean,
                                              num_reps=1000, alpha=0.05, ci_sides=1, studentized=False)[1])

    def step(self, snapshot_mode: str = 'latest', meta_info: dict = None):
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

        if self.curr_checkpoint == -2:
            # Train the initial policies in the source domain
            self.train_init_policies()
            self.reached_checkpoint()  # setting counter to -1

        if self.curr_checkpoint == -1:
            # Evaluate the initial policies in the target domain
            self.eval_init_policies()
            self.reached_checkpoint()  # setting counter to 0

        if self.curr_checkpoint == 0:
            # Normalize the input data and standardize the output data
            cands_norm = self.uc_normalizer.project_to(self.cands)
            cands_values_stdized = standardize(self.cands_values).unsqueeze(1)

            # Create and fit the GP model
            gp = SingleTaskGP(cands_norm, cands_values_stdized)
            gp.likelihood.noise_covar.register_constraint('raw_noise', GreaterThan(1e-5))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)
            print_cbt('Fitted the GP.', 'g')

            # Acquisition functions
            if self.acq_fcn_type == 'UCB':
                acq_fcn = UpperConfidenceBound(gp, beta=self.acq_param.get('beta', 0.1), maximize=True)
            elif self.acq_fcn_type == 'EI':
                acq_fcn = ExpectedImprovement(gp, best_f=cands_values_stdized.max().item(), maximize=True)
            elif self.acq_fcn_type == 'PI':
                acq_fcn = ProbabilityOfImprovement(gp, best_f=cands_values_stdized.max().item(), maximize=True)
            else:
                raise pyrado.ValueErr(given=self.acq_fcn_type, eq_constraint="'UCB', 'EI', 'PI'")

            # Optimize acquisition function and get new candidate point
            cand_norm, acq_value = optimize_acqf(
                acq_function=acq_fcn,
                bounds=to.stack([to.zeros(self.cand_dim), to.ones(self.cand_dim)]),
                q=1,
                num_restarts=self.acq_restarts,
                raw_samples=self.acq_samples
            )
            next_cand = self.uc_normalizer.project_back(cand_norm)
            print_cbt(f'Found the next candidate: {next_cand.numpy()}', 'g')
            self.cands = to.cat([self.cands, next_cand], dim=0)
            save_prefix_suffix(self.cands, 'candidates', 'pt', self.save_dir, meta_info)
            self.reached_checkpoint()  # setting counter to 1

        if self.curr_checkpoint == 1:
            # Train and evaluate a new policy, repeat if the resulting policy did not exceed the success threshold
            wrapped_trn_fcn = until_thold_exceeded(
                self.thold_succ_subrtn.item(), self.max_subrtn_rep
            )(self.train_policy_sim)
            wrapped_trn_fcn(self.cands[-1, :], prefix=f'iter_{self._curr_iter}')
            self.reached_checkpoint()  # setting counter to 2

        if self.curr_checkpoint == 2:
            # Evaluate the current policy in the target domain
            policy = load_prefix_suffix(self.policy, 'policy', 'pt', self.save_dir,
                                        meta_info=dict(prefix=f'iter_{self._curr_iter}'))
            self.curr_cand_value = self.eval_policy(
                self.save_dir, self._env_real, policy, self.mc_estimator, f'iter_{self._curr_iter}',
                self.num_eval_rollouts_real
            )
            self.cands_values = to.cat([self.cands_values, self.curr_cand_value.view(1)], dim=0)
            save_prefix_suffix(self.cands_values, 'candidates_values', 'pt', self.save_dir, meta_info)

            # Store the argmax after training and evaluating
            curr_argmax_cand = BayRn.argmax_posterior_mean(
                self.cands, self.cands_values.unsqueeze(1), self.uc_normalizer, self.acq_restarts, self.acq_samples
            )
            self.argmax_cand = to.cat([self.argmax_cand, curr_argmax_cand], dim=0)
            save_prefix_suffix(self.argmax_cand, 'candidates_argmax', 'pt', self.save_dir, meta_info)
            self.reached_checkpoint()  # setting counter to 0

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        # Policies (and value functions) of every iteration are saved by the subroutine in train_policy_sim()
        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            joblib.dump(self._env_sim, osp.join(self.save_dir, 'env_sim.pkl'))
            joblib.dump(self._env_real, osp.join(self.save_dir, 'env_real.pkl'))
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

    @staticmethod
    def argmax_posterior_mean(cands: to.Tensor,
                              cands_values: to.Tensor,
                              uc_normalizer: UnitCubeProjector,
                              num_restarts: int,
                              num_samples: int) -> to.Tensor:
        """
        Compute the GP input with the maximal posterior mean.

        :param cands: candidates a.k.a. x
        :param cands_values: observed values a.k.a. y
        :param uc_normalizer: unit cube normalizer used during the experiments (can be recovered form the bounds)
        :param num_restarts: number of restarts for the optimization of the acquisition function
        :param num_samples: number of samples for the optimization of the acquisition function
        :return: un-normalized candidate with maximum posterior value a.k.a. x
        """
        # Normalize the input data and standardize the output data
        cands_norm = uc_normalizer.project_to(cands)
        cands_values_stdized = standardize(cands_values)

        # Create and fit the GP model
        gp = SingleTaskGP(cands_norm, cands_values_stdized)
        gp.likelihood.noise_covar.register_constraint('raw_noise', GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Find position with maximal posterior mean
        cand_norm, acq_value = optimize_acqf(
            acq_function=PosteriorMean(gp),
            bounds=to.stack([to.zeros_like(uc_normalizer.bound_lo), to.ones_like(uc_normalizer.bound_up)]),
            q=1,
            num_restarts=num_restarts,
            raw_samples=num_samples
        )

        cand = uc_normalizer.project_back(cand_norm.detach())
        print_cbt(f'Converged to argmax of the posterior mean: {cand.numpy()}', 'g', bright=True)
        return cand

    @staticmethod
    def train_argmax_policy(load_dir: str,
                            env_sim: MetaDomainRandWrapper,
                            subrtn: Algorithm,
                            num_restarts: int,
                            num_samples: int,
                            policy_param_init: to.Tensor = None,
                            valuefcn_param_init: to.Tensor = None,
                            subrtn_snapshot_mode: str = 'best') -> Policy:
        """
        Train a policy based on the maximizer of the posterior mean.

        :param load_dir: directory to load from
        :param env_sim: simulation environment
        :param subrtn: algorithm which performs the policy / value-function optimization
        :param num_restarts: number of restarts for the optimization of the acquisition function
        :param num_samples: number of samples for the optimization of the acquisition function
        :param policy_param_init: initial policy parameter values for the subroutine, set `None` to be random
        :param valuefcn_param_init: initial value function parameter values for the subroutine, set `None` to be random
        :param subrtn_snapshot_mode: snapshot mode for saving during training of the subroutine
        :return: the final BayRn policy
        """
        # Load the required data
        cands = to.load(osp.join(load_dir, 'candidates.pt'))
        cands_values = to.load(osp.join(load_dir, 'candidates_values.pt')).unsqueeze(1)
        bounds = to.load(osp.join(load_dir, 'bounds.pt'))
        uc_normalizer = UnitCubeProjector(bounds[0, :], bounds[1, :])

        if cands.shape[0] > cands_values.shape[0]:
            print_cbt(
                f'There are {cands.shape[0]} candidates but only {cands_values.shape[0]} evaluations. Ignoring the'
                f'candidates without evaluation for computing the argmax.', 'y')
            cands = cands[:cands_values.shape[0], :]

        # Find the maximizer
        argmax_cand = BayRn.argmax_posterior_mean(cands, cands_values, uc_normalizer, num_restarts, num_samples)

        # Set the domain randomizer
        env_sim.adapt_randomizer(argmax_cand.numpy())

        # Reset the subroutine's algorithm which includes resetting the exploration
        subrtn.reset()

        # Do a warm start
        subrtn.init_modules(
            warmstart=True, policy_param_init=policy_param_init, valuefcn_param_init=valuefcn_param_init
        )

        subrtn.train(snapshot_mode=subrtn_snapshot_mode, meta_info=dict(suffix='argmax'))
        return subrtn.policy
