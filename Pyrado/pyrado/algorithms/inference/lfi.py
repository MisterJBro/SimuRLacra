from typing import Optional, Callable, Type

import joblib
import pyrado
import torch as to
import os.path as osp

from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger, TensorBoardPrinter, LoggerAware
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe import PosteriorEstimator
from sbi.inference.base import simulate_for_sbi
from sbi.user_input.user_input_checks import prepare_for_sbi
from torch.distributions import Distribution


class EnvSimulator(Callable):
    """
    Mapping from the environment system parameters to a trajectory-based rollout using a control-policy.
    """

    def __init__(
        self,
        env: Env,
        policy: Policy,
        param_names: list,
    ):
        self.name = env.name
        self.env = env
        self.policy = policy
        self.param_names = param_names

    def __call__(self, params):
        ro = rollout(
            self.env,
            self.policy,
            eval=True,
            reset_kwargs=dict(domain_param=dict(zip(self.param_names, params.squeeze()))),
        )
        return to.tensor(ro.observations).view(-1, 1).squeeze().to(dtype=to.float32)


class LFI(LoggerAware):
    """
    SBI-Wrapper.
    This class currently only works with posterior estimators and currently excludes
    likelihood- and density-ratio-estimators. This might be added later.
    Examplary file in '/pyrado/scripts/lfi/....py'
    """

    def __init__(
        self,
        save_dir: str,
        simulator: Callable,
        params_names,
        prior: Distribution,
        inference: Type[PosteriorEstimator] = None,
        flow: Callable[[], DirectPosterior] = None,
        posterior: DirectPosterior = None,
        max_iter: int = 5,
        num_sim: int = 1000,
        logger: Optional[StepLogger] = None,
        save_name: str = "algo",
    ):
        self._save_dir = save_dir
        self.posterior = posterior
        self.simulator = simulator
        self.prior = prior
        self._ddp_policy = None
        self.ddp_policy_params = None
        self.params_names = params_names
        self._max_iter = max_iter
        self._curr_iter = 0
        self._num_sim = num_sim
        self._cnt_samples = 0
        self.batch_simulator, self.batch_prior = prepare_for_sbi(self.simulator, self.prior)

        self._save_name = save_name

        if logger is not None:
            self._logger = logger

        # sbi should use the same summary writer as this algo
        summary_writer = None
        for p in self.logger.printers:
            if isinstance(p, TensorBoardPrinter):
                summary_writer = p.writer

        self.inference = None
        if inference is not None:
            self.inference = inference(prior=self.prior, density_estimator=flow, summary_writer=summary_writer)

    def set_posterior(self, posterior: DirectPosterior):
        """
        Set posterior from the outside if a learned model exists.
        """
        self.posterior = posterior

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Trains the posterior using SNPE using observed rollouts and the prior distribution

        """
        # TODO: raise exception if flow, prior or inference is not given. In this case only the posterior is given
        #  and should only be used for evaluation

        # get real-world rollouts from meta_info
        rollouts_real = meta_info["rollouts_real"]

        # set proposal prior
        proposal_prior = self.batch_prior
        if self._curr_iter == 0:
            theta, x = simulate_for_sbi(
                self.batch_simulator, proposal_prior, num_simulations=self._num_sim, simulation_batch_size=1
            )
            _ = self.inference.append_simulations(theta, x).train()
            proposal_prior = self.inference.build_posterior()
            self._curr_iter += 1
        while self._curr_iter < self._max_iter:
            # generate data to train
            if len(rollouts_real) == 1:
                theta, x = simulate_for_sbi(
                    self.batch_simulator, proposal_prior, num_simulations=self._num_sim, simulation_batch_size=1
                )
                self.inference.append_simulations(theta, x)
            else:
                for ro in rollouts_real:
                    proposal_prior.set_default_x(ro)
                    theta, x = simulate_for_sbi(
                        self.batch_simulator, proposal_prior, num_simulations=self._num_sim, simulation_batch_size=1
                    )
                    self.inference.append_simulations(theta, x)

            _ = self.inference.train()
            # set proposal prior
            proposal_prior = self.inference.build_posterior()
            self._curr_iter += 1
            self.logger.add_value("Current Iteration", self._curr_iter)
        self.posterior = proposal_prior
        self.make_snapshot(snapshot_mode=snapshot_mode, meta_info=meta_info)

    def evaluate(
        self,
        obs_traj: to.Tensor,
        num_samples: int = 1000,
        compute_quantity: dict = None,
    ):
        """
        Evaluates the posterior by calculating parameter samples given observed data, its log probability
        and the simulated trajectory.
        """
        # default computations
        compute_dict = {"log_prob": False, "sample_params": False, "sim_traj": False}
        if compute_quantity is not None:
            # TODO: raise exception if keys do not fit given quantities
            compute_dict.update(compute_quantity)

        # generate sample parameters
        prop_params = to.stack([self.posterior.sample((num_samples,), x=obs) for obs in obs_traj], dim=0)

        log_prob, sim_traj = None, None
        num_obs = obs_traj.shape[0]
        len_obs = obs_traj.shape[1]
        if compute_dict["sim_traj"]:
            sim_traj = to.empty((num_obs, num_samples, len_obs))
        if compute_dict["log_prob"]:
            log_prob = to.empty((num_obs, num_samples))

        cnt = 0
        for o in range(num_obs):
            # compute log probability
            if compute_dict["log_prob"]:
                log_prob[o, :] = self.posterior.log_prob(prop_params[o, :, :], x=obs_traj[o, :])

            # compute trajectories
            if compute_dict["sim_traj"]:
                for s in range(num_samples):
                    if not s % 10:
                        print(
                            "\r[train_lfi.py/evaluate_lfi] Observation: ({}|{}), Sample: ({}|{})".format(
                                o, num_obs, s, num_samples
                            ),
                            end="",
                        )
                    # compute trajectories for each observation and every sample
                    sim_traj[o, s, :] = self.simulator(prop_params[o, s, :].unsqueeze(0))
            cnt += 1
        if not compute_dict["sample_params"]:
            prop_params = None
        return prop_params, log_prob, sim_traj

    def make_snapshot(self, snapshot_mode: str, meta_info: dict = None):
        """
        Make a snapshot of the training progress.
        This method is called from the subclasses and delegates to the custom method `save_snapshot()`.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        self.save_snapshot(meta_info)
        if snapshot_mode == "latest":
            self.save_snapshot(meta_info)
        else:
            raise pyrado.ValueErr(given=snapshot_mode, eq_constraint="'latest', 'best', or 'no'")

    def save_snapshot(self, meta_info: dict = None):
        # joblib.dump(self, osp.join(self._save_dir, f"{self._save_name}.pkl"))

        pyrado.save(self.posterior, "posterior", "pt", self._save_dir, meta_info, use_state_dict=False)

        if "rollouts_real" not in meta_info:
            raise pyrado.KeyErr(keys="rollouts_real", container=meta_info)
        pyrado.save(meta_info["rollouts_real"], "rollouts_real", "pkl", self._save_dir, meta_info)
