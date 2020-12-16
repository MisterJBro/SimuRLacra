from collections import Callable
import torch as to
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import simulate_for_sbi
from sbi.inference.posteriors.direct_posterior import DirectPosterior
import log_prob_plot


def train_lfi(
    simulator,
    inference,
    prior,
    x_o,
    num_samples,
    num_rounds: int = 5,
    num_sim: int = 1000,
    summary: SummaryWriter = None,
    eval_plot: bool = True,
    num_plot: int = 4,
):
    for _ in range(num_plot):
        proposal_prior = prior
        for _ in range(num_rounds):
            theta, x = simulate_for_sbi(simulator, proposal_prior, num_simulations=num_sim, simulation_batch_size=1)
            _ = inference.append_simulations(theta, x).train()
            proposal_prior = inference.build_posterior().set_default_x(x_o)
        posterior = proposal_prior

        # plot log prob for different number of simulations
        if eval_plot == True:
            samples = posterior.sample((num_samples,), x=x_o)
            sum_samples = sum(samples) / len(samples)
            log_prob_plot.add_plot(posterior, sum_samples, x_o, num_sim)
            num_sim = num_sim * 2

    if eval_plot == True:
        log_prob_plot.plot_log()

    return posterior


def evaluate_lfi(simulator: Callable, posterior: DirectPosterior, observations, num_samples: int = 1000):
    """
        INPUT:
    ...
    observations:   to.Tensor[num_observations, trajectory_size]

        OUTPUT:
    proposals:      to.Tensor[num_observations, num_samples, parameter_size]
    log_prob:       to.Tensor[?, ?]
    trajectories:   to.Tensor[num_observations, num_samples, trajectory_size]
    """
    num_observations = observations.shape[0]
    trajectory_size = observations.shape[1]

    # compute proposals for each observation
    proposals = to.stack([posterior.sample((num_samples,), x=obs) for obs in observations], dim=0)

    print()
    trajectories = to.empty((num_observations, num_samples, trajectory_size))
    log_prob = to.empty((num_observations, num_samples))
    cnt = 0
    for o in range(num_observations):
        # compute log probability
        log_prob[o, :] = posterior.log_prob(proposals[o, :, :], x=observations[o, :])
        for s in range(num_samples):
            if not s % 10:
                print(
                    "\r[train_lfi.py/evaluate_lfi] Observation: ({}|{}), Sample: ({}|{})".format(
                        o, num_observations, s, num_samples
                    ),
                    end="",
                )
            # compute trajectories for each observation and every sample
            trajectories[o, s, :] = simulator(proposals[o, s, :].unsqueeze(0))
        cnt += 1

    return proposals, log_prob, trajectories


def save_nn(nn: to.nn.Module, path: str):
    to.save(nn.state_dict(), path)
    print("saved model at:\t {}".format(path))


def load_nn(nn, path: str):
    state_dict = to.load(path)
    nn.load_state_dict(state_dict=state_dict)
    print("loaded model from: {}".format(path))
