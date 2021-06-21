"""
Tests a policy
"""
import torch as to
from multiprocessing import freeze_support

import pyrado
from pyrado.algorithms.step_based.ppo_gae import PPOGAE
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.utils.data_types import RenderMode
from pyrado.policies.recurrent.rnn import LSTMPolicy
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.experiments import load_experiment
from pyrado.logger.experiment import Experiment, ask_for_experiment
import multiprocessing as mp


if __name__ == "__main__":
    ex_dir = ask_for_experiment()
    env, pol, _ = load_experiment(ex_dir)
    print(pol)

    while True:
        input("Press some key to continue:")
        ro = rollout(env, pol, render_mode=RenderMode(text=True, video=True))

        print(f"Return: {ro.undiscounted_return()}")
