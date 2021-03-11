import torch as to
import numpy as np
from multiprocessing import Pipe
from pyrado.sampling.buffer import Buffer
from pyrado.sampling.env_worker import Worker
from pyrado.environments.base import Env
from typing import List

class Envs:
    """
    Central instance to manage all environment workers. Gives them commands in parallel.
    """

    def __init__(self, cpu_num: int, env_num: int, env: Env, game_len: int, gamma: float, lam: float, env_list: List = []):
        """
        Constructor

        :param cpu_num: number of used cpu cores
        :param env_num: number of environments for parallel sampling
        :param env: environment for simulation
        :param game_len: max length of trajectory
        :param gamma: discount factor
        :param lam: lambda factor for GAE
        """
        assert (
            cpu_num > 0 and env_num >= cpu_num
        ), "CPU num has to be greater 0 and env num has to be greater or equal to env num!"

        self.env_num = env_num
        test_env = env
        self.obs_dim = (test_env.obs_space.flat_dim,)
        self.act_num = (test_env.act_space.flat_dim,)
        del test_env

        self.cpu_num = cpu_num
        self.channels = [Pipe() for _ in range(cpu_num)]
        self.env_num_worker = int(env_num / cpu_num)
        self.rest_env_num = (env_num % cpu_num) + self.env_num_worker
        self.workers = [
            Worker(env_list[i] if len(env_list) == cpu_num else env, self.channels[i][1], i, self.rest_env_num if i == cpu_num - 1 else self.env_num_worker)
            for i in range(cpu_num)
        ]
        [w.start() for w in self.workers]

        self.buf = Buffer(self.obs_dim, self.act_num, game_len, gamma, lam, env_num)
        self.obss = None

    def reset(self):
        """ Resets all workers. """
        self.buf.reset()
        [c[0].send(["reset", None]) for c in self.channels]
        msg = [c[0].recv() for c in self.channels]

        self.obss = np.concatenate(msg, axis=0)
        return self.obss

    def step(self, acts: np.ndarray, vals: np.ndarray):
        """
        Executes a step on all workers and returns the results.

        :param acts: joints actions for all workers
        :param vals: predicted values
        """
        [
            c[0].send(
                [
                    "step",
                    acts[
                        i * self.env_num_worker : self.env_num
                        if i == self.cpu_num - 1
                        else (i + 1) * self.env_num_worker
                    ],
                ]
            )
            for i, c in enumerate(self.channels)
        ]
        msg = [c[0].recv() for c in self.channels]
        obs_msg, rew_msg, done_ind = [], [], []
        for i, (o, r, d) in enumerate(msg):
            obs_msg.append(o)
            rew_msg.append(r)

            for j in range(self.env_num_worker):
                if d[j]:
                    index = j + self.env_num_worker * i
                    self.buf.sections[index].append(self.buf.ptr + 1)
                    done_ind.append(index)

        rews = np.concatenate(rew_msg, axis=0)
        n_obss = np.concatenate(obs_msg, axis=0)
        self.buf.store(self.obss, acts, rews, vals, done_ind)
        self.obss = n_obss

        return n_obss, done_ind

    def close(self):
        """ Closes all workers and their environments. """
        [c[0].send(["close", None]) for c in self.channels]

    def ret_and_adv(self) -> [np.ndarray, np.ndarray]:
        """ Calculates the return and advantages in the buffer. """
        self.buf.ret_and_adv()
        rews = self.buf.get_rews()
        rets = np.array([np.sum(r) for r in rews])
        all_lengths = np.array([len(r) for r in rews])
        return rets, all_lengths

    def get_data(self, device: to.device):
        """
        Get the buffer data as tensors.

        :param device: device for tensors
        """
        data_tensors = to_tensors(self.buf.get_data(), device)
        data_tensors.append(self.buf.sections)
        return data_tensors


def to_tensors(arrays: np.ndarray, device: to.device):
    """
    Converts the array of numpy arrays into an array of tensors.

    :param device: device for tensors
    """
    tensors = []

    for a in arrays:
        tensor = to.as_tensor(a, dtype=to.float32).to(device)
        tensors.append(tensor)

    return tensors
