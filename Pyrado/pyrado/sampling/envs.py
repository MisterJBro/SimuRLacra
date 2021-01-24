import torch
import numpy as np
from multiprocessing import Pipe
from pyrado.sampling.buffer import Buffer
from pyrado.sampling.env_worker import Worker


class Envs:
    """
    Central instance to manage all environment workers. Gives them commands in parallel.
    """

    def __init__(self, cpu_num, env_num, env, game_len, gamma, lam):
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
            Worker(env, self.channels[i][1], i, self.rest_env_num if i == cpu_num - 1 else self.env_num_worker)
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

    def step(self, acts, vals):
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
        obs_msg, rew_msg = [], []
        for i, (o, r, d) in enumerate(msg):
            obs_msg.append(o)
            rew_msg.append(r)

            for j in range(self.env_num_worker):
                if d[j]:
                    index = j + self.env_num_worker * i
                    self.buf.sections[index].append(self.buf.ptr + 1)

        rews = np.concatenate(rew_msg, axis=0)
        n_obss = np.concatenate(obs_msg, axis=0)
        self.buf.store(self.obss, acts, rews, vals)
        self.obss = n_obss

        return n_obss

    def close(self):
        """ Closes all workers and their environments. """
        [c[0].send(["close", None]) for c in self.channels]

    def ret_and_adv(self):
        """ Calculates the return and advantages in the buffer. """
        self.buf.ret_and_adv()
        return self.buf.avg_rew()

    def get_data(self, device):
        """
        Get the buffer data as tensors.

        :param device: device for tensors
        """
        return to_tensors(self.buf.get_data(), device)


def to_tensors(arrays, device):
    """
    Converts the array of numpy arrays into an array of tensors.

    :param device: device for tensors
    """
    tensors = []

    for a in arrays:
        tensor = torch.as_tensor(a, dtype=torch.float32).to(device)
        tensors.append(tensor)

    return tensors
