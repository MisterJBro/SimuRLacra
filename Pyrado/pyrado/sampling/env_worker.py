import numpy as np
from copy import deepcopy
from multiprocessing import Process, Pipe
from pyrado.environments.base import Env


class Worker(Process):
    """
    Worker Process that has a list of environments which it manages.
    It receives commands from envs, what to do with these environments.
    Workers simulate in parallel and independently without sharing anything between
    workers to improve efficiency.
    """

    def __init__(self, env: Env, channel: Pipe, idx: int, env_num: int):
        """
        Constructor
        :param env: environment to parallelize
        :param channel: channel for communication between worker and master process
        :param idx: idx of worker
        :param env_num: number of environments for parallel sampling
        """
        super(Worker, self).__init__()
        self.channel = channel
        self.idx = idx
        self.obss = None

        self.env_num = env_num
        self.env = env

    def run(self):
        """ Creates a list of environments, and waits for commands to simulate a step, to reset or close them. """
        self.envs = [deepcopy(self.env) for _ in range(self.env_num)]

        command, acts = self.channel.recv()
        while command != "close":
            if command == "reset":
                self.reset()
            elif command == "step":
                self.step(acts)

            command, acts = self.channel.recv()

        self.close()

    def reset(self):
        """ Resets all environments. """
        obss = np.array([env.reset() for env in self.envs])
        self.channel.send(obss)

    def step(self, acts: np.array):
        """
        Executes a step on all own environments.
        :param acts: actions for step simulation
        """
        n_obss, rews, dones = [], [], []

        for i in range(self.env_num):
            act = np.array([acts[i]]).reshape(-1)
            n_obs, rew, done, _ = self.envs[i].step(act)

            if done:
                n_obs = self.envs[i].reset()

            n_obss.append(n_obs)
            rews.append(rew)
            dones.append(done)
        n_obss, rews, dones = np.array(n_obss), np.array(rews), np.array(dones)

        self.channel.send((n_obss, rews, dones))

    def close(self):
        """ Closes all own environments. """
        for env in self.envs:
            env.close()