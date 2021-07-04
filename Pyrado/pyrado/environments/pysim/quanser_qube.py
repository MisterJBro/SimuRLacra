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

from abc import abstractmethod

import numpy as np
import torch as to
from init_args_serializer.serializable import Serializable

from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.quanser import max_act_qq
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import QCartPoleSwingUpRewFcn


class QQubeSim(SimPyEnv, Serializable):
    """Base Environment for the Quanser Qube swing-up and stabilization task"""

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        raise NotImplementedError

    @abstractmethod
    def _create_spaces(self):
        raise NotImplementedError

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            g=9.81,  # gravity [m/s**2]
            Rm=8.4,  # motor resistance [Ohm]
            km=0.042,  # motor back-emf constant [V*s/rad]
            Mr=0.095,  # rotary arm mass [kg]
            Lr=0.085,  # rotary arm length [m]
            Dr=5e-6,  # rotary arm viscous damping [N*m*s/rad], original: 0.0015, identified: 5e-6
            Mp=0.024,  # pendulum link mass [kg]
            Lp=0.129,  # pendulum link length [m]
            Dp=1e-6,  # pendulum link viscous damping [N*m*s/rad], original: 0.0005, identified: 1e-6
            V_thold_neg=0,  # min. voltage required to move the servo in negative the direction [V]
            V_thold_pos=0,  # min. voltage required to move the servo in positive the direction [V]
        )

    def _calc_constants(self):
        Mr = self.domain_param["Mr"]
        Mp = self.domain_param["Mp"]
        Lr = self.domain_param["Lr"]
        Lp = self.domain_param["Lp"]
        g = self.domain_param["g"]

        # Moments of inertia
        Jr = Mr * Lr ** 2 / 12  # inertia about COM of the rotary pole [kg*m^2]
        Jp = Mp * Lp ** 2 / 12  # inertia about COM of the pendulum pole [kg*m^2]

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + Mp * Lr ** 2
        self._c[1] = 0.25 * Mp * Lp ** 2
        self._c[2] = 0.5 * Mp * Lp * Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5 * Mp * Lp * g

    def _dyn(self, t, x, u):
        r"""
        Compute $\dot{x} = f(x, u, t)$.

        :param t: time (if the dynamics explicitly depend on the time)
        :param x: state
        :param u: control command
        :return: time derivative of the state
        """
        km = self.domain_param["km"]
        Rm = self.domain_param["Rm"]
        Dr = self.domain_param["Dr"]
        Dp = self.domain_param["Dp"]

        # Decompose state
        th, al, thd, ald = x
        sin_al = np.sin(al)
        sin_2al = np.sin(2 * al)

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * sin_al ** 2
        b = self._c[2] * np.cos(al)
        c = self._c[3]
        det = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = km * (u - km * thd) / Rm
        c0 = self._c[1] * sin_2al * thd * ald - self._c[2] * sin_al * ald * ald
        c1 = -0.5 * self._c[1] * sin_2al * thd * thd + self._c[4] * sin_al
        x = trq - Dr * thd - c0
        y = -Dp * ald - c1

        # Compute qdd = M^{-1} @ [x, y]
        thdd = (c * x - b * y) / det
        aldd = (a * y - b * x) / det

        return np.array([thd, ald, thdd, aldd], dtype=np.float64)

    def _step_dynamics(self, act: np.ndarray):
        # Apply a voltage dead zone, i.e. below a certain amplitude the system is will not move.
        # This is a very simple model of static friction.
        if self.domain_param["V_thold_neg"] <= act <= self.domain_param["V_thold_pos"]:
            act = 0

        # Compute the derivative
        thd, ald, thdd, aldd = self._dyn(None, self.state, act)

        # Integration step (Runge-Kutta 4)
        k = np.zeros(shape=(4, 4))  # derivatives
        k[0, :] = np.array([thd, ald, thdd, aldd])
        for j in range(1, 4):
            if j <= 2:
                s = self.state + self._dt / 2.0 * k[j - 1, :]
            else:
                s = self.state + self._dt * k[j - 1, :]
            thd, ald, thdd, aldd = self._dyn(None, self.state, act)
            k[j, :] = np.array([s[2], s[3], thdd, aldd])
        self.state += self._dt / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3])

    def _init_anim(self):
        # Import PandaVis Class
        from pyrado.environments.pysim.pandavis import QQubeVis

        # Create instance of PandaVis
        self._visualization = QQubeVis(self, self._rendering)


class QQubeSwingUpSim(QQubeSim):
    """
    Environment which models Quanser's Furuta pendulum called Quanser Qube.
    The goal is to swing up the pendulum and stabilize at the upright position (alpha = +-pi).
    """

    name: str = "qq-su"

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([115.0 / 180 * np.pi, 4 * np.pi, 20 * np.pi, 20 * np.pi])  # [rad, rad, rad/s, rad/s]
        max_obs = np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf])  # [-, -, -, -, rad/s, rad/s]
        max_init_state = np.array([2.0, 1.0, 0.5, 0.5]) / 180 * np.pi  # [rad, rad, rad/s, rad/s]

        self._state_space = BoxSpace(-max_state, max_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])
        self._obs_space = BoxSpace(
            -max_obs, max_obs, labels=["sin_theta", "cos_theta", "sin_alpha", "cos_alpha", "theta_dot", "alpha_dot"]
        )
        self._init_space = BoxSpace(
            -max_init_state, max_init_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"]
        )
        self._act_space = BoxSpace(-max_act_qq, max_act_qq, labels=["V"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        rew_fcn = QCartPoleSwingUpRewFcn(factor=[0.7, 0.25, 0.05], max_dist=1.0, max_act=6.0, scales=[np.pi, 2.0, 8.0])

        return RadiallySymmDesStateTask(self.spec, state_des, rew_fcn, idcs=[1])

    def observe(self, state, dtype=np.ndarray):
        if dtype is np.ndarray:
            return np.array(
                [np.sin(state[0]), np.cos(state[0]), np.sin(state[1]), np.cos(state[1]), state[2], state[3]]
            )
        elif dtype is to.Tensor:
            return to.cat(
                (
                    state[0].sin().view(1),
                    state[0].cos().view(1),
                    state[1].sin().view(1),
                    state[1].cos().view(1),
                    state[2].view(1),
                    state[3].view(1),
                )
            )


class QQubeStabSim(QQubeSim):
    """
    Environment which models Quanser's Furuta pendulum called Quanser Qube.
    The goal is to stabilize the pendulum at the upright position (alpha = +-pi).

    .. note::
        This environment is only for testing purposes, or to find the PD gains for stabilizing the pendulum at the top.
    """

    name: str = "qq-st"

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([120.0 / 180 * np.pi, 4 * np.pi, 20 * np.pi, 20 * np.pi])  # [rad, rad, rad/s, rad/s]
        min_init_state = np.array([-5.0 / 180 * np.pi, 175.0 / 180 * np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]
        max_init_state = np.array([5.0 / 180 * np.pi, 185.0 / 180 * np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]

        self._state_space = BoxSpace(-max_state, max_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])
        self._obs_space = self._state_space
        self._init_space = BoxSpace(min_init_state, max_init_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])
        self._act_space = BoxSpace(-max_act_qq, max_act_qq, labels=["V"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        rew_fcn = QCartPoleSwingUpRewFcn(factor=0.8, max_dist=1.0, max_act=5.0, scales=[np.pi, 2.0])

        return RadiallySymmDesStateTask(self.spec, state_des, rew_fcn, idcs=[1])

    def observe(self, state, dtype=np.ndarray):
        # Directly observe the noise-free state
        return state.copy()
