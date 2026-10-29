"""Ambiente Gymnasium (física no MuJoCo): braço de 4 juntas controlado por torque.

Tarefa: levar a ponta até um alvo 3D chegando com um ângulo de elevação pedido
(ex.: "por cima"). São 4 requisitos (x, y, z, ângulo) para 4 juntas: sem redundância.

Estado (21): cos q (4), sin q (4), q̇ (4), posição do alvo (3), cos/sin φ* (2),
             erro de posição Δp (3), erro de ângulo Δφ (1)
Ação    (4): a ∈ [-1, 1]⁴  →  τ_i = gear_i · a_i (gear definido no XML)
Recompensa:  r = −d − w·|Δφ| − λ‖a‖²  (+ R_suc no sucesso, − R_lim em violação)
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

XML = Path(__file__).with_name("braco4.xml")


def _ajustar_angulo(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class Braco4Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        frame_skip=10,          # 10 × 2 ms = agente decide a cada 20 ms
        max_passos=200,         # 4 s por tentativa
        w_angulo=0.1,
        lambda_tau=0.01,
        tol_pos=0.02,           # [m]
        tol_ang=np.deg2rad(10),
        tol_vel=0.5,            # ‖q̇‖ máx. para considerar "parado" [rad/s]
        vel_max=10.0,           # limite de velocidade das juntas [rad/s]
        r_sucesso=10.0,
        r_limite=10.0,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self.frame_skip = frame_skip
        self.max_passos = max_passos
        self.w_angulo = w_angulo
        self.lambda_tau = lambda_tau
        self.tol_pos, self.tol_ang, self.tol_vel = tol_pos, tol_ang, tol_vel
        self.vel_max = vel_max
        self.r_sucesso, self.r_limite = r_sucesso, r_limite

        self._id_ponta = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ponta")
        self._id_alvo = self.model.body("alvo").mocapid[0]
        self._lim = self.model.jnt_range.copy()  # limites das juntas [rad]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _ponta(self):
        """Posição da ponta e ângulo de elevação da mão (0 = horizontal, −90° = para baixo)."""
        pos = self.data.site_xpos[self._id_ponta].copy()
        eixo = self.data.site_xmat[self._id_ponta].reshape(3, 3)[:, 2]  # eixo z local da mão
        return pos, float(np.arcsin(np.clip(eixo[2], -1.0, 1.0)))

    def _config_aleatoria(self):
        """Sorteia ângulos dentro dos limites com a ponta acima do chão."""
        while True:
            q = self.np_random.uniform(self._lim[:, 0], self._lim[:, 1])
            self.data.qpos[:] = q
            mujoco.mj_kinematics(self.model, self.data)
            pos, ang = self._ponta()
            if pos[2] > 0.05:
                return q, pos, ang

    def _obs(self):
        q, qd = self.data.qpos, self.data.qvel
        pos, ang = self._ponta()
        return np.concatenate([
            np.cos(q), np.sin(q), qd,
            self.alvo_pos, [np.cos(self.alvo_ang), np.sin(self.alvo_ang)],
            pos - self.alvo_pos, [_ajustar_angulo(ang - self.alvo_ang)],
        ]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # alvo = pose alcançável (garante que existe solução com o ângulo pedido)
        _, self.alvo_pos, self.alvo_ang = self._config_aleatoria()
        self.data.mocap_pos[self._id_alvo] = self.alvo_pos
        # pose inicial independente
        q0, _, _ = self._config_aleatoria()
        self.data.qpos[:] = q0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.passo = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = a
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.passo += 1

        pos, ang = self._ponta()
        d = float(np.linalg.norm(pos - self.alvo_pos))
        erro_ang = abs(_ajustar_angulo(ang - self.alvo_ang))
        recompensa = -d - self.w_angulo * erro_ang - self.lambda_tau * float(a @ a)

        sucesso = (d < self.tol_pos and erro_ang < self.tol_ang
                   and np.linalg.norm(self.data.qvel) < self.tol_vel)
        violacao = bool(np.any(np.abs(self.data.qvel) > self.vel_max) or pos[2] < 0.0)
        if sucesso:
            recompensa += self.r_sucesso
        if violacao:
            recompensa -= self.r_limite

        terminated = bool(sucesso or violacao)
        truncated = self.passo >= self.max_passos
        info = {"distancia": d, "erro_angulo": erro_ang,
                "sucesso": bool(sucesso), "violacao": violacao}
        return self._obs(), recompensa, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
