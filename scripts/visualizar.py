"""Abre o visualizador do MuJoCo e roda o braço.

Uso:
    python scripts/visualizar.py                      # ações aleatórias
    python scripts/visualizar.py resultados/sac_braco4.zip   # política treinada
"""
import sys
import time

import mujoco.viewer

from braco4 import Braco4Env

env = Braco4Env()
politica = None
if len(sys.argv) > 1:
    from stable_baselines3 import PPO, SAC
    algo = SAC if "sac" in sys.argv[1] else PPO
    politica = algo.load(sys.argv[1])

obs, _ = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        a = env.action_space.sample() if politica is None else politica.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)
        viewer.sync()
        time.sleep(env.frame_skip * env.model.opt.timestep)
        if term or trunc:
            obs, _ = env.reset()
