"""Treino com Stable-Baselines3 (próximas etapas).

Uso:
    python scripts/treinar.py --algo ppo --passos 500000
    python scripts/treinar.py --algo sac --passos 300000
"""
import argparse

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

import braco4  # noqa: F401  (registra Braco4-v0)

ALGOS = {"ppo": PPO, "sac": SAC}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS, default="ppo")
    ap.add_argument("--passos", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_envs = 8 if args.algo == "ppo" else 1
    env = make_vec_env("Braco4-v0", n_envs=n_envs, seed=args.seed)
    modelo = ALGOS[args.algo]("MlpPolicy", env, verbose=1, seed=args.seed)
    modelo.learn(total_timesteps=args.passos)
    modelo.save(f"resultados/{args.algo}_braco4")
