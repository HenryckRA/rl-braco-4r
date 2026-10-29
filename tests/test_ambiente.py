import numpy as np
from gymnasium.utils.env_checker import check_env

from braco4 import Braco4Env


def test_api_gymnasium():
    check_env(Braco4Env(), skip_render_check=True)


def test_alvo_alcancavel_e_acima_do_chao():
    env = Braco4Env()
    for s in range(20):
        env.reset(seed=s)
        assert env.alvo_pos[2] > 0.05
        assert -np.pi / 2 <= env.alvo_ang <= np.pi / 2


def test_episodio_aleatorio():
    env = Braco4Env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (21,)
    for _ in range(env.max_passos):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r) and np.all(np.isfinite(obs))
        if term or trunc:
            break


def test_sem_torque_o_braco_cai():
    """Com torque zero a gravidade deve puxar o braço (sanidade da física)."""
    env = Braco4Env()
    env.reset(seed=1)
    z0 = env._ponta()[0][2]
    for _ in range(25):
        env.step(np.zeros(4))
    assert env._ponta()[0][2] < z0
