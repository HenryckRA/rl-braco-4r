from gymnasium.envs.registration import register

from .ambiente import Braco4Env

register(id="Braco4-v0", entry_point="braco4.ambiente:Braco4Env")

__all__ = ["Braco4Env"]
