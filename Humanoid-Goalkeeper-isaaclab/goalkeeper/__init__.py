"""Goalkeeper environment package — register with gymnasium."""
import gymnasium as gym

from . import agents
from .goalkeeper_env import GoalkeeperEnv
from .goalkeeper_env_cfg import GoalkeeperEnvCfg

gym.register(
    id="Isaac-Goalkeeper-Direct-v0",
    entry_point="goalkeeper:GoalkeeperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GoalkeeperEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:GoalkeeperPPORunnerCfg",
    },
)

__all__ = ["GoalkeeperEnv", "GoalkeeperEnvCfg"]
