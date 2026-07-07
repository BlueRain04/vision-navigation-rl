import gymnasium as gym
from .drone_env import QuadcopterEnv
from .drone_env_cfg import QuadcopterEnvCfg

gym.register(
    id="Drone-Nav-Direct-v0",
    entry_point=f"{__name__}.drone_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_cfg:QuadcopterEnvCfg",
    },
)
