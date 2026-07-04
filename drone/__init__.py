import gymnasium as gym
from .drone_env import DroneNavEnv
from .drone_env_cfg import DroneNavEnvCfg

gym.register(
    id="Drone-Nav-Direct-v0",
    entry_point=f"{__name__}.drone_env:DroneNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_env_cfg:DroneNavEnvCfg",
    },
)