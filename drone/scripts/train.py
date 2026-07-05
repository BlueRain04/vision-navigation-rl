import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import drone
from drone.drone_env_cfg import DroneNavEnvCfg
from drone.learning.skrl.agent import get_agent
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.wrappers.torch import wrap_env

env_cfg = DroneNavEnvCfg()
env_cfg.scene.num_envs = 16
env_cfg.sim.device = "cuda:0"

env = gym.make("Drone-Nav-Direct-v0", cfg=env_cfg)
env = wrap_env(env.unwrapped)
agent = get_agent(env, device="cuda:0")
agent.init()

trainer_cfg = {"timesteps": 10000, "headless": True}
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
trainer.train()

simulation_app.close()
