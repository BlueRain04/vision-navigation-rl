import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import drone
from drone.drone_env_cfg import QuadcopterEnvCfg
from drone.learning.skrl.agent import get_agent
from skrl.trainers.torch import SequentialTrainer
from isaaclab_rl.skrl import SkrlVecEnvWrapper

env_cfg = QuadcopterEnvCfg()
env_cfg.scene.num_envs = 64
env_cfg.sim.device = "cuda:0"

env = gym.make("Drone-Nav-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
if args_cli.video:
    video_kwargs = {
        "video_folder": "videos",
        "step_trigger": lambda step: step == 0,
        "video_length": 500,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
env = SkrlVecEnvWrapper(env, ml_framework="torch")
agent = get_agent(env, device="cuda:0")
agent.init()

checkpoint_path = "/workspace/vision-navigation-rl/runs/26-07-17_08-10-59-185986_PPO/checkpoints/best_agent.pt"
agent.load(checkpoint_path)

trainer_cfg = {"timesteps": 10000, "headless": True}
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
trainer.train()

simulation_app.close()
