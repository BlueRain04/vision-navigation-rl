import gymnasium as gym
from drone.learning.skrl.agent import get_agent
from skrl.trainers.torch import SequentialTrainer

env = gym.make("Drone-Nav-Direct-v0", num_envs=16)
agent = get_agent(env, device="cuda")
agent.init()
trainer_cfg = {"timesteps": 10000, "headless": True}
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
trainer.train()
