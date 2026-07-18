import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from .models import DroneSharedModel

def get_agent(env, device, experiment_cfg=None):
    """
    constructs the PPO agent for drone navigatio
    """
    #configuration
    rollout_length = 256  #steps per environment before update "check later if needed"
    
    #1 memory
    memory = RandomMemory(memory_size=rollout_length, num_envs=env.num_envs, device=device)

    #2 model
    models = {}
    models["policy"] = DroneSharedModel(env.observation_space, env.action_space, device)
    models["value"]  = models["policy"]  #shared backbone

    #3 PPO config
    cfg = PPO_DEFAULT_CONFIG.copy()
    
    cfg["rollouts"]         = rollout_length
    cfg["learning_epochs"]  = 8 
    cfg["mini_batches"]     = 8 
    cfg["discount_factor"]  = 0.99
    cfg["lambda"]           = 0.95
    
    #learning rate
    cfg["learning_rate"] = 3.0e-4 
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01} 
    
    #clips
    cfg["grad_norm_clip"]      = 1.0
    cfg["ratio_clip"]          = 0.2
    cfg["value_clip"]          = 0.2
    cfg["clip_predicted_values"] = True
    
    #loss scaling
    cfg["entropy_loss_scale"] = 0.01 
    cfg["value_loss_scale"]   = 1.0
    
    #preprocessors
    #image/tensor input -> no scaler (handled by model/norm)
    cfg["state_preprocessor"] = None 
    #value output -> standard scaler helps PPO convergence
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device} 
    
    #logging
    cfg["experiment"]["write_interval"]      = 100 
    cfg["experiment"]["checkpoint_interval"] = 1000
    cfg["experiment"]["wandb"]               = False 
    if experiment_cfg:
        cfg["experiment"]["directory"] = experiment_cfg.get("directory", "logs/skrl/drone")

    #4 create agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    return agent
