import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

class DroneSharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0,
                 reduction="sum", **kwargs):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        #expected input: (N, T, H, W, 7)
        #channels: 0-2 (RGB), 3-6 (States: GoalX, GoalY, GoalZ, Dist)
        self.rgb_ch = 3
        self.state_ch = 4
        
        #auto-detect shape or fallback
        try:
            self.t_steps = observation_space.shape[0]
            self.h = observation_space.shape[1]
            self.w = observation_space.shape[2]
        except:
            self.t_steps = 3
            self.h = 64
            self.w = 64

        #1 visual encoder (shared over time steps)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.rgb_ch, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),  # 64 -> 30
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),           # 30 -> 13
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.ReLU(),         # 13 -> 5
            nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.BatchNorm2d(256), nn.ReLU(),        # 5 -> 2
            nn.Flatten()
        )

        #calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, self.rgb_ch, self.h, self.w)
            self.cnn_out = self.cnn(dummy).shape[1]

        #2 fusion network
        #inputs: (T steps * CNN features) + (state vector)
        input_dim = (self.t_steps * self.cnn_out) + self.state_ch
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        #3 heads
        self.policy_mean = nn.Linear(256, action_space.shape[0])
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
        self.value_head = nn.Linear(256, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs, role=""):
        #1 unpack observation
        #obs shape: (N, T, H, W, 7)
        obs = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        
        #2 extract components
        #RGB: First 3 channels across all T, H, W
        rgb_stack = obs[..., :self.rgb_ch] 
        
        #state: last 3 channels
        #since these were expanded, we only need one vector per environment
        #Take T=0, H=0, W=0
        state_vec = obs[:, 0, 0, 0, -self.state_ch:]  #shape (N, 4)

        #3 process images
        cnn_feats = []
        for t in range(self.t_steps):
            #extract frame t: (N, H, W, 3)
            frame = rgb_stack[:, t, ...]
            #permute for Torch Conv2d: (N, 3, H, W)
            frame = frame.permute(0, 3, 1, 2)
            cnn_feats.append(self.cnn(frame))
        
        #concatenate time steps: (N, T*CNN_Out)
        visual_emb = torch.cat(cnn_feats, dim=1)
        
        #4 fuse
        joint_emb = torch.cat([visual_emb, state_vec], dim=1)
        shared = self.net(joint_emb)

        #5 output
        if role == "policy":
            return self.policy_mean(shared), self.log_std, {}
        elif role == "value":
            return self.value_head(shared), {}