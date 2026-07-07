import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

class DroneSharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0,
                 reduction="sum", **kwargs):
        
        Model.__init__(self, observation_space, action_space, device) #initilize the base model
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction) #initilize the stochastic policy
        DeterministicMixin.__init__(self, clip_actions) #initilize the deterministic policy

        #expected input: three tensors, RGB (, 3, H, W), depth (, 3, V), state (, X, Y, Z, dist)
        #channels: 3 (RGB), 4 (States: GoalX, GoalY, GoalZ, Dist), 1 (depth: depth value)
        self.rgb_ch = 3
        self.state_ch = 4
        self.depth_ch = 1
        
        #auto-detect shape or fallback
        try:
            self.t_steps_rgb = observation_space["rgb"].shape[0] #the history length
            self.h_rgb = observation_space["rgb"].shape[1] #the hight of an image
            self.w_rgb = observation_space["rgb"].shape[2] #the width of the image
            self.t_steps_depth = observation_space["depth"].shape[0]
            self.h_depth = observation_space["depth"].shape[1]
            self.w_depth = observation_space["depth"].shape[2]
        except:
            self.t_steps_rgb = 3
            self.h_rgb = 64
            self.w_rgb = 64
            self.t_steps_depth = 3
            self.h_depth = 64
            self.w_depth = 64

        #1 visual encoder (shared over time steps)
        self.cnn_rgb = nn.Sequential(
            nn.Conv2d(self.rgb_ch, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),  # 64 -> 30
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),           # 30 -> 13
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.ReLU(),         # 13 -> 5
            nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.BatchNorm2d(256), nn.ReLU(),        # 5 -> 2
            nn.Flatten()
        )
                     
        self.cnn_depth = nn.Sequential(
            nn.Conv2d(self.depth_ch, 16, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),  # 64 -> 30
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),           # 30 -> 13
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.ReLU(),         # 13 -> 5
            nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.BatchNorm2d(256), nn.ReLU(),        # 5 -> 2
            nn.Flatten()
        )

        #calculate CNN output size after concatnating rgb with depth
        with torch.no_grad():
            dummy_rgb = torch.zeros(1, self.rgb_ch, self.h_rgb, self.w_rgb)
            self.cnn_out_rgb = self.cnn_rgb(dummy_rgb).shape[1]

         with torch.no_grad():
            dummy_depth = torch.zeros(1, self.depth_ch, self.h_depth, self.w_depth)
            self.cnn_out_depth = self.cnn_depth(dummy_depth).shape[1]

        self.cnn_out = self.cnn_out_rgb + self.cnn_out_depth

        #2 fusion network
        #inputs: (T steps * CNN features) + (state vector)
        input_dim = (self.t_steps_rgb * self.cnn_out) + self.state_ch #since t_steps_rgb == t_steps_depth
        
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
        obs = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        
        #2 extract components, rgb_stack (batch_size, 64, 64, 3), depth_stack (batch_size, 64, 64, 1)
        rgb_stack = obs["rgb"]
        depth_stack = obs["depth"]
        state_vec = obs["state"]
        #RGB: First 3 channels across all T, H, W
        #rgb_stack = image_rgb[..., :self.rgb_ch]
        
        #state: last 3 channels
        #since these were expanded, we only need one vector per environment
        #Take T=0, H=0, W=0
        #state_vec = obs[:, 0, 0, 0, -self.state_ch:]  #shape (N, 4)

        #3 process images
        cnn_feats = [] #not sure if we add the rgb and depth in cnn_feat for each time step (in history) or add all the rgb frames (3) then add the depth frames (3)...
        for t in range(self.t_steps_rgb): #since t_steps_rgb == t_steps_depth
            #extract frame t: (N, H, W, 3)
            frame_rgb = rgb_stack[:, t, ...]
            frame_depth = depth_stack[:, t, ...]
            #permute for Torch Conv2d: (N, 3, H, W)
            frame_rgb = frame_rgb.permute(0, 3, 1, 2)
            frame_depth = frame_depth.permute(0, 3, 1, 2)
            cnn_feats.append(self.cnn_rgb(frame_rgb))
            cnn_feats.append(self.cnn_depth(frame_depth))
            cnn_feat
        
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
