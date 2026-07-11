from __future__ import annotations
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnv
from .drone_env_cfg import QuadcopterEnvCfg
from isaaclab.assets import Articulation, RigidObject
from dataclasses import replace
from isaaclab.sensors import TiledCamera, ContactSensor
import isaaclab.utils.math as math_utils
import math

def contact_penalty(env,
    contact_sensor_name: str,
    threshold: float = 1.0, 
    ) -> torch.Tensor:
    """
    Penalize collisions with obstacles using filtered contact sensors.
    Uses force_matrix_w: (E, B, F, 3). Returns (E,) in [-1, 1].
    """
    total_penalty = torch.zeros(env.num_envs, device=env.device) 
    sensor = env.scene.sensors.get(contact_sensor_name, None) #get the body sensor from the scene
    if sensor is None:
        return total_penalty
    fm = getattr(sensor.data, "force_matrix_w", None) #get the force matrix attribute
    if isinstance(fm, torch.Tensor) and fm.numel() > 0:
        strength = torch.norm(fm, dim=-1).amax(dim=(1, 2)) #for every env find the force
        excess = torch.clamp(strength - threshold, min = 0.0) #if the force is higher than a threshold then add to excess
        total_penalty += excess
    return torch.tanh(total_penalty) #mapping to [-1, 1]

def terminate_on_contact(env,
    contact_sensor_name: str,
    threshold: float = 1.0,
    ) -> torch.Tensor:
    """Terminate ONLY if filtered obstacle contacts exceed threshold."""
    term = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sensor = env.scene.sensors.get(contact_sensor_name, None) #get the body sensor from the scene
    if sensor is None:
        return term
    fm = getattr(sensor.data, "force_matrix_w", None)
    if isinstance(fm, torch.Tensor) and fm.numel() > 0:
        strength = torch.norm(fm, dim=-1).amax(dim=(1, 2))
        term |= strength > threshold
    return term

def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                ),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

def define_goal_marker():
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal",
        markers= {
            "sphere": sim_utils.SphereCfg(
                radius = 1.0,
                visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg)

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs): #need to add the contact sensor
        super().__init__(cfg, render_mode, **kwargs)
        self._rgb_hist: torch.Tensor | None = None #start the sps with None camera hist, later could be a tensor
        self._depth_hist: torch.Tensor | None = None #start the sps with None camera hist, later could be a tensor
        self.history_len = cfg.history_len
        self.num_obstacles = 5
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_dist = torch.zeros((self.num_envs,), device=self.device)
       # self.arrows = define_markers()
       # self.arrows.set_visibility(True)
        self.goal_marker = define_goal_marker()
        self.goal_marker.set_visibility(True)
        self._body_id = self._robot.find_bodies("body")[0] #this is linked to the camera, make sure that the crazyflie has body index
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum() #get the robot mass
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm() #get the gravity
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item() #get the robit weight
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device) #create the thrust tensor
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device) #create the moment tensor
        self._moment_scale = torch.as_tensor(self.cfg.moment_scale, dtype=torch.float32, device=self.device) #get the moment scale from cfg and make it a tensor
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device) #check
        self._forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._prev_yaw = torch.zeros(self.num_envs, device=self.device)
        self._was_near_obstacle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_counter = 0
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "collision_reward",
                "dist_delta",
                "progress_reward",
                "success_reward",
               # "alignment_reward",
           #     "backward_penalty",
                "ang_vel",
                "heading_error_penalty",
                "yaw_change_reward",
              #  "avoid_success_reward"
            ]
        }

    def _setup_scene(self): #need to add the contact sensor
        self._robot = Articulation(self.cfg.robot) #get the robot from the cfg
        self.scene.articulations["robot"] = self._robot #add the robot to the scene
        self.cfg.terrain.spawn.func(self.cfg.terrain.prim_path, self.cfg.terrain.spawn) #create the terrain in the env
        if hasattr(self.cfg, "sky_light") and self.cfg.sky_light is not None: #add the lightning
            self.cfg.sky_light.spawn.func(
                self.cfg.sky_light.prim_path, self.cfg.sky_light.spawn
            )
            
        #add the obstacles
        obs_configs = [
            self.cfg.obstacle1,
            self.cfg.obstacle2,
            self.cfg.obstacle3,
            self.cfg.obstacle4,
            self.cfg.obstacle5,
        ]
        for i, obs_cfg in enumerate(obs_configs, 1):
            obs_cfg.spawn.func(
                f"/World/envs/env_0/Obstacle{i}",
                obs_cfg.spawn,
                translation=obs_cfg.init_state.pos,
                orientation=obs_cfg.init_state.rot,
            )
        combined_obs_cfg = replace(
            self.cfg.obstacle1, prim_path="/World/envs/env_.*/Obstacle.*"
        )
        self.obstacle = RigidObject(combined_obs_cfg)
        self.scene.rigid_objects["obstacle"] = self.obstacle #add the obs in the scene
        self.robot_camera = TiledCamera(self.cfg.tiled_camera) #get the camera's data from cfg
        self.contact_body = ContactSensor(self.cfg.contact_sensor_body)
        self.scene.sensors["tiled_camera"] = self.robot_camera #add the camera in the scene
        self.scene.sensors["contact_sensor_body"] = self.contact_body
        self.scene.clone_environments(copy_from_source=True) #some of the envs in the eps inherit from each other for fast training
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def _get_goal_vec(self):
        return self.target_pos - self._robot.data.root_pos_w
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.step_counter += 1
        self._actions = actions.clone().clamp(-1.0, 1.0) #clone the action for independent memory then clip it
      #  self._actions = 0.8 * self._actions + 0.2 * actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0 #get the thrust action range [-1, 1] and map it to [0, 1] to apply in simulator, assign it as Z force since X Y are not applicable
        self._moment[:, 0, :] = self._moment_scale * self._actions[:, 1:] #take the roll, pitch, and yas from action scale them then add to moment tensor
        self.goal_marker.visualize(self.target_pos)
        if self.common_step_counter % 500 == 0:
            print(f"Action mean : {self.actions.mean():.3f}")
            print(f"Action std  : {self.actions.std():.3f}")
            print(
                f"Thrust act  : mean={self.actions[:,0].mean():.3f}, "
                f"std={self.actions[:,0].std():.3f}, "
                f"min={self.actions[:,0].min():.3f}, "
                f"max={self.actions[:,0].max():.3f}"
            )
            print(f"Roll act    : {self.actions[:,1].mean():.3f}")
            print(f"Pitch act   : {self.actions[:,2].mean():.3f}")
            print(f"Yaw act     : {self.actions[:,3].mean():.3f}")
            print(f"Thrust force : {self._thrust[:, 0, 2].mean():.3f}")
      #  self._visualize_arrows()

    def _visualize_arrows(self):
        goal_vec = self._get_goal_vec()
        command_yaws = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
        command_rot = math_utils.quat_from_angle_axis(command_yaws, self.up_dir)
        robot_rot = self._robot.data.root_quat_w

        loc = self._robot.data.root_pos_w.clone()
        loc[:, 2] += 0.5
        locs = torch.cat((loc, loc), dim=0)
        rots = torch.cat((robot_rot, command_rot), dim=0)

        num = self.num_envs
        indices = torch.cat(
            (
                torch.zeros(num, dtype=torch.int32, device=self.device),
                torch.ones(num, dtype=torch.int32, device=self.device),
            ),
            dim=0,
        )

       # self.arrows.visualize(
          #  translations=locs, orientations=rots, marker_indices=indices
       # )
        
    def _apply_action(self): #apply the action into the env
        self._robot.set_external_force_and_torque(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        self.robot_camera.update(dt=self.cfg.sim.dt * self.cfg.decimation) #get the camera frame

        #update contact sensor
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.contact_body.update(dt=dt)

        camera_data_rgb = self.robot_camera.data.output["rgb"].float() / 255.0
        camera_data_depth = self.robot_camera.data.output["distance_to_image_plane"].float()
        camera_data_depth = torch.clamp(camera_data_depth, 0.0, 20.0)
        camera_data_depth = camera_data_depth / 20.0
        #we need to change this and add the depth
        if self._rgb_hist is None: #if this is the first frame
            self._rgb_hist = (
                camera_data_rgb.unsqueeze(1)
                .repeat(1, self.history_len, 1, 1, 1) #wait to add the other two frames
                .contiguous()
            )
        else:
            new_frame = camera_data_rgb.unsqueeze(1)
            self._rgb_hist = torch.cat(
                [self._rgb_hist[:, 1:], new_frame], dim=1
            )
        if self._depth_hist is None: #if this is the first frame
            self._depth_hist = (
                camera_data_depth.unsqueeze(1)
                .repeat(1, self.history_len, 1, 1, 1) #wait to add the other two frames
                .contiguous()
            )
        else:
            new_frame = camera_data_depth.unsqueeze(1)
            self._depth_hist = torch.cat(
                [self._depth_hist[:, 1:], new_frame], dim=1
            )

        N, T, H, W, _ = self._rgb_hist.shape

        goal_vec = self._get_goal_vec()
        goal_dist = torch.linalg.norm(goal_vec, dim=-1, keepdim=True)
        unit_goal = goal_vec / (goal_dist + 1e-6)

        # extract yaw from quaternion
        _, _, yaw = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        yaw = yaw.unsqueeze(-1)
        
        #get the velocity
        lin_vel = self._robot.data.root_lin_vel_b

        state_input = torch.hstack((unit_goal, goal_dist, yaw, lin_vel))
        
        return {"policy": {
            "rgb": self._rgb_hist,
            "depth": self._depth_hist,
            "state": state_input,
        }} #what the model recives
    
    def _get_rewards(self) -> torch.Tensor:
        #1 velocity
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        ang_vel = torch.clamp(ang_vel, max=10.0)
        #obstacle detection from depth
        min_depth = self._depth_hist[:, -1].amin(dim=(1, 2, 3))
        obstacle_detected = min_depth < 0.15 #we might need to tune the 0.15

        #current yaw
        _, _, yaw = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        
        #2 close to goal
        robot_lin_vel = self._robot.data.root_lin_vel_w[:, :3]
        goal_vec = self._get_goal_vec()
        goal_dir = goal_vec / (torch.norm(goal_vec, dim=-1, keepdim=True) + 1e-6)
        velocity_proj = torch.sum(robot_lin_vel * goal_dir, dim=-1)
        progress_reward = velocity_proj * 2.5

        #3 distance reward
        dist = torch.linalg.norm(goal_vec, dim=-1)
        dist_delta = self.prev_dist - dist
        self.prev_dist = dist.clone()

        #4 collision penalty
        collision_val = contact_penalty(self, "contact_sensor_body", threshold=0.1)

        #5 success
        reached = dist < self.cfg.target_reach_threshold
        success_reward = reached.float() * 100.0
        
        #5 heading error, only penalized when clear (rule 1)
        forwards = math_utils.quat_apply(self._robot.data.root_quat_w, self._forward_vec_b)
        heading_alignment = torch.sum(forwards * goal_dir, dim=-1)  # 1 = facing goal, -1 = facing away
        heading_error_penalty = torch.where(
            obstacle_detected,
            torch.zeros_like(heading_alignment),
            (1.0 - heading_alignment),  # penalize misalignment only when clear
        )

        #6 yaw-change reward, only when obstacle detected (rule 2)
        yaw_diff = torch.abs(yaw - self._prev_yaw)
        yaw_diff = torch.remainder(yaw_diff + math.pi, 2 * math.pi) - math.pi  #wrap to [-pi, pi]
        yaw_change_reward = torch.where(
            obstacle_detected,
            torch.clamp(torch.abs(yaw_diff), max=0.3) * 0.3,
            torch.zeros_like(yaw_diff),
        )
        self._prev_yaw = yaw.clone()

        #7 avoid-success bonus (Rule 3)
        # fires when drone WAS near an obstacle last step, and is no longer, and didn't crash
      #  just_cleared = self._was_near_obstacle & (~obstacle_detected) & (~collision_val.bool())
      # avoid_success_reward = just_cleared.float() * 20.0
      #  self._was_near_obstacle = obstacle_detected.clone()
        #6 alignment reward
      #  forwards = math_utils.quat_apply(self._robot.data.root_quat_w, self._forward_vec_b)
      #  goal_dir = goal_vec / (torch.norm(goal_vec, dim=-1, keepdim=True) + 1e-6)
       # alignment = torch.sum(forwards * goal_dir, dim=-1)
      #  alignment_reward = alignment * 0.5

        #7 Backward penalty
       # backward_act = torch.sum(torch.clamp(self.actions, max=0.0), dim=1)
       # backward_penalty = backward_act * 0.5

        rewards = {
            "ang_vel": ang_vel * -0.005,
            "collision_reward": collision_val * -12.0,
            "dist_delta": dist_delta * 0.7,
            "progress_reward": progress_reward,
            "success_reward": success_reward,
            #"alignment_reward": alignment_reward,
            "heading_error_penalty": -heading_error_penalty * 0.03,
            "yaw_change_reward": yaw_change_reward * 0.05,
          #  "avoid_success_reward": avoid_success_reward * 0.3,
         #   "backward_penalty": backward_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1) #finish the eps when time ends

        goal_vec = self._get_goal_vec()
        dist = torch.linalg.norm(goal_vec, dim=-1)
        reached = dist < self.cfg.target_reach_threshold #when robot reached the goal by threshold

        crashed = terminate_on_contact(self, "contact_sensor_body", threshold=0.1) #when crashed with ods

        return (reached | crashed), time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None): #not all envs reset on the same time
        if env_ids is None: #if all envs are done
            env_ids = self._robot._ALL_INDICES #selecting all envs to reset
        num_resets = len(env_ids) #check

        print(
            f"""
        Episode:
        Progress : {self._episode_sums['progress_reward'][env_ids].mean():8.2f}
        Distance : {self._episode_sums['dist_delta'][env_ids].mean():8.2f}
        Collision: {self._episode_sums['collision_reward'][env_ids].mean():8.2f}
        Heading  : {self._episode_sums['heading_error_penalty'][env_ids].mean():8.2f}
        Yaw      : {self._episode_sums['yaw_change_reward'][env_ids].mean():8.2f}
        Ang Vel  : {self._episode_sums['ang_vel'][env_ids].mean():8.2f}
        success_reward: {self._episode_sums['success_reward'][env_ids].mean():8.2f}
        """
        )
        #reset robot
        root_state = self._robot.data.default_root_state[env_ids].clone() #get a copy from the robot template
        root_state[:, :3] += self.scene.env_origins[env_ids] #get the robot (X, Y, Z) position

        #random yaw "check"
        rand_yaw = torch.zeros((num_resets, 3), device=self.device)
        rand_yaw[:, 2] = torch.rand(num_resets, device=self.device) * 2.0 * math.pi

        _, _, yaw0 = math_utils.euler_xyz_from_quat(root_state[:, 3:7])
        self._prev_yaw[env_ids] = yaw0

        self._was_near_obstacle[env_ids] = False

        quat = math_utils.quat_from_euler_xyz(
            rand_yaw[:, 0], rand_yaw[:, 1], rand_yaw[:, 2]
        )
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0

        self._robot.write_root_state_to_sim(root_state, env_ids) #check

        if self._rgb_hist is not None: #reset camera
            self._rgb_hist[env_ids] = 0.0
            
        if self._depth_hist is not None: #reset camera
            self._depth_hist[env_ids] = 0.0

        #reset goal
        goal_radii = torch.empty(num_resets, device=self.device).uniform_(3.5, 5.0) #sample each goal in envs a distance from the origin
        goal_thetas = torch.empty(num_resets, device=self.device).uniform_( #direction of the goal
            -math.pi, math.pi
        )

        self.target_pos[env_ids, 0] = ( #goal X position
            self.scene.env_origins[env_ids, 0]
            + goal_radii * torch.cos(goal_thetas)
        )
        self.target_pos[env_ids, 1] = ( #goal Y position
            self.scene.env_origins[env_ids, 1]
            + goal_radii * torch.sin(goal_thetas)
        )
        self.target_pos[env_ids, 2] = 1.2
       # goal_z = torch.empty(num_resets, device=self.device).uniform_(1.0, 3.0) #might increase the Z to make it harder (also compared to obs)
       #  self.target_pos[env_ids, 2] = ( #goal Z position
        #     self.scene.env_origins[env_ids, 2]
        #     + goal_z
       #  )

        #reset prev distances for delta-distance reward
        goal_vec_init = self.target_pos[env_ids] - self._robot.data.root_pos_w[env_ids]
        self.prev_dist[env_ids] = torch.linalg.norm(goal_vec_init, dim=-1)

        #reset obstacles with spacing constraints
        obs_base_ids = env_ids * self.num_obstacles
        all_obs_ids = torch.cat(
            [obs_base_ids + i for i in range(self.num_obstacles)], dim=0
        ) #mapping each env with it's obs
        total_obs = len(all_obs_ids)

        positions_env: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        env_ids_list = env_ids.tolist()

        for env in env_ids_list:
            for key in self._episode_sums.keys():
                self._episode_sums[key][env_ids] = 0.0
            origin = self.scene.env_origins[env]
            goal = self.target_pos[env]
            positions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

            for _ in range(self.num_obstacles):
                while True: #colud be improved with victorization!
                    radius = (
                        torch.rand((), device=self.device) * (3.0 - 1.5) + 1.5
                    )
                    theta = (
                        torch.rand((), device=self.device) * (2 * math.pi) - math.pi
                    )
                    x = origin[0] + radius * torch.cos(theta)
                    y = origin[1] + radius * torch.sin(theta)
                    sample_z = torch.empty((), device=self.device).uniform_(1.0, 3.0)
                    z = origin[2] + sample_z

                    #obstacle-origin spacing
                    if torch.sqrt( #obs must not be close to the origin "< 1.5"
                        (x - origin[0]) ** 2 + (y - origin[1]) ** 2 + (z - origin[2]) ** 2
                    ) < 1.5:
                        continue
                    #obstacle-goal spacing
                    if torch.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2 + (z - goal[2]) ** 2) < 0.8: #must not block the goal
                        continue

                    too_close = False #must not overlap woth other obs
                    for (ox, oy, oz) in positions:
                        if torch.sqrt((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2) < 0.8:
                            too_close = True
                            break
                    if too_close:
                        continue

                    positions.append((x, y, z))
                    break

            positions_env[env] = positions

        #flatten positions for all obstacles
        obs_x_list = []
        obs_y_list = []
        obs_z_list = []
        for i in range(self.num_obstacles):
            for env in env_ids_list:
                x, y, z = positions_env[env][i]
                obs_x_list.append(x)
                obs_y_list.append(y)
                obs_z_list.append(z)

        obs_x = torch.stack(obs_x_list)
        obs_y = torch.stack(obs_y_list)
        obs_z = torch.stack(obs_z_list)

        obs_state = self.obstacle.data.default_root_state[all_obs_ids].clone()
        obs_state[:, 0] = obs_x
        obs_state[:, 1] = obs_y
        obs_state[:, 2] = obs_z

        self.obstacle.write_root_state_to_sim(obs_state, all_obs_ids)
        self.episode_length_buf[env_ids] = 0
