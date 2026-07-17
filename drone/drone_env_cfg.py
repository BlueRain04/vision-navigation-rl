from isaaclab.utils import configclass
import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from dataclasses import replace
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
import random
from isaaclab.envs import ViewerCfg

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    #environment
    decimation = 4 #get one frame out of 4 frames
    episode_length_s = 20.0 #length of eps
    action_space = 2 #number of actions
    history_len = 3 #number of frames to stack
    observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(history_len, 64, 64, 3), dtype=float),#obs: RGB image
        "depth": gym.spaces.Box(low=0, high=1, shape=(history_len, 64, 64, 1), dtype=float),  #obs: depth image
        "state": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(8,), dtype=float), #obs: state
    })
    state_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,)) #create the state space vector
    filter_to_obstacle = [ #create 10 obstacles for each env
    f"/World/envs/env_.*/Obstacle{i}"
    for i in range(1, 11)
    ]
    
    viewer: ViewerCfg = ViewerCfg(
        eye=(4.0, -4.0, 3.5), #camera position, offset diagonally and elevated
        lookat=(0.0, 0.0, 1.5), #looking at roughly drone-height, not ground level
        env_index=0, #which parallel environment to view (env_0)
    )

    #simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100, #every 0.01 second compute the physics
        render_interval=decimation, #draw a frame every four t
    )

    terrain: AssetBaseCfg = AssetBaseCfg( #creating a simple terrain
        prim_path="/World/ground", #asset location
        spawn=sim_utils.GroundPlaneCfg(size=(300.0, 300.0), color=(0.8, 0.8, 0.8)), #size and color
    )

    #scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg( #we run multiple envs for each drone
        num_envs=64, #number of parallel envs
        env_spacing=20, #space between the envs "we don't want overlapping!"
        replicate_physics=True, #one template for all envs for efficiency
    )

    #lighting
    sky_light = AssetBaseCfg( 
        prim_path="/World/skyLight", #light path
        spawn=sim_utils.DomeLightCfg( #apply dome light
            intensity=1000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    #obstacles
    #base configuration for one obstacle
    _base_obstacle = RigidObjectCfg( 
        prim_path="/World/envs/env_.*/Obstacle1",
        spawn=sim_utils.MultiAssetSpawnerCfg( #multi asset since it handles different objects
            assets_cfg =
            [
                sim_utils.CylinderCfg( 
                radius=0.2,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0), #heavy enough not to slide easily
                ),
            ],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4), #make the last result of the time step more accurate and close to what actually happened
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 1.0)), #the initial position of the obstacles
    )

    #define 10 distinct obstacles using 'replace' to change the prim_path
    obstacle1: RigidObjectCfg = _base_obstacle
    obstacle2: RigidObjectCfg = replace( 
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle2"
    )
    obstacle3: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle3"
    )
    obstacle4: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle4"
    )
    obstacle5: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle5"
    )
    obstacle6: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle6"
    )
    obstacle7: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle7"
    )
    obstacle8: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle8"
    )
    obstacle9: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle9"
    )
    obstacle10: RigidObjectCfg = replace(
        _base_obstacle, prim_path="/World/envs/env_.*/Obstacle10"
    )

    #robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace( #create the robot object
        prim_path="/World/envs/env_.*/Robot", #multiple robots for multiple envs
        spawn=CRAZYFLIE_CFG.spawn.replace(
            activate_contact_sensors=True, #we are enabling the sensors for collision
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, #treat this object as a rigid body
                kinematic_enabled=False, #not a kinematic but a dynamic object "kinematic doesn't allow computing forces..."
                disable_gravity=False, #important for use case
                max_linear_velocity=5.0, #linear velocity is capped at 5.0 m/s
                max_angular_velocity=10.0, #sets the roll, pitch, yaw max at 10.0 m/s 
                max_depenetration_velocity=5.0, #limits how quickly the physics engine is allowed to separate two overlapping rigid bodies
                enable_gyroscopic_forces=True, #disables simulation of the extra effect created by spinning rotors
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4, #make the last result of the time step more accurate and close to what actually happened
            solver_velocity_iteration_count=1, #make the last result of the time step more accurate and close to what actually happened
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    thrust_to_weight = 1.61 #should be > 1 so that the drone can make thrust
    moment_scale = [0.05, 0.05, 0.02] #it's the [roll, pitch, yaw]

    #sensors
    tiled_camera: TiledCameraCfg = TiledCameraCfg( #attaching one camera to each robot
        prim_path="/World/envs/env_.*/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"), #posistion of the camera relative to the body, no rotation
        data_types=["rgb", "distance_to_image_plane"], #returning color image
        spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, #contrlos the FOV (field of view)
        horizontal_aperture=20.955, #physical width of the camera "pre-defined"
        clipping_range=(0.1, 20.0), #distance threshold, anything far from 20m we can't see, closer than 10 cm not rendered as well
        ),
        width=64, #image resolution
        height=64,
    )

    contact_sensor_body: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        history_length=1,
        track_air_time=False, #don't keep track of how long the object has been in the air
        update_period=0.0, #update the sensor every simulation step
        debug_vis=False,
        filter_prim_paths_expr=filter_to_obstacle,
    )

    #rewards / logic
    target_reach_threshold = 0.8 #range from target to reach it
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01

    max_lin_vel = 2.0 #max commandable velocity per axisin m/s
    max_yaw_rate = 0.8 #max commandable yaw rate in rad/s
    
    kp_vel = 10.0 #velocity -> acceleration (P gain only, keep simple)
    max_tilt_angle = 0.6 #radians (~60 deg) safety cap
    
    kp_att = 0.008 #attitude -> torque (P and D on angle error / angular velocity)
    kd_att = 0.0015
    kp_yaw = 0.001
    kd_yaw = 0.001
    max_thrust = 0.6
    max_torque = 0.01
    max_altitude = 6.0   
    target_altitude = 1.5
    alt_penalty_scale = -0.07
    max_vert_vel = 0.8
    max_fwd_vel = 1.5
    kp_alt = 3.0
    ki_alt = 0.5
    kd_alt = 1.0
    kp_vel_pitch = 0.15
    ki_vel_pitch = 0.02
    kd_vel_pitch = 0.05
