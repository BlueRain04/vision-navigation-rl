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

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    #environment
    decimation = 4
    episode_length_s = 60.0
    action_space = 4
    history_len = 3 #number of frames to stack
    observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(history_len, 64, 64, 3), dtype=float),
        "depth": gym.spaces.Box(low=0, high=255, shape=(history_len, 64, 64, 1), dtype=float), #check the high value
        "state": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(4,), dtype=float),
    })
    state_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,)) 
    filter_to_obstacle = [ #create 10 obstacles for each env
    f"/World/envs/env_.*/Obstacle{i}"
    for i in range(1, 11)
    ]

    #simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100, #every 0.01 second compute the physics
        render_interval=decimation, #draw a frame every two t
    )

    terrain: AssetBaseCfg = AssetBaseCfg( #creating a simple terrain
        prim_path="/World/ground", #asset location
        spawn=sim_utils.GroundPlaneCfg(size=(300.0, 300.0), color=(0.8, 0.8, 0.8)), #size and color
    )

    #scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg( #we run multiple envs for each drone
        num_envs=16, #start with 16 then change it later to 3600
        env_spacing=20, #space between the envs "we don't want overlapping!"
        replicate_physics=True, #one template for all envs for efficiency"
    )

    #lighting
    sky_light = AssetBaseCfg( 
        prim_path="/World/skyLight",
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
            assets_cfg = #different shapes of obstacles
            [
                sim_utils.CuboidCfg( #represents (wall, pillar, building, box)
                size=(0.5, 0.5, 2.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                ),
                sim_utils.CylinderCfg( #represents (tree, pole, column)
                radius=0.35,
                height=2.2,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                ),
                sim_utils.ConeCfg( #represents (traffic)
                radius=0.5,
                height=2.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                ),
            ],
            random_choice=True, #choose the object randomlly
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            #heavy enough not to slide easily
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
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    thrust_to_weight = 1.61 #should be > 1 so that the drone can make thrust
    moment_scale = [1.25, 1.92, 0.154] #it's the [roll, pitch, yaw]

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
        track_air_time=False,
        update_period=0.0,
        debug_vis=False,
        filter_prim_paths_expr=filter_to_obstacle,
    )

    #rewards / logic
    target_reach_threshold = 0.4
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
