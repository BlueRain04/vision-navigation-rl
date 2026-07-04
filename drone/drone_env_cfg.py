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

@configclass
class DroneNavEnvCfg(DirectRLEnvCfg):
    #1 simulation setting
    decimation = 4
    episode_length_s = 60.0
    filter_to_obstacle = [
    f"/World/envs/env_.*/Obstacle{i}"
    for i in range(1, 11)
    ]

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, #start with 16 then change it later
        env_spacing=15.0,
        replicate_physics=True
    )

    #2 lighting & terrain
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHave\
            n/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(300.0, 300.0), color=(0.8, 0.8, 0.8)),
    )

    #3 obstacles
    #base configuration for one obstacle
    _base_obstacle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle1",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.CuboidCfg(
                size=(0.5, 0.5, 2.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            )],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            #heavy enough not to slide easily
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 1.0)),
    )

    #define 5 distinct obstacles using 'replace' to change the prim_path
    #define 10 distinct obstacles using 'replace' to change the prim_path
    obstacle1: RigidObjectCfg = _base_obstacle
    obstacle2: RigidObjectCfg = replace( #potential bug, declate init_state for each one or use ransomizer
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

    #4 robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    thrust_to_weight = 1.9
    moment_scale = 0.01

    #5 sensors
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/camera",#the body and camera might be wrong!
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
        ),
        width=64,
        height=64,
    )

    contact_sensor_body: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body", #the body might be wrong!
        history_length=1,
        track_air_time=False,
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=filter_to_obstacle,
    )

    #6 spaces
    num_actions = 4
    history_len = 3 #number of frames to stack
    
    #3 channels (RGB) + 3 channels (goal vector X, Y, dist expanded) = 6
    num_channels = 7
    
    observation_space = gym.spaces.Box(
        low=0, high=255, 
        shape=(history_len, 64, 64, num_channels),
        dtype=float
    )
    
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
    state_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,))

    #7 rewards / logic
    action_scale = 5.0
    target_reach_threshold = 0.4