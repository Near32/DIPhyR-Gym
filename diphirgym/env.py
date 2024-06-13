import pybulletgym as pbg
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
from diphirgym.rigid_bodies_robot import RigidBodiesRobot


class DIPhiREnv(BaseBulletEnv):
  def __init__(self, **kwargs):
    self.robot_name = 'rigid-bodies'
    #TODO: observation space is usually derive form the robot class...
    # Need to redefine it below for dict containing images and text.
    self.obs_dim = 1
    #TODO: similarly, the action space is defined in the robot calss..
    # Currently, this is fine, but eventually it will be necessary to
    # propose a pusher-like end-effector, without all the low-level elements?
    self.action_dim = 1

    self.model_urdf_string = '' #TODO
    self.rigid_bodies = RigidBodiesRobot(
      model_urdf_string=self.model_urdf_string, #TODO
      robot_name=self.robot_name,
      action_dim=self.action_dim,
      obs_dim=self.obs_dim,
      basePosition=None,
      baseOrientation=None,
      fixed_based=True,
      self_collision=True,
    )
  
    BaseBulletEnv.__init__(
      self,
      robot=self.rigid_bodies,
    )
    
  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.0020, frame_skip=5)

    
