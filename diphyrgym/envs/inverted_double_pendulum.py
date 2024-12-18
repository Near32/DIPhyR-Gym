'''
Licence.
'''
import os 

from diphyrgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from diphyrgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.robots.pendula.inverted_double_pendulum import InvertedDoublePendulum
from diphyrgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene


class InvertedDoublePendulumDIPhyREnv(BaseBulletEnv):
    def __init__(
        self, 
        model_xml=os.path.join(os.path.dirname(__file__), "../xmls/inverted_double_pendulum.xml"), 
        **kwargs,
    ):
        '''
        Generate an InvertedDoublePendulum environment with domain randomisation.
        '''
        self.robot = InvertedDoublePendulum(model_xml=model_xml)
        BaseBulletEnv.__init__(self, self.robot, **kwargs)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        '''
        Create a single player scene with a stadium scene.
        '''
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)
    
    def _reset(self, **kwargs):
        '''
        Resets the environment
        '''
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def _step(self, a):
        '''
        Steps the environment
        :param a: action
        '''
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.pos_x self.pos_y
        # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
        # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
        dist_penalty = 0.01 * self.robot.pos_x ** 2 + (self.robot.pos_y + 0.3 - 2) ** 2
        # v1, v2 = self.model.data.qvel[1:3]   TODO when this fixed https://github.com/bulletphysics/bullet3/issues/1040
        # vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        vel_penalty = 0
        alive_bonus = 10
        done = False #self.robot.pos_y + 0.3 <= 1
        self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        '''
        Adjust the camera look at point. 
        '''
        self.camera.move_and_look_at(0,1.2,1.2, 0,0,0.5)
