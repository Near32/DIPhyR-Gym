import os 
import numpy as np

from diphirgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from diphirgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.robots.pendula.interted_pendulum import InvertedPendulum
from diphirgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene


class InvertedPendulumDIPhiREnv(BaseBulletEnv):
    def __init__(
        self, 
        model_xml=os.path.join(os.path.dirname(__file__), "../xmls/inverted_pendulum.xml"), 
        **kwargs,
    ):
        self.robot = InvertedPendulum(model_xml=model_xml)
        BaseBulletEnv.__init__(self, self.robot, **kwargs)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def _reset(self, **kwargs):
        if self.stateId >= 0:
            # print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        # print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
        return r

    def _step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.pos_x self.pos_y
        vel_penalty = 0
        '''
        if self.robot.swingup:
            reward = np.cos(self.robot.theta)
            done = False
        else:
            reward = 1.0
            done = np.abs(self.robot.theta) > .2
        '''
        reward = np.cos(self.robot.theat)
        done = False 
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)


