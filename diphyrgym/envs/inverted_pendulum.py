'''
Licence.
'''
import os 
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from diphyrgym.utils import randomize_MJCF, update_plot, unwrap_angles 
from diphyrgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from diphyrgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.robots.pendula.interted_pendulum import InvertedPendulum
from diphyrgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene

import os


class InvertedPendulumDIPhyREnv(BaseBulletEnv):
    def __init__(
        self, 
        model_xml=os.path.join(os.path.dirname(__file__), "../xmls/inverted_pendulum.xml"), 
        output_dir='/run/user/{uid}/DIPhyR/inverted_pendulum', 
        show_phase_space_diagram=False,
        save_metadata=False,
        **kwargs,
    ):
        '''
        Generate an InvertedPendulum environment with domain randomisation.
        '''

        self.timestamp = None
        self.show_phase_space_diagram = show_phase_space_diagram
        self.randomised_model_xml = None
        self.phase_space_csv = None
        self.save_metadata = save_metadata

        self.model_xml = model_xml
        self.output_dir = output_dir
        if '{uid}' in output_dir:
            # Get the current user's UID and username
            uid = os.getuid()
            tmpfs_path = self.output_dir.split('{uid}')[0] + str(uid) 
            assert os.path.exists(tmpfs_path)
            self.output_dir = output_dir.format(uid=uid) # username = os.getlogin()

        self.xml_dir = os.path.join(self.output_dir, 'xmls')
        self.phase_space_dir = os.path.join(self.output_dir, 'phase_spaces')
        
        os.makedirs(self.phase_space_dir, exist_ok=True)
        os.makedirs(self.xml_dir, exist_ok=True)

        if self.save_metadata: 
            self.json_file_path = os.path.join(self.output_dir, 'metadata.json')
            # Load or initialize the metadata JSON file
            if os.path.exists(self.json_file_path):
                try:
                    with open(self.json_file_path, 'r') as json_file:
                        self.metadata = json.load(json_file)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON file: {e}")
                    self.metadata = {}
            else:
                self.metadata = {}
        
        self.robot = InvertedPendulum(model_xml=model_xml)
        self.kwargs = kwargs
        BaseBulletEnv.__init__(self, self.robot, **kwargs)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client, timestep=0.0165, frame_skip=1):
        '''
        Create a single player scene with a stadium scene.
        '''
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=timestep, frame_skip=frame_skip)

    def save_data(self):
        '''
        Saves the phase space data to a CSV file
        '''
        data = {
          'theta': self.ps_xs['theta'], 'theta_dot': self.ps_vxs['theta'],
          'x': self.ps_xs['x'], 'x_dot': self.ps_vxs['x']
        }
        df = pd.DataFrame(data)
    
        phase_space_filename = f'ps.{self.timestamp}.csv'
        phase_space_filepath = os.path.join(self.phase_space_dir, phase_space_filename)
        df.to_csv(phase_space_filepath, index=True)
    
        # Update metadata JSON
        xml_relative_path = os.path.relpath(self.randomized_model_xml, self.xml_dir)
        phase_space_relative_path = os.path.relpath(phase_space_filepath, self.phase_space_dir)
        metadata_entry = {
            'timestamp': self.timestamp,
            'xml_filepath': xml_relative_path,
            'phase_space_filepath': phase_space_relative_path,
        }
        self.metadata[self.timestamp] = metadata_entry
    
        with open(self.json_file_path, 'w') as json_file:
            json.dump(self.metadata, json_file, indent=4)
    
    def _reset(self, **kwargs):
        '''
        Resets the environment
        '''
        if self.save_metadata and hasattr(self, 'ps_xs'):
            self.save_data()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.save_metadata else 'latest'
        self.randomized_model_xml = os.path.join(self.output_dir, f"xmls/ip.{self.timestamp}.xml") 
        randomize_MJCF(
            base_filepath=self.model_xml,
            output_filepath=self.randomized_model_xml,
            rgbdName2tagName2attrName={
              "pole": {
                "geom": ["fromto"],
                "velocity/linear": ["x", "y", "z"],
                "velocity/angular": ["x", "y", "z"],
              }, 
              "cart": {
                "joint": ["range"],
                "geom": ["size"],
                "velocity/linear": ["x", "y", "z"],
                "velocity/angular": ["x", "y", "z"],
              }, 
            },
            np_random=self.np_random,
        ) 
        self.robot = InvertedPendulum(model_xml=self.randomized_model_xml) 
        self.robot.np_random = self.np_random
        # Warning: render mode may have been set before calling this function:
        isRender = self.isRender
        BaseBulletEnv.__init__(self, self.robot, **self.kwargs)
        # We cannot restore anymore because the model is randomized
        self.isRender = isRender
        '''
        if self.stateId >= 0:
            # print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
            self._p.restoreState(self.stateId)
        '''
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        # print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
        # Phase Space Diagram GUI:
        if self.show_phase_space_diagram: 
            self.ps_xs = {'theta': [], 'x': []}
            self.ps_vxs = {'theta': [], 'x': []}
            # Setup matplotlib figure
            self.ps_x_fig, self.ps_x_ax = plt.subplots(figsize=(6, 4), dpi=100)
            self.ps_theta_fig, self.ps_theta_ax = plt.subplots(figsize=(6, 4), dpi=100)
            #self.ps_ax.set_xlim(-np.pi, np.pi)
            #self.ps_ax.set_ylim(-10, 10)
            self.ps_theta_ax.set_title('Phase Space Diagram: Cartpole')
            self.ps_theta_ax.set_xlabel('Pole Angle (radians)')
            self.ps_theta_ax.set_ylabel('Pole Angular Velocity (radians/s)')
            self.ps_theta_ax.grid(True)
            
            self.ps_x_ax.set_title('Phase Space Diagram: Cartpole')
            self.ps_x_ax.set_xlabel('Cart Position (m)')
            self.ps_x_ax.set_ylabel('Cart Linear Velocity (m/s)')
            self.ps_x_ax.grid(True)
            
        return r
    
    def _step(self, a):
        '''
        Steps the environment
        :param a: action
        '''
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
        reward = np.cos(self.robot.theta)
        done = False 
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        
        # Phase Space Diagram :
        # Extract pole angle and angular velocity after calc_state call:
        angle = self.robot.theta
        pole_angle = math.atan2(math.sin(angle), math.cos(angle))
        pole_angular_velocity = self.robot.theta_dot
        if self.show_phase_space_diagram: 
            self.ps_xs['theta'].append(pole_angle)
            self.ps_vxs['theta'].append(pole_angular_velocity)
 
        cart_pos = self.robot.x
        cart_linear_velocity = self.robot.x_dot
        if self.show_phase_space_diagram: 
            self.ps_xs['x'].append(cart_pos)
            self.ps_vxs['x'].append(cart_linear_velocity)
 
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        '''
        Adjust the camera look at point. 
        '''
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5, distance=40, yaw=0, pitch=0)

    def close(self, **kwargs):
        '''
        Close the environment.
        '''
        if self.save_metadata:
            self.save_data() 
        if self.show_phase_space_diagram:
            self.ps_img_theta = update_plot(unwrap_angles(self.ps_xs['theta']), self.ps_vxs['theta'], self.ps_theta_fig, ax=self.ps_theta_ax)#self.ps_line)
            self.ps_img_theta.save('inverted_pendulum.theta.png', format='PNG')
            self.ps_img_theta.show()

            self.ps_img_x = update_plot(self.ps_xs['x'], self.ps_vxs['x'], self.ps_x_fig, ax=self.ps_x_ax)#self.ps_line)
            self.ps_img_x.save('inverted_pendulum.x.png', format='PNG')
            self.ps_img_x.show()
        BaseBulletEnv.close(self)

