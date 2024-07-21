import os 
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

from diphirgym.utils import randomize_MJCF 
from diphirgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from diphirgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.robots.pendula.interted_pendulum import InvertedPendulum
from diphirgym.thirdparties.pybulletgym.pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene


# Function to update the plot
def update_plot(angles, velocities, fig, ax=None, line=None):
    if line is None:
        assert ax is not None
        line, = ax.plot(angles, velocities, 'g-')
    else:
        line.set_data(angles, velocities)
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return img

def unwrap_angles(angles):
    return np.unwrap(angles)

class InvertedPendulumDIPhiREnv(BaseBulletEnv):
    def __init__(
        self, 
        model_xml=os.path.join(os.path.dirname(__file__), "../xmls/inverted_pendulum.xml"), 
        output_dir='/tmp/DIPhiR/inverted_pendulum', 
        show_phase_space_diagram=False,
        **kwargs,
    ):
        self.timestamp = None
        self.show_phase_space_diagram = show_phase_space_diagram
        self.randomised_model_xml = None
        self.phase_space_csv = None
        
        self.model_xml = model_xml
        self.output_dir = output_dir
        self.xml_dir = os.path.join(self.output_dir, 'xmls')
        self.phase_space_dir = os.path.join(self.output_dir, 'phase_spaces')
        
        os.makedirs(self.phase_space_dir, exist_ok=True)
        os.makedirs(self.xml_dir, exist_ok=True)

        
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

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)
    
    def save_data(self):
        data = {'theta': self.ps_xs, 'theta_dot': self.ps_vxs}
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
        if hasattr(self, 'ps_xs'):
            self.save_data()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        #BaseBulletEnv.__init__(self, self.robot, **self.kwargs)
        # We cannot restore anymore because the model is randomized
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
        self.ps_xs = []
        self.ps_vxs = [] 
        # Setup matplotlib figure
        self.ps_fig, self.ps_ax = plt.subplots(figsize=(6, 4), dpi=100)
        #self.ps_ax.set_xlim(-np.pi, np.pi)
        #self.ps_ax.set_ylim(-10, 10)
        self.ps_ax.set_title('Phase Space Diagram: Cartpole')
        self.ps_ax.set_xlabel('Pole Angle (radians)')
        self.ps_ax.set_ylabel('Pole Angular Velocity (radians/s)')
        self.ps_ax.grid(True)
        
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
        reward = np.cos(self.robot.theta)
        done = False 
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        
        # Phase Space Diagram :
        # Extract pole angle and angular velocity after calc_state call:
        angle = self.robot.theta
        pole_angle = math.atan2(math.sin(angle), math.cos(angle))
        pole_angular_velocity = self.robot.theta_dot
        self.ps_xs.append(pole_angle)
        self.ps_vxs.append(pole_angular_velocity)
 
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5, distance=40, yaw=0, pitch=0)

    def close(self, **kwargs):
        self.save_data() 
        self.ps_img = update_plot(unwrap_angles(self.ps_xs), self.ps_vxs, self.ps_fig, ax=self.ps_ax)#self.ps_line)
        self.ps_img.save('inverted_pendulum.png', format='PNG')
        if self.show_phase_space_diagram or self.isRender:
            self.ps_img.show()
        BaseBulletEnv.close(self)

