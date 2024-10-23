'''
Licence.
'''
from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot

class RigidBodiesRobot(URDFBasedRobot):
    def __init__(self, model_urdf_string, robot_name, action_dim, obs_dim, basePosition=None, baseOrientation=None, fixed_base=False, self_collision=False):
        '''
        Generate a RigidBodiesRobot.
        '''
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)

        self.model_urdf = ''
        self.model_urdf_string = model_urdf_string
        self.basePosition = basePosition if basePosition is not None else [0, 0, 0]
        self.baseOrientation = baseOrientation if baseOrientation is not None else [0, 0, 0, 1]
        self.fixed_base = fixed_base

    def reset(self, bullet_client):
        '''
        reset
        '''
        self._p = bullet_client
        self.ordered_joints = []

        if self.self_collision:
            #TODO: need to figure out what kind of SetJointMotorControl2 to use in the addToScene initialisation:
            # The current implementation from XmlBasedRobot class is pybullt.POSITION_CONTROL, which might not 
            # be adequate for our use case.
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                self._p.loadURDFString(self.model_urdf_string,
                basePosition=self.basePosition,
                baseOrientation=self.baseOrientation,
                useFixedBase=self.fixed_base,
                flags=pybullet.URDF_USE_SELF_COLLISION),
            )
        else:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                self._p,
                self._p.loadURDFString(self.model_urdf_string,
                    basePosition=self.basePosition,
                    baseOrientation=self.baseOrientation,
                    useFixedBase=self.fixed_base,
                ),
            )

        self.robot_specific_reset(self._p)

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()

        return s

    def robot_specific_reset(self, physicsClient):
        '''
        Apply a random impulse to each rigid body.
        '''
        self._p = bullet_client
        rbids = [part.bodyIndex for part_name, part in self.parts.items()]
        for rbid in rbids:
            force_vector = self.np_random.uniform(-0.01, 0.01, 3) #TODO: figure out if the norm is sufficient
            self._p.applyExternalForce(
                rbid,
                -1,
                force_vector,
                [0,0,0],
                self._p.WORLD_FRAME,
            )

    def apply_action(self, a):
        '''
        TODO: replace below with an actual pusher-like end effector with impulses
        TODO: need to figure out how to apply impulses on specific objects 
        '''
        assert (np.isfinite(a).all())
        pass

    def calc_state(self):
        '''
        TODO: build the image (or rather in the env method calling this) and the textual simulation trace
        '''
        pass
 
