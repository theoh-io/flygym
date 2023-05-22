import gymnasium as gym
from gymnasium import spaces
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#from scipy.spatial import ConvexHull, Point
from flygym.util.data import default_pose_path


class MyNMF(gym.Env):
    def __init__(self, reward="default", verbose=0,**kwargs):
        self.nmf = NeuroMechFlyMuJoCo(**kwargs)
        num_dofs = len(self.nmf.actuated_joints)
        bound = 0.5
        self.action_space = spaces.Box(low=-bound, high=bound,
                                       shape=(num_dofs,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(num_dofs,))
        self.joint_pos= np.zeros(num_dofs)
        self.joint_vel= np.zeros(num_dofs)
        self.joint_torques= np.zeros(num_dofs)
        self.fly_pos = np.zeros(3)
        self.fly_vel = np.zeros(3)
        self.fly_ori = np.zeros(3)
        self.contact_forces=np.zeros(6*3)
        self.end_effectors=np.zeros(6*3)

        self.reward_function=reward
        self.reward_total={"rew": 0}
        self.reward_terms={"rew_vel": 0}
        
        self.verbose=verbose

        self.ori_ref=[-1.5, 0, 0.88]

        self.coeff_vel=1
        self.coeff_flipped=100
        self.coeff_z=-0.01
        self.coeff_energy=-1e05
        self.coeff_yaw=-2
        self.coeff_roll=-2
        #self.coeff_drift=-0.01


    
    def _parse_obs(self, raw_obs):
        features = [
            #joint angle
            raw_obs['joints'][0, :].flatten(),
            #joint velocity
            #raw_obs['joints'][1, :].flatten(),
            #joint torque
            #raw_obs['joints'][1, :].flatten(),
            # raw_obs['fly'].flatten(),
            # what else would you like to include?
        ]
        #print(raw_obs['joints'].shape)
        #print(np.array(features).shape)
        return np.concatenate(features, dtype=np.float32)
    
    def reset(self):
        #self.reward_terms={"rew_total": 0}
        raw_obs, info = self.nmf.reset()
        return self._parse_obs(raw_obs), info
    
    def upside_down(self):
        """ Check if the robot is upside down."""
        
        margin= 0.3
        if np.any(self.fly_ori-self.ori_ref < -margin) or np.any(self.fly_ori-self.ori_ref > margin):
            if self.verbose:
                print("flipped")
            return True
        else:
            return False
    
    def reward_energy(self):
        """ Reward function aiming at maximizing forward velocity."""
        # reward for forward velocity

        vel_reward = -self.fly_vel[0]/1000
        z_penalty = (self.fly_vel[2]/1000)**2
        if self.upside_down():
            flipped_reward=-1
        else:
            flipped_reward=0    
        
        # minimize energy 
        energy_reward = 0 
        
        for tau,vel in zip(self.joint_torques,self.joint_vel):
             energy_reward += np.abs(np.dot(tau,vel)) * self.nmf.timestep

        # def compute_mechanical_work(self, joint_velocities, joint_torques):
        # """ Computes the mechanical work spent by the animal. """
        # return np.abs(
        #     joint_torques@joint_velocities.T
        # ) * self.time_step / self.run_time

        reward = self.coeff_vel*vel_reward \
                 + self.coeff_flipped*flipped_reward \
                 + self.coeff_z* z_penalty \
                 + self.coeff_energy * energy_reward
        
        self.reward_total={"rew": reward}
        self.reward_terms={ "vel_rew": self.coeff_vel*vel_reward, \
                            "z_penalty": self.coeff_z*z_penalty, \
                            "energy": self.coeff_energy*energy_reward}
        
        if self.verbose:
            print(f"joint vel: {self.joint_vel}")
            print(f"torques: {self.joint_torques}")
            print(f"reward_ tot: {reward}")
            print(f"vel:{self.coeff_vel*vel_reward}, z:{self.coeff_z*z_penalty}, energy: {self.coeff_energy*energy_reward}")
        

        return reward # keep rewards positive
        
    
    def reward_straight(self):
        """ Reward function aiming at maximizing forward velocity while penalizing lateral drift."""
        

        vel_reward = -self.fly_vel[0]/1000
        z_penalty = (self.fly_vel[2]/1000)**2
        
        if self.upside_down():
            flipped_reward=-1
        else:
            flipped_reward=0    
        
        #minmize roll (not fall on the side)
        roll_reward = np.abs(self.fly_ori[0]-self.ori_ref[0])

        # minimize yaw (go straight) 
        yaw_reward = np.abs(self.fly_ori[2]-self.ori_ref[2])

         
        
        # penalty body y - don't drift laterally
        #drift_reward = abs(self.robot.GetBasePosition()[1]) #0.01


        reward = self.coeff_vel*vel_reward \
                 + self.coeff_flipped*flipped_reward \
                 + self.coeff_z*z_penalty \
                 + self.coeff_yaw * yaw_reward \
                 + self.coeff_roll * roll_reward\
                 #+ self.coeff_drift * drift_reward \

        if self.verbose:
            print(f"total rew {reward}")
            print(f"orientation {yaw_reward}, {roll_reward}")
       
        self.reward_total={"rew": reward}
        self.reward_terms={ "vel_rew": self.coeff_vel*vel_reward, \
                            "z_penalty": self.coeff_z*z_penalty, \
                            "yaw": self.coeff_yaw * yaw_reward, \
                            "roll": self.coeff_roll * roll_reward}#, \
                            #"drift": self.coeff_drift * drift_reward}
        return reward # keep rewards positive
    
    def support_polygon(self, contact_points):
        # Assuming 'end_effectors' is a numpy array with shape (N, 2) representing 2D positions of end effectors
        #contact_points = self.end_effectors  # Set the contact points as the end effectors positions
        # Compute the convex hull of the contact points
        hull = ConvexHull(contact_points)
        # Get the vertices of the support polygon
        support_polygon_vertices = contact_points[hull.vertices]
        return support_polygon_vertices
    
    def is_inside_polygon(self, CoP, poly_vertices):
        # Assuming 'cop' is a numpy array representing the 2D position of the Center of Pressure
        cop_point = Point(CoP[0], CoP[1])
        # Create a polygon object from the support polygon vertices
        support_polygon = Polygon(poly_vertices)
        # Check if the CoP point is inside the support polygon
        is_inside = support_polygon.contains(cop_point)
        return is_inside

    def reward_COP(self):
        """ Reward function aiming at penalizing CoP outside of support polygon."""
        

        vel_reward = -self.fly_vel[0]/1000
        z_penalty = (self.fly_vel[2]/1000)**2
        
        if self.upside_down():
            flipped_reward=-1
        else:
            flipped_reward=0    
        
        
        # Compute the center of pressure
        total_force = np.sum(self.contact_forces, axis=0)
        if self.verbose:
            print(f"contact_forces {self.contact_forces.shape}")
            print(f"end eff pos {self.end_effectors.shape}")
            print(f"sum {np.sum(self.contact_forces*self.end_effectors, axis=0).shape}")
            print(f"total force {np.linalg.norm(total_force).shape}")
            

        if np.linalg.norm(total_force) > 0:
            cop = np.sum(self.contact_forces*self.end_effectors, axis=0) / np.linalg.norm(total_force)
        else:
            cop = np.zeros(3)
        
        polygon_vertices=self.support_polygon(self.end_effectors)
        CoP_inside=self.is_inside_polygon(cop, polygon_vertices)
       
        if CoP_inside:
            cop_reward=1
        else:
            cop_reward=-20 
        # Compute the convex hull of the support polygon
        # support_points = self.sim.model.site_pos[['left_foot', 'right_foot', 'front_left_foot', 
        #                                            'front_right_foot', 'back_left_foot', 'back_right_foot']]
        # support_hull = ConvexHull(support_points[:, :2])
        
        # # Compute the center of mass
        # com = self.sim.data.subtree_com[self.sim.model.body_name2id('torso')]
        
        # # Check if the CoM is inside the support polygon
        # if not support_hull.contains(Point(com[:2])):
        #     penalty = -0.1
        #     reward += penalty

        reward = self.coeff_vel*vel_reward \
                 + self.coeff_flipped*flipped_reward \
                 +self.coeff_z*z_penalty \
                 +self.coeff_CoP*cop_reward
        
        if self.verbose:
            print(f"contact_forces {self.contact_forces}")
            print(f"end eff pos {self.end_effectors}")
            print(f"total force {np.linalg.norm(total_force)}")
            print(f"CoP {cop}")
            print(f"CoP inside: {CoP_inside}")
        
        self.reward_total={"rew": reward}
        self.reward_terms={"vel_rew": self.coeff_vel*vel_reward, "z_penalty": self.coeff_z*z_penalty, "CoP": self.coeff_CoP*cop_reward}
        return reward # keep rewards positive
    
    def reward_full(self):
        return 0
    
    def reward(self):
        """ Reward function aiming at maximizing forward velocity."""
        # reward for forward velocity
        vel_reward = -self.fly_vel[0]/1000
        z_penalty = (self.fly_vel[2]/1000)**2

        if self.upside_down():
            flipped_reward=-1
        else:
            flipped_reward=0    

        reward = self.coeff_vel*vel_reward \
                 + self.coeff_flipped*flipped_reward \
                 +self.coeff_z*z_penalty

        self.reward_total={"rew": reward}
        self.reward_terms={"vel_rew": self.coeff_vel*vel_reward, "z_penalty": self.coeff_z*z_penalty}
        if self.verbose:
            print(f"reward_ tot: {reward}")
            print(f"vel:{self.coeff_vel*vel_reward}, z:{self.coeff_z*z_penalty}")
        return reward # keep rewards positive
        

    def act_angle_diff(self, raw_action):
        init_pose=list(self.nmf.init_pose.values())
        #print(init_pose)
        action=raw_action+init_pose
        return action
    

    def step(self, action):
        # Later adapt the action with MLP 
        action=self.act_angle_diff(action)
        raw_obs, info = self.nmf.step({'joints': action})
        obs = self._parse_obs(raw_obs)
        self.joint_pos = raw_obs['joints'][0, :]
        self.joint_vel = raw_obs['joints'][1, :]
        self.joint_torques = raw_obs['joints'][2, :]
        #positive: behind right up
        self.fly_pos = raw_obs['fly'][0, :]
        self.fly_vel = raw_obs['fly'][1, :]
        self.fly_ori = raw_obs['fly'][2, :]
        self.contact_forces=raw_obs['contact_forces']
        self.end_effectors=raw_obs['end_effectors']
        if self.reward_function=="default":
            reward = self.reward()  # what is your reward function?
        elif self.reward_function=="energy":
            reward=self.reward_energy()
        elif self.reward_function=="straight":
            reward=self.reward_straight()
        elif self.reward_function=="COP":
            reward=self.reward_COP()
        elif self.reward_function=="full":
            reward=self.reward_full()
        if self.upside_down():
            terminated = True
            self.reset()
        else:
            terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.nmf.render()
    
    def close(self):
        return self.nmf.close()