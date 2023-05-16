import gymnasium as gym
from gymnasium import spaces
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
import numpy as np
#from scipy.spatial import ConvexHull, Point

class MyNMF(gym.Env):
    def __init__(self, **kwargs):
        self.nmf = NeuroMechFlyMuJoCo(**kwargs)
        num_dofs = len(self.nmf.actuated_joints)
        bound = 0.5
        self.action_space = spaces.Box(low=-bound, high=bound,
                                       shape=(num_dofs,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(num_dofs,))
        self.joint_pos= np.zeros(num_dofs)
        self.fly_pos = np.zeros(3)
        self.fly_vel = np.zeros(3)
        self.fly_ori = np.zeros(3)
        self.contact_forces=np.zeros(6*3)
        self.end_effectors=np.zeros(6*3)
        self.reward_terms={"rew_total": 0}

    
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
        self.reward_terms={"rew_total": 0}
        raw_obs, info = self.nmf.reset()
        return self._parse_obs(raw_obs), info
    
    def upside_down(self):
        """ Check if the robot is upside down."""
        ori_ref=[-1.5, 0, 0.88]
        margin= 0.3
        if np.any(self.fly_ori-ori_ref < -margin) or np.any(self.fly_ori-ori_ref > margin):
            print("flipped")
            return True
        else:
            return False
    
    def reward(self):
        """ Reward function aiming at maximizing forward velocity."""
        # reward for forward velocity
        vel_tracking_reward = -self.fly_vel[0]
        print(f"fly vel {self.fly_vel/1000}")
        #standing stability
        z_penalty = -0.5*(self.fly_vel[2]/1000)**2
        #print(reward)
        if self.upside_down():
            flipped_reward=-100
        else:
            flipped_reward=0    
        #vel_tracking_reward = 0.2 * np.exp( -1/ 0.25 *  (self.robot.GetBaseLinearVelocity()[0] - des_vel_x)**2 )
        
        
        # print(f"contact_forces {self.contact_forces}")
        # print(f"end eff pos {self.end_effectors}")
        # # Compute the center of pressure
        # total_force = np.sum(self.contact_forces, axis=0)
        # if total_force > 0:
        #     cop = np.sum(contact_forces, axis=0)* / total_force
        # else:
        #     cop = np.zeros(3)
        
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
        
        
        # # minimize yaw (go straight) 
        # yaw_reward = -0.3 * np.abs(self.robot.GetBaseOrientationRollPitchYaw()[2]) 

        # #minmize roll (not fall on the side)
        # roll_reward = -0.3 * np.abs(self.robot.GetBaseOrientationRollPitchYaw()[0]) 

        # # penalty body y - don't drift laterally
        # drift_reward = -0.01 * abs(self.robot.GetBasePosition()[1]) #0.01

        # #linear velocity penalty body z (mentioned in paper)
        # vel_body_z_penalty = -0.05*(self.robot.GetBaseLinearVelocity()[2])**2
        # # minimize energy 
        # energy_reward = 0 
        # for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
        # energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

        # def compute_mechanical_work(self, joint_velocities, joint_torques):
        # """ Computes the mechanical work spent by the animal. """
        # return np.abs(
        #     joint_torques@joint_velocities.T
        # ) * self.time_step / self.run_time

        reward = vel_tracking_reward \
                 + flipped_reward \
                 +z_penalty
        #         + yaw_reward \
        #         + roll_reward\
        #         + drift_reward \
        #         + vel_body_z_penalty \
        #         - 0.01 * energy_reward \
        #         - 0.1 * np.linalg.norm(self.robot.GetBaseOrientation() - np.array([0,0,0,1]))
        print(reward)
        self.reward_terms=self.reward_terms={"rew_total": reward, "vel_rew": vel_tracking_reward, "flipped_reward": flipped_reward}
        return reward # keep rewards positive
        
    def step(self, action):
        # Later adapt the action with MLP 

        raw_obs, info = self.nmf.step({'joints': action})
        obs = self._parse_obs(raw_obs)
        self.joint_pos = raw_obs['joints'][0, :]
        #positive: behind right up
        self.fly_pos = raw_obs['fly'][0, :]
        self.fly_vel = raw_obs['fly'][1, :]
        self.fly_ori = raw_obs['fly'][2, :]
        self.contact_forces=raw_obs['contact_forces']
        self.end_effectors=raw_obs['end_effectors']
        reward = self.reward()  # what is your reward function?
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