import gymnasium as gym
from gymnasium import spaces
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#from scipy.spatial import ConvexHull, Point
from flygym.util.data import default_pose_path
from decentralized import Decentralized_Controller


class MyNMF(gym.Env):
    def __init__(self, obs_mode="default", control_mode="RL", reward="default", verbose=0,**kwargs):
        self.nmf = NeuroMechFlyMuJoCo(**kwargs)
        self.num_dofs = len(self.nmf.actuated_joints)
        self.bound = 0.1

        self.num_steps=10_000
        self.stab_steps=0
        self.counter=0
        self.timestep=self.nmf.timestep    

        self.control_mode=control_mode
        if control_mode=="RL":
            print("training a pure RL controller")
        elif control_mode=="CPG":
            print("training a CPG RL controller")
        elif control_mode=="Decentralized":
            self.decentralized=Decentralized_Controller(self.nmf)
            print("training RL-Decentralized controller")
        else:
            print(f"!!! Control mode {control_mode} is not implemented")
        
        self.init_action_space()
        self.obs_mode=obs_mode
        self.init_obs_space()

        #self.init_stab()
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,))
        self.joint_pos= np.zeros(self.num_dofs)
        self.joint_vel= np.zeros(self.num_dofs)
        self.joint_torques= np.zeros(self.num_dofs)
        self.fly_pos = np.zeros(3)
        self.fly_vel = np.zeros(3)
        self.fly_ori = np.zeros(3)
        self.contact_forces=np.zeros(6*3)
        self.end_effectors=np.zeros(6*3)
        self.metrics={"pos_x": self.fly_pos[0], "pos_y": self.fly_pos[1], "pos_z": self.fly_pos[2]}
        self.reward_function=reward
        self.reward_total={"rew": 0}
        self.reward_terms={"rew_vel": 0}
        
        self.verbose=verbose

        self.ori_ref=[-1.5, 0, 0.88]
        self.flipped_margin=[1.5, 1.5, 0.5]

        self.coeff_vel=1
        self.coeff_flipped=200
        self.coeff_height=50
        self.coeff_vel_z=-0.005
        self.coeff_pos_z=-30
        self.coeff_energy=-1e05
        self.coeff_yaw=-2
        self.coeff_roll=-4
        self.coeff_drift=-0.005
        
    def init_action_space(self):
        if self.control_mode=="RL":
            print("using joint angle diff as actions")
            self.action_space = spaces.Box(low=-self.bound, high=self.bound,
                                       shape=(self.num_dofs,))
        elif self.control_mode=="CPG":
            print("training a CPG RL controller")
        elif self.control_mode=="Decentralized":
            print("training RL-Decentralized controller")
            self.action_space = spaces.Box(low=-self.bound, high=self.bound,
                                       shape=(6,))
        else:
            print(f"!!! Control mode {self.control_mode} is not implemented")
    
    def init_obs_space(self):
        num_joints=18
        if self.obs_mode=="default":
            print("using default observation space")
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs,))
        elif self.obs_mode=="standard":
            dim=num_joints*3
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))
        elif self.obs_mode=="augmented":
            dim=self.num_dofs*3+3*4+3*4+6*3+6*3
            #print(f"dim {dim}")
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))
        
    
    def _parse_obs(self, raw_obs):
        if self.obs_mode=="default":
            features = [raw_obs['joints'][0, :].flatten()]
        elif self.obs_mode=="standard":
            features = [
            #joint angle
            raw_obs['joints'][0, :].flatten(),
            #joint velocity
            raw_obs['joints'][1, :].flatten(),
            #joint torque
            raw_obs['joints'][2, :].flatten()
            #raw_obs['joints'][1, :].flatten(),
            # raw_obs['fly'].flatten(),
            # what else would you like to include?
            ]
            #print(raw_obs['joints'].shape)
            #print(np.array(features).shape)
        elif self.obs_mode=="augmented":
            features = [
                raw_obs['joints'][0, :].flatten(),
                raw_obs['joints'][1, :].flatten(),
                raw_obs['joints'][2, :].flatten(),
                raw_obs['fly'][0, :].flatten(), #fly pos
                raw_obs['fly'][1, :].flatten(), #fly vel
                raw_obs['fly'][2, :].flatten(), #fly ori
                raw_obs['fly'][3, :].flatten(), #fly ang vel
                raw_obs['contact_forces'].flatten(),
                raw_obs['end_effectors'].flatten()]
        #print(np.array(features).shape)
        return np.concatenate(features, dtype=np.float32)
    
    def reset(self):
        #self.reward_terms={"rew_total": 0}
        self.counter=0
        if self.verbose:
            print("resetting the environment")
        raw_obs, info = self.nmf.reset()
        return self._parse_obs(raw_obs), info
    

    def height_ok(self):
        """ Check if the robot is upside down."""
        boundaries=[1.5, 2]
        if self.fly_pos[2]/1000 > boundaries[0] and self.fly_pos[2]/1000 < boundaries[1]:
            if self.verbose:
                print("z height acceptable")
            return True
        else:
            #if self.verbose:
            print("z height not acceptable")
            return False

    def upside_down(self):
        """ Check if the robot is upside down."""
        for i in range(len(self.fly_ori)):
            margin= self.flipped_margin[i]    
            if np.abs(self.fly_ori[i]-self.ori_ref[i]) > margin:
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
        vel_reward = self.fly_vel[0]/1000
        ref_z=1.9
        z_pos_penalty = (np.abs(self.fly_pos[2]/1000-ref_z))**2
        if self.upside_down():
            flipped_reward=-1
        else:
            flipped_reward=0    
        
        #minmize roll (not fall on the side)
        roll_reward = np.abs(self.fly_ori[0]-self.ori_ref[0])
        # minimize yaw (go straight) 
        yaw_reward = np.abs(self.fly_ori[2]-self.ori_ref[2])
        # penalty body y - don't drift laterally
        drift_reward = np.abs(self.fly_pos[1]) #0.01


        reward = self.coeff_vel*vel_reward \
                 + self.coeff_flipped*flipped_reward \
                 + self.coeff_pos_z*z_pos_penalty \
                 + self.coeff_yaw * yaw_reward \
                 + self.coeff_roll * roll_reward\
                 + self.coeff_drift * drift_reward \

        if self.verbose:
            print(f"total rew {reward}")
            print(f"vel {self.coeff_vel*vel_reward}, z_pos {self.coeff_pos_z*z_pos_penalty}")
            print(f"yaw {self.coeff_yaw*yaw_reward}, roll {self.coeff_roll*roll_reward} drift {self.coeff_drift*drift_reward}")
       
        self.reward_total={"rew": reward}
        self.reward_terms={ "vel_rew": self.coeff_vel*vel_reward, \
                            "z_penalty": self.coeff_pos_z*z_pos_penalty, \
                            "yaw": self.coeff_yaw * yaw_reward, \
                            "roll": self.coeff_roll * roll_reward, \
                            "drift": self.coeff_drift * drift_reward}
        return reward 
    
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
    
    def reward_stab(self):
        """ Reward function aiming at maximizing forward velocity."""
        # reward for forward velocity
        #vel_reward = self.fly_vel[0]/1000
        #vel_reward = self.fly_vel[0]/1000
        z_vel_penalty = (self.fly_vel[2]/1000)**2
        #print(f"z_pos {self.fly_pos[2]/1000}")
        #print(f"pitch {self.fly_ori}")
        z_pos_penalty = (np.abs(self.fly_pos[2]/1000-1.9))**2
        #print(z_pos_penalty)
                #minmize roll (not fall on the side)
        pitch_reward = (self.fly_ori[2])**2 #default pitch is 0
        if self.upside_down():
            flipped_reward=-1
        else:
            flipped_reward=0 

        if not self.height_ok(): 
            height_rew=-1
        else:
            height_rew=0 

        reward = self.coeff_flipped*flipped_reward \
                 +self.coeff_height*height_rew \
                 +self.coeff_pos_z*z_pos_penalty \
                 -1*pitch_reward \
                + 10*self.counter*self.timestep
                 #self.coeff_pos_z*z_pos_penalty
                #-1*z_pos_penalty \
                #-0.5*z_vel_penalty \
        self.reward_total={"rew": reward}
        self.reward_terms={"z_vel_penalty": self.coeff_vel_z*z_vel_penalty}
        if self.verbose:
            print(f"reward_ tot: {reward}")
            print(f"z_vel:{-0.1 *z_vel_penalty}, pos_z:{self.coeff_pos_z*z_pos_penalty}, pitch: {pitch_reward}, counter: {10*self.counter*self.timestep}")
        return reward # keep rewards positive
        
    
    def reward(self):
        """ Reward function aiming at maximizing forward velocity."""
        # reward for forward velocity
        vel_reward = self.fly_vel[0]/1000
        #vel_reward = self.fly_vel[0]/1000
        #z_vel_penalty = (self.fly_vel[2]/1000)**2

        init_pose=list(self.nmf.init_pose.values())
        z_pos_penalty = ((self.fly_pos[2]-init_pose[2])/1000)**2

        if self.upside_down():
            flipped_reward=-1
        else:
            flipped_reward=0    

        reward = self.coeff_vel*vel_reward \
                 + self.coeff_flipped*flipped_reward \
                 +self.coeff_vel_z*z_vel_penalty \
                 +self.coeff_pos_z*z_pos_penalty

        self.reward_total={"rew": reward}
        self.reward_terms={"vel_rew": self.coeff_vel*vel_reward, "z_vel_penalty": self.coeff_vel_z*z_vel_penalty, "z_pos_penalty": self.coeff_pos_z*z_pos_penalty}
        if self.verbose:
            print(f"reward_ tot: {reward}")
            print(f"vel:{self.coeff_vel*vel_reward}, z_vel:{self.coeff_vel_z*z_vel_penalty}, z_pos:{self.coeff_pos_z*z_pos_penalty}, flipped:{self.coeff_flipped*flipped_reward}")
        return reward # keep rewards positive
        

    def act_angle_diff(self, raw_action):
        init_pose=list(self.nmf.init_pose.values())
        #print(init_pose)
        action=raw_action+init_pose
        return action

    def act_prev_angle(self, raw_action):
        prev_angles=self.nmf._get_observation()["joints"][0,:]
        action=raw_action+prev_angles
        return action

    def stabilization(self, action):
        #to stabilize the fly at the beginning of the simulation  
        while self.counter<self.stab_steps:
            action=np.zeros(self.num_dofs) 
            action=self.act_prev_angle(action)  
            raw_obs, info = self.nmf.step({'joints': action})
            self.counter+=1
        else: 
            if self.verbose: print("finished stabilization")

    def init_stab(self):
        from pathlib import Path
        import pkg_resources
        import pickle
        # Load recorded data
        data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
        with open(data_path / 'behavior' / 'single_steps.pkl', 'rb') as f:
            data = pickle.load(f)

            # Interpolate 5x
        step_duration = len(data['joint_LFCoxa'])
        interp_step_duration = int(step_duration * data['meta']['timestep'] / self.timestep)
        step_data_block_base = np.zeros((len(self.nmf.actuated_joints), interp_step_duration))
        measure_t = np.arange(step_duration) * data['meta']['timestep']
        interp_t = np.arange(interp_step_duration) * self.timestep
        for i, joint in enumerate(self.nmf.actuated_joints):
            step_data_block_base[i, :] = np.interp(interp_t, measure_t, data[joint])

        step_data_block_manualcorrect = step_data_block_base.copy()

        print(f"actuated joints: {self.nmf.actuated_joints}")
        for side in ["L", "R"]:
            #step_data_block_manualcorrect[self.nmf.actuated_joints.index(f"joint_{side}MCoxa")] += np.deg2rad(10) # Protract the midlegs
            step_data_block_manualcorrect[self.nmf.actuated_joints.index(f"joint_{side}HFemur")] += np.deg2rad(-5) # Retract the hindlegs
            step_data_block_manualcorrect[self.nmf.actuated_joints.index(f"joint_{side}HTarsus1")] -= np.deg2rad(15) # Tarsus more parallel to the ground (flexed) (also helps with the hindleg retraction)
            step_data_block_manualcorrect[self.nmf.actuated_joints.index(f"joint_{side}FFemur")] += np.deg2rad(15) # Protract the forelegs (slightly to conterbalance Tarsus flexion)
            step_data_block_manualcorrect[self.nmf.actuated_joints.index(f"joint_{side}FTarsus1")] -= np.deg2rad(15) # Tarsus more parallel to the ground (flexed) (add some retraction of the forelegs)

        self.step_data_block_manualcorrect = step_data_block_manualcorrect  
        n_joints = len(self.nmf.actuated_joints)
        legs = ["RF", "RM", "RH", "LF", "LM", "LH"]

        #leg_ids = np.arange(len(legs)).astype(int)
        self.joint_ids = np.arange(n_joints).astype(int)
        # Map the id of the joint to the leg it belongs to (usefull to go through the steps for each legs)
        #match_leg_to_joints = np.array([i  for joint in self.nmf.actuated_joints for i, leg in enumerate(legs) if leg in joint])

    def stabilization2(self, action):
        #to stabilize the fly at the beginning of the simulation 
        n_stabilisation_steps = 1000 
        print("here")
        while self.counter<n_stabilisation_steps:
            action = {'joints': self.step_data_block_manualcorrect[self.joint_ids, 0]}
            raw_obs, info = self.nmf.step({'joints': action})
            self.counter+=1
        else: 
            if self.verbose: print("finished stabilization")


    def step(self, action):
        # Later adapt the action with MLP 
        self.stabilization(action)

        if self.control_mode=="RL":
            #action=self.act_angle_diff(action)
            action=self.act_prev_angle(action) 
            raw_obs, info = self.nmf.step({'joints': action})
        # if self.counter<self.stab_steps: 

        if self.control_mode=="Decentralized":
            #print(action)
            action=self.decentralized.stepping(action)
            raw_obs, info = self.nmf.step(action)#
            #raw_obs=self.nmf._get_observation()
            #info=self.nmf._get_info()
        #     self.stabilization()    
        

        self.counter+=1
        if self.verbose: print(f"counter: {self.counter}")

        obs = self._parse_obs(raw_obs)
        self.joint_pos = raw_obs['joints'][0, :]
        self.joint_vel = raw_obs['joints'][1, :]
        self.joint_torques = raw_obs['joints'][2, :]
        #positive: behind right up
        self.fly_pos = raw_obs['fly'][0, :]
        self.fly_vel = raw_obs['fly'][1, :]
        self.fly_ori = raw_obs['fly'][2, :]
        self.metrics={"pos_x": self.fly_pos[0], "pos_y": self.fly_pos[1], "pos_z": self.fly_pos[2]}
        self.contact_forces=raw_obs['contact_forces']
        self.end_effectors=raw_obs['end_effectors']
        if self.verbose:
            print(f"fly pos: {self.fly_pos}")
            print(f"fly vel: {self.fly_vel}")
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
        elif self.reward_function=="stab":
            reward=self.reward_stab()
        if self.upside_down() or self.counter>self.num_steps :
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