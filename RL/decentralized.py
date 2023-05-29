import numpy as np
import pickle
from pathlib import Path

class Decentralized_Controller():
    def __init__(self,nmf,num_steps=14776) :
        self.stepping_advancement = np.zeros(6).astype(float)
        out_dir = Path('../decentralized_ctrl')
        with open(out_dir / "manual_corrected_data.pickle", 'rb') as handle:
            self.step_data_block_manualcorrect, self.leg_swing_starts, self.leg_stance_starts, n_steps_stabil = pickle.load(handle)
        self.interp_step_duration = np.shape(self.step_data_block_manualcorrect)[1]
        self.n_stabilisation_steps = n_steps_stabil
        self.nmf_gapped=nmf
        self.legs = ["RF", "LF", "RM", "LM", "RH", "LH"]
        self.leg_ids = np.arange(len(self.legs)).astype(int)
        self.leg_corresp_id = dict(zip(self.legs, self.leg_ids))
        self.n_joints = len(self.nmf_gapped.actuated_joints)
        self.joint_ids = np.arange(self.n_joints).astype(int)
        self.match_leg_to_joints = np.array([i  for joint in self.nmf_gapped.actuated_joints for i, leg in enumerate(self.legs) if leg in joint])
        self.all_initiated_legs=[]
        self.match_leg_to_joints = np.array([i  for joint in self.nmf_gapped.actuated_joints for i, leg in enumerate(self.legs) if leg in joint])
        self.all_legs_contact_forces = np.zeros((num_steps,len(self.legs)))
        self.leg_force_sensors_ids = {leg:[] for leg in self.legs}
        self.num_steps=num_steps
        self.rule5_corresponding_legs = {"LH":["LM"], "LM":["LH", "RM","LF"], "LF":["LM", "RF"], "RH":["RM"], "RM":["RH", "LM","RF"], "RF":["LF", "RM"]}#"LH":["RH","LM"], "RH":["LH","RM"]

        for i, collision_geom in enumerate(self.nmf_gapped.collision_tracked_geoms):
            for leg in self.legs:
                if collision_geom.startswith(leg):
                    self.leg_force_sensors_ids[leg].append(i)  

    def rule5_decrease_increase_timestep(self,leg_contact_forces, i, counter_since_last_increase, prev_output, min_decrease_interval=50, window_size=10):
    #This function send a 1 if the step size should be decreases it returns -1 if the step size can be increased again
    #The function waits for a couple of steps before seding the signal to decrease the step size again

        step_size_action = 0
        if i < window_size:
            window_size = i

        if counter_since_last_increase < min_decrease_interval:
            counter_since_last_increase += 1
        else:
            counter_since_last_increase = 0
            if np.median(np.diff(leg_contact_forces[i-window_size:i])) < 1 and  prev_output == 1:
                step_size_action = -1

        if np.median(np.diff(leg_contact_forces[i-window_size:i])) > 1 and  counter_since_last_increase == 0:
            step_size_action = 1
            counter_since_last_increase += 1
        
        return step_size_action, counter_since_last_increase 

    def update_stepping_advancement(self,stepping_advancement, leg_idx, interp_step_duration):

        #print(leg)
        #print('interp_step_dur =', interp_step_duration)
        if stepping_advancement[leg_idx] >= interp_step_duration-1:
            stepping_advancement[leg_idx] = 0
        elif stepping_advancement[leg_idx] > 0:
            stepping_advancement[leg_idx] +=1
        return stepping_advancement
    
    def update_stepping_advancement_w5(self,stepping_advancement, leg_idxs, interp_step_duration,i,all_legs_contact_forces):
        #print("legs?")
        #print(stepping_advancement[k])
        for leg_idx in leg_idxs : 
            if stepping_advancement[leg_idx] >= interp_step_duration-1: #swing stance
                stepping_advancement[leg_idx] = 0
                # print("step>1")
            elif stepping_advancement[leg_idx] > 0:
                #print(stepping_advancement[k])
                if all_legs_contact_forces[i,leg_idx] == 0 and stepping_advancement[leg_idx]<1276: 
                    stepping_advancement[leg_idx] +=2
                    #   print(k,"Deux",stepping_advancement[k])
                else :
                    stepping_advancement[leg_idx] +=0.1
                    #  print(k,"demi",stepping_advancement[k])    
        return stepping_advancement
        

    def stepping(self,leg_scores,i=5000,percent_margin=0.001) : 
        """
        input : 
        leg_scores : array of dimension (1,6) containing the score for every leg 

        output : 

        
        """
        initiating_leg = np.argmax(leg_scores)
        within_margin_legs = leg_scores[initiating_leg]-leg_scores <= leg_scores[initiating_leg]*percent_margin
        rule5_step_size_action = np.zeros((len(self.legs), self.num_steps))
        counter_since_last_increase = np.zeros(len(self.legs))
        legs_prev_step_size_action = np.zeros(len(self.legs))
        # If multiple legs are within the margin choose randomly among those legs
        
        if np.sum(within_margin_legs) > 1:
            initiating_leg = np.random.choice(np.where(within_margin_legs)[0])
            #print("rdm choice")

        # If the maximal score is zero or less (except for the first step after stabilisation to initate the locomotion) or if the leg is already stepping
        if (leg_scores[initiating_leg] <= 0 and not i == self.n_stabilisation_steps+1) or self.stepping_advancement[initiating_leg] > 0:
            initiating_leg = None
            #print("none")
        
        else:
            self.stepping_advancement[initiating_leg] += 1
            self.all_initiated_legs.append([initiating_leg, i])
            
        if (np.floor(self.stepping_advancement[self.match_leg_to_joints]).astype(int)).all() != 1278 :
            joint_pos = self.step_data_block_manualcorrect[self.joint_ids, np.floor(self.stepping_advancement[self.match_leg_to_joints]).astype(int)] # ICI round 
        action = {'joints': joint_pos}
        obs= self.nmf_gapped._get_observation()
        #self.obs_list_cruse_gapped.append(obs)
        self.all_legs_contact_forces[i,:]=[np.sum(obs["contact_forces"][self.leg_force_sensors_ids[leg]]) for leg in self.legs]
        updated_legs=[]

        for l, leg in enumerate(self.legs):
            rule5_step_size_action[l, i], counter_since_last_increase[l] = self.rule5_decrease_increase_timestep(np.array(self.all_legs_contact_forces)[:,l], i, counter_since_last_increase[l], legs_prev_step_size_action[l])
            legs_prev_step_size_action[l] = rule5_step_size_action[l, i] if not rule5_step_size_action[l, i] == 0 else legs_prev_step_size_action[l]
            if rule5_step_size_action[l, i]==1 or legs_prev_step_size_action[l]==1: 
                #print("On est là")
                #print(rule5_step_size_action[l, i],legs_prev_step_size_action[l])
                corr_legs=[self.leg_corresp_id[l] for l in self.rule5_corresponding_legs[leg]]
                self.stepping_advancement = self.update_stepping_advancement_w5(self.stepping_advancement, corr_legs, self.interp_step_duration,i,self.all_legs_contact_forces) 
                #store updated legs
                updated_legs += [self.leg_corresp_id[l] for l in self.rule5_corresponding_legs[leg]]

        for l, leg in enumerate(self.legs):
            if l not in updated_legs :     
                self.stepping_advancement = self.update_stepping_advancement(self.stepping_advancement, l, self.interp_step_duration)        
        return action
                
    
