
import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
#from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from tqdm import trange
from flygym.util.config import all_leg_dofs
from scipy.signal import medfilt
from scipy.integrate import ode
import PIL.Image
from env import MyNMF

# Initialize simulation
run_time = 1
out_dir = Path('../CPGs')

friction = 1.0

physics_config = {
    'joint_stiffness': 2500,
    'friction': (friction, 0.005, 0.0001),
    'gravity': (0, 0, -9.81e5)}
terrain_config = {'fly_pos': (0, 0, 300),
                  'friction': (friction, 0.005, 0.0001)}

nmf_env = MyNMF(render_mode='viewer',
                        verbose=0,
                        reward="default",
                         timestep=1e-4,
                         render_config={'playspeed': 0.1, 'camera': 'Animat/camera_left_top'},
                         init_pose='default',
                         actuated_joints=all_leg_dofs)


####################
# Tripod Gait ####3#
######################
# Load recorded data
data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
with open(data_path / 'behavior' / 'single_steps.pkl', 'rb') as f:
    data = pickle.load(f)

    # Interpolate 5x
step_duration = len(data['joint_LFCoxa'])
interp_step_duration = int(step_duration * data['meta']['timestep'] / nmf_env.nmf.timestep)
step_data_block_base = np.zeros((len(nmf_env.nmf.actuated_joints), interp_step_duration))
measure_t = np.arange(step_duration) * data['meta']['timestep']
interp_t = np.arange(interp_step_duration) * nmf_env.nmf.timestep
for i, joint in enumerate(nmf_env.nmf.actuated_joints):
    step_data_block_base[i, :] = np.interp(interp_t, measure_t, data[joint])

step_data_block_manualcorrect = step_data_block_base.copy()

for side in ["L", "R"]:
    step_data_block_manualcorrect[nmf_env.nmf.actuated_joints.index(f"joint_{side}MCoxa")] += np.deg2rad(10) # Protract the midlegs
    step_data_block_manualcorrect[nmf_env.nmf.actuated_joints.index(f"joint_{side}HFemur")] += np.deg2rad(-5) # Retract the hindlegs
    step_data_block_manualcorrect[nmf_env.nmf.actuated_joints.index(f"joint_{side}HTarsus1")] -= np.deg2rad(15) # Tarsus more parallel to the ground (flexed) (also helps with the hindleg retraction)
    step_data_block_manualcorrect[nmf_env.nmf.actuated_joints.index(f"joint_{side}FFemur")] += np.deg2rad(15) # Protract the forelegs (slightly to conterbalance Tarsus flexion)
    step_data_block_manualcorrect[nmf_env.nmf.actuated_joints.index(f"joint_{side}FTarsus1")] -= np.deg2rad(15) # Tarsus more parallel to the ground (flexed) (add some retraction of the forelegs)

n_joints = len(nmf_env.nmf.actuated_joints)


legs = ["RF", "RM", "RH", "LF", "LM", "LH"]
n_oscillators = len(legs)

leg_ids = np.arange(len(legs)).astype(int)
joint_ids = np.arange(n_joints).astype(int)
# Map the id of the joint to the leg it belongs to (usefull to go through the steps for each legs)
match_leg_to_joints = np.array([i  for joint in nmf_env.nmf.actuated_joints for i, leg in enumerate(legs) if leg in joint])

def advancement_transfer(phases, step_dur=interp_step_duration):
    """From phase define what is the corresponding timepoint in the joint dataset
    In the case of the oscillator, the period is 2pi and the step duration is the period of the step
    We have to match those two"""

    period = 2*np.pi
    #match length of step to period phases should have a period of period mathc this perios to the one of the step
    t_indices = np.round(np.mod(phases*step_dur/period, step_dur-1)).astype(int)
    t_indices = t_indices[match_leg_to_joints]
    
    return t_indices


num_steps_base = int(run_time / nmf_env.nmf.timestep)



dt = nmf_env.nmf.timestep  # seconds
t = np.arange(0, run_time, dt)

# lets say we want 10 oscillations in the time period
n_steps = 10
frequencies = np.ones(n_oscillators) * n_steps / run_time

# For now each oscillator have the same amplitude
target_amplitude = 1.0
target_amplitudes = np.ones(n_oscillators) * target_amplitude
rate = 10
rates = np.ones(n_oscillators) * rate

# The bias matrix is define as follow: each line is the i oscillator and each column is the j oscillator couplign goes from i to j
# We express the bias in percentage of cycle 
phase_biases_measured= np.array([[0, 0.425, 0.85, 0.51, 0, 0],
                                  [0.425, 0, 0.425, 0, 0.51, 0],
                                  [0.85, 0.425, 0, 0, 0, 0.51],
                                  [0.51, 0, 0, 0, 0.425, 0.85],
                                  [0, 0.51, 0, 0.425, 0, 0.425],
                                  [0, 0, 0.51, 0.85, 0.425, 0]])
                                
phase_biases_idealized = np.array([[0, 0.5, 1.0, 0.5, 0, 0],
                                   [0.5, 0, 0.5, 0, 0.5, 0],
                                   [1.0, 0.5, 0, 0, 0, 0.5],
                                   [0.5, 0, 0, 0, 0.5, 1.0],
                                   [0, 0.5, 0, 0.5, 0, 0.5],
                                   [0, 0, 0.5, 1.0, 0.5, 0]]) 
# Phase bias of one is the same as zero (1 cycle difference)
# If we would use a phase bias of zero, we would need to change the coupling weight strategy

phase_biases_l_tetra = np.array([[0, 0.325, 0.675, 0.325, 0, 0],
                                [0.325, 0, 0.325, 0, 0.325, 0],
                                [0.675, 0.325, 0, 0, 0, 0.325],
                                [0.325, 0, 0, 0, 0.325, 0.675],
                                [0, 0.325, 0, 0.325, 0, 0.325],
                                [0, 0, 0.325, 0.625, 0.325, 0]]) 

phase_biases_r_tetra = np.array([[0, 0.325, 0.625, 0.625, 0, 0],
                                   [0.325, 0, 0.325, 0, 0.625, 0],
                                   [0.625, 0.325, 0, 0, 0, 0.625],
                                   [0.625, 0, 0, 0, 0.325, 0.625],
                                   [0, 0.625, 0, 0.325, 0, 0.325],
                                   [0, 0, 0.625, 0.625, 0.325, 0]]) 

phase_biases_wave = np.array([[0, 0.175, 0.325, 0.5, 0, 0],
                            [0.175, 0, 0.175, 0, 0.5, 0],
                            [0.325, 0.175, 0, 0, 0, 0.5],
                            [0.5, 0, 0, 0, 0.175, 0.325],
                            [0, 0.5, 0, 0.175, 0, 0.175],
                            [0, 0, 0.5, 0.325, 0.175, 0]]) 

phase_biases = phase_biases_measured * 2 * np.pi

coupling_weights = (np.abs(phase_biases) > 0).astype(float) * 10.0 #* 10.0



def phase_oscillator(_time, state):
    """Phase oscillator model used in Ijspeert et al. 2007"""
    
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]

    # NxN matrix with the phases of the oscillators
    phase_matrix = np.tile(phases, (n_oscillators, 1))

    # NxN matrix with the amplitudes of the oscillators
    amp_matrix = np.tile(amplitudes, (n_oscillators, 1))

    freq_contribution = 2*np.pi*frequencies

    #  scaling of the phase differences between oscillators by the amplitude of the oscillators and the coupling weights
    scaling = np.multiply(amp_matrix, coupling_weights)

    # phase matrix and transpose substraction are analogous to the phase differences between oscillators, those should be close to the phase biases
    phase_shifts_contribution = np.sin(phase_matrix-phase_matrix.T-phase_biases)

    # Here we compute the contribution of the phase biases to the derivative of the phases
    # we mulitply two NxN matrices and then sum over the columns (all j oscillators contributions) to get a vector of size N 
    coupling_contribution = np.sum(np.multiply(scaling, phase_shifts_contribution), axis=1)

    # Here we compute the derivative of the phases given by the equations defined previously. 
    # We are using for that matrix operations to speed up the computation
    dphases =  freq_contribution + coupling_contribution
    
    damplitudes = np.multiply(rates, target_amplitudes-amplitudes)
    
    return np.concatenate([dphases, damplitudes])

def sine_output(phases, amplitudes):
        return amplitudes * (1 + np.cos(phases))


np.random.seed(42)

# Set solver
solver = ode(f=phase_oscillator)
solver.set_integrator('dop853')
initial_values = np.random.rand(2*n_oscillators)
solver.set_initial_value(y=initial_values, t=nmf_env.nmf.curr_time)

# Initialize states and amplitudes
n_stabilisation_steps = 1000
num_steps = n_stabilisation_steps + num_steps_base

phases = np.zeros((num_steps, n_oscillators))
amplitudes = np.zeros((num_steps, n_oscillators))

joint_angles = np.zeros((num_steps, n_joints))

obs_list_tripod = []

for i in range(num_steps):

    res = solver.integrate(nmf_env.nmf.curr_time)
    phase = res[:n_oscillators]
    amp = res[n_oscillators:2*n_oscillators]

    phases[i, :] = phase
    amplitudes[i, :] = amp    

    if i> n_stabilisation_steps:
        indices = advancement_transfer(phase)
        # scale the amplitude of the joint angles to the output amplitude (High values of amplitude will highly alter the steps) 
        # With an amplitude of one, the joint angles will be the same as the one from the base step
        # With an amplitude of zero, the joint angles will be the same as the zero inidices of the base step (default pose)
        # The rest is a linear interpolation between those two
        action = {'joints': step_data_block_manualcorrect[joint_ids, 0] + \
                  (step_data_block_manualcorrect[joint_ids, indices]-step_data_block_manualcorrect[joint_ids, 0])*amp[match_leg_to_joints]}
        #action = {'joints': step_data_block_base[joint_ids, indices]}
    else:
        action = {'joints': step_data_block_manualcorrect[joint_ids, 0]}

    joint_angles[i, :] = action['joints']
    
    #print(action['joints'])
    obs, reward, terminated, truncated, info = nmf_env.step(action['joints'])
    #obs, info = nmf_env.nmf.step(action)
    obs_list_tripod.append(obs)

    raw_obs = nmf_env.nmf._get_observation()
    joint_pos = raw_obs['joints'][0, :]
    fly_pos = raw_obs['fly'][0, :]
    fly_vel = raw_obs['fly'][1, :]
    fly_ori = raw_obs['fly'][2, :]
    #print(f"position: {fly_pos/1000}")
    #print(f"orientation: {fly_vel/1000}")
    nmf_env.render()

# obs, _ = nmf_env_rendered.reset()
# obs_list = []
# rew_list = []
# for i in range(int(run_time / nmf_env_rendered.nmf.timestep)):
#     action, _ = nmf_model.predict(obs)
#     obs, reward, terminated, truncated, info = nmf_env_rendered.step(action)
#     obs_list.append(obs)
#     rew_list.append(reward)
#     nmf_env_rendered.render()

