from flygym.util.config import all_leg_dofs
from flygym.util.config import leg_dofs_3_per_leg, leg_dofs_2_per_leg

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
#using the config 3 Dofs per leg we just get the joint positions in observation
# if we would like to use other info in observation need to change observation space
import os
import argparse
import datetime

parser = argparse.ArgumentParser(
                    prog='RL Training',
                    description='RL training on mujoco',
                    epilog='use -r to choose the reward function')

parser.add_argument('-n', '--name', default="0")
parser.add_argument('-r', '--reward', default="default")      # option that takes a value
args = parser.parse_args()


COLAB=False
print(f"working dir: {os.getcwd()}")
# Get the absolute path of the parent directory of the cwd
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# Construct the path to the TL_logs directory
date=datetime.datetime.now()
day=date.strftime("%d")
month=date.strftime("%m")
SAVE_NAME=f"{day}{month}_{args.name}_{args.reward}"
#SAVE_NAME="trash_test"


print(SAVE_NAME)
SAVE_PATH = 'RL_logs/models/'+f"PPO_{SAVE_NAME}/" #datetime.now().strftime("%m%d%y%H%M%S") + '/'
if COLAB:
    SAVE_PATH=os.path.join(os.getcwd(),SAVE_PATH)
    TB_LOG="RL_logs/logdir"
else:
    SAVE_PATH = os.path.join(parent_dir, SAVE_PATH)
    TB_LOG="../RL_logs/logdir"
os.makedirs(SAVE_PATH, exist_ok=True)
print(SAVE_PATH)


from utils_RL import CheckpointCallback
from env import MyNMF

config_decentralized = {'reward': args.reward,
                        'verbose':0,
                        'obs_mode': 'augmented',
                        'control_mode': "Decentralized",
                        'render_mode': 'headless',
                        'timestep': 1e-4,
                        'init_pose': 'default',
                        'actuated_joints': all_leg_dofs,
                        }   

config_RL = {'reward': args.reward,
            'verbose':0,
            'obs_mode': 'augmented',
            'control_mode': "RL",
            'render_mode': 'headless',
            'timestep': 1e-4,
            'init_pose': 'default',            
            'actuated_joints': leg_dofs_3_per_leg,
            'terrain_config':{'fly_pos':(0, 0, 600)}}   

nmf_env_headless = MyNMF(**config_decentralized)  # which DoFs would you use?

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=SAVE_PATH,name_prefix='rl_model', env=nmf_env_headless, rew_freq=1000, verbose=2)


nmf_model = PPO(MlpPolicy, nmf_env_headless, verbose=1, tensorboard_log=TB_LOG)
nmf_model.learn(total_timesteps=1000_000, log_interval=1,callback=checkpoint_callback, tb_log_name=SAVE_NAME)

env=nmf_model.get_env()
env.close()
