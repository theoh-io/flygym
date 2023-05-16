from flygym.util.config import all_leg_dofs
from flygym.util.config import leg_dofs_3_per_leg

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
#using the config 3 Dofs per leg we just get the joint positions in observation
# if we would like to use other info in observation need to change observation space
import os


COLAB=False
print(f"working dir: {os.getcwd()}")
# Get the absolute path of the parent directory of the cwd
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# Construct the path to the TL_logs directory
SAVE_NAME="1605_vel_rew_flipped"
SAVE_PATH = 'RL_logs/models/'+f"PPO_{SAVE_NAME}/" #datetime.now().strftime("%m%d%y%H%M%S") + '/'
if COLAB:
    SAVE_PATH=os.path.join(os.getcwd(),SAVE_PATH)
    TB_L0G="RL_logs/logdir"
else:
    SAVE_PATH = os.path.join(parent_dir, SAVE_PATH)
    TB_LOG="../RL_logs/logdir"
os.makedirs(SAVE_PATH, exist_ok=True)
print(SAVE_PATH)


from utils_RL import CheckpointCallback
from env import MyNMF

run_time = 0.5
nmf_env_headless = MyNMF(render_mode='headless',
                         timestep=1e-4,
                         init_pose='stretch',
                         actuated_joints=leg_dofs_3_per_leg)  # which DoFs would you use?

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=SAVE_PATH,name_prefix='rl_model', env=nmf_env_headless, rew_freq=1000, verbose=2)


nmf_model = PPO(MlpPolicy, nmf_env_headless,verbose=1, tensorboard_log=TB_LOG)
nmf_model.learn(total_timesteps=30_000, log_interval=1,callback=checkpoint_callback, tb_log_name=SAVE_NAME)

env=nmf_model.get_env()
env.close()
