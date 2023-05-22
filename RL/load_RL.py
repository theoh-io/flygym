from flygym.util.config import all_leg_dofs
from flygym.util.config import leg_dofs_3_per_leg

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
#using the config 3 Dofs per leg we just get the joint positions in observation
# if we would like to use other info in observation need to change observation space
import os
import cv2


from utils_RL import CheckpointCallback
from utils_RL import get_latest_model
from env import MyNMF

run_time = 0.5

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
interm_dir = "RL_logs/models/"
exp_name = "PPO_1605_z_penalty"
log_dir = interm_dir + exp_name
log_dir = os.path.join(parent_dir, log_dir)
#stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)


nmf_env_rendered = MyNMF(render_mode='viewer',
                         timestep=1e-4,
                         init_pose='default',
                         render_config={'playspeed': 0.1,
                                        'camera': 'Animat/camera_left_top'},
                         actuated_joints=leg_dofs_3_per_leg)

nmf_model = PPO.load(model_name, env=nmf_env_rendered)


obs, _ = nmf_env_rendered.reset()
obs_list = []
rew_list = []
for i in range(int(run_time / nmf_env_rendered.nmf.timestep)):
    action, _ = nmf_model.predict(obs)
    obs, reward, terminated, truncated, info = nmf_env_rendered.step(action)
    obs_list.append(obs)
    rew_list.append(reward)
    nmf_env_rendered.render()

# closing all open windows
cv2.destroyAllWindows()

# path_vids="../RL_logs/vids"
# path=os.path.join(path_vids,'1605_flipped_penalty.mp4')
# nmf_env_rendered.nmf.save_video(path)
# nmf_env_rendered.close()


