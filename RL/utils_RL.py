import os
#from stable_baselines.bench.monitor import load_results
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


def get_latest_model(path):
	""" Returns most recent model saved in path directory. """
	files = os.listdir(path)
	paths = [os.path.join(path, basename) for basename in files if basename.endswith('.zip')]
	return max(paths, key=os.path.getctime)


class CheckpointCallback(BaseCallback):
    """
    Added Custom Logging to the Callback 
    Callback for saving a model and vec_env parameters every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', env=None, rew_freq=100, verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.env=env
        self.rew_freq=rew_freq

    def _init_callback(self):# -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
       
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        # Add a TensorBoardOutputFormat object to the logger's output formats
        #if isinstance(self.logger, TensorBoardLogger):
        tb_format = TensorBoardOutputFormat(self.logger.dir)
        self.logger.output_formats.append(tb_format)
        output_formats = self.logger.output_formats
        print(output_formats)
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))


    def _on_step(self):# -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)

            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            #self.training_env.save(stats_path)

            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))

        if self.n_calls % 500 == 0: # self.save_freq < 10000 and 
            # also print out path periodically for off-policy aglorithms: SAC, TD3, etc.
            print('=================================== Save path is {}'.format(self.save_path))

        if self.n_calls % self.rew_freq == 0 and self.env is not None:
            #print(self.env.envs[0].env)
            self.tb_formatter.writer.add_scalars("reward/", self.env.reward_total, self.num_timesteps)
            self.tb_formatter.writer.add_scalars("reward_terms/", self.env.reward_terms, self.num_timesteps)
            self.tb_formatter.writer.add_scalars("metrics/", self.env.metrics, self.num_timesteps)
            self.tb_formatter.writer.flush()

        return True
