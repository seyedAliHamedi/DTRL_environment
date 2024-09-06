import os

from environment.env import Environment
import pandas as pd
from data.configs import environment_config
import torch.multiprocessing as mp


if __name__ == '__main__':
    mp.set_start_method("spawn")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = '{:,.5f}'.format

    print("Creating environment...")
    env = Environment(10000, False)
    print("Running environment...")
    env.run()
