from environment.env import Environment
import pandas as pd
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = '{:,.5f}'.format
    env = Environment(500, False)
    env.run()
