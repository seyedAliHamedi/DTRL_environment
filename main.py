import os

from environment.env import Environment
import pandas as pd
from data.configs import environment_config, monitor_config
import itertools


def run_env(run_index, iteration, config, path=monitor_config['paths']['time']['plot'],
            log_file="environment/config_testing/env_run_log.txt"):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    start_msg = (f"Started test={run_index} | CONFIG:: multi_agent={config['multi_agent']}, "
                 f"size: {config['window']['size']}, max_jobs: {config['window']['max_jobs']}, "
                 f"clock: {config['window']['clock']}\n")
    print(start_msg)
    env = Environment(n_iterations=iteration, display=False, config=config, path=path)
    total_t, jobs_done, wait_queue = env.run()
    result_msg = (f"Run{run_index}: total_t={total_t}, iter={iteration}, jobs_done={jobs_done - 1}, "
                  f"wait-queue={wait_queue}\n")
    print(result_msg)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    with open(log_file, "a") as f:
        f.write(start_msg)
        f.write(result_msg)
        f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


def test_configs(configs, iteration, log_file="env_run_log.txt"):
    all_configs = []
    for config in configs:
        for combination in itertools.product(config['multi_agent'], config['size'], config['clock'],
                                             config['max_jobs']):
            all_configs.append({
                "multi_agent": combination[0],
                "window": {"size": combination[1], "max_jobs": combination[3], "clock": combination[2]},
                "environment": environment_config['environment']
            })

    for idx, config in enumerate(all_configs):
        run_env(run_index=idx, iteration=iteration, config=config,
                path=f'environment/config_testing/time_plot{idx}.png', log_file=log_file)


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = '{:,.5f}'.format

    from data.testing_configs import testing_conf

    test_configs(testing_conf, 1000)
