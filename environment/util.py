import numpy as np
from matplotlib import pyplot as plt


def plot_histories(init_punish, lossHistory, avg_time_history, avg_energy_history, avg_fail_history, iot_usage,
                   mec_usage, cc_usage):
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))

    plt.suptitle(f"Training History with setup 1 , initial punish: {init_punish}",
                 fontsize=16, fontweight='bold')

    loss_values = lossHistory
    axs[0, 0].plot(loss_values, label='Average Loss',
                   color='blue', marker='o')  # Add markers for clarity
    axs[0, 0].set_title('Average Loss History')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot for average time history
    time_values = np.array(avg_time_history)  # Ensure data is in numpy array
    axs[0, 1].plot(time_values, label='Average Time', color='red', marker='o')
    axs[0, 1].set_title('Average Time History')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Time')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    time_lower_bound = 0.00625
    time_middle_bound = 0.0267
    time_upper_bound = 1
    # axs[0, 1].axhline(y=time_lower_bound, color='blue',
    #                   linestyle='--', label='Lower Bound (0.00625)')
    # axs[0, 1].axhline(y=time_middle_bound, color='green',
    #                   linestyle='--', label='Middle Bound (0.0267)')
    # axs[0, 1].axhline(y=time_upper_bound, color='red',
    #                   linestyle='--', label='Upper Bound (1)')
    axs[0, 1].legend()

    # Plot for average energy history
    energy_values = np.array(avg_energy_history)
    axs[1, 0].plot(energy_values, label='Average Energy',
                   color='green', marker='o')
    axs[1, 0].set_title('Average Energy History')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Energy')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    energy_lower_bound = 0.0000405
    energy_middle_bound = 0.100746
    energy_upper_bound = 1.2
    # axs[1, 0].axhline(y=energy_lower_bound, color='blue',
    #                   linestyle='--', label='Lower Bound (0.0000405)')
    # axs[1, 0].axhline(y=energy_middle_bound, color='green',
    #                   linestyle='--', label='Middle Bound (0.100746)')
    # axs[1, 0].axhline(y=energy_upper_bound, color='red',
    #                   linestyle='--', label='Upper Bound (1.2)')
    axs[1, 0].legend()

    # Plot for average fail history
    fail_values = np.array(avg_fail_history)
    axs[1, 1].plot(fail_values, label='Average Fail',
                   color='purple', marker='o')
    axs[1, 1].set_title('Average Fail History')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Fail Count')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Plot for devices usage history
    axs[2, 0].plot(iot_usage, label='IoT Usage', color='blue', marker='o')
    axs[2, 0].plot(mec_usage, label='MEC Usage', color='orange', marker='x')
    axs[2, 0].plot(cc_usage, label='Cloud Usage', color='green', marker='s')
    axs[2, 0].set_title('Devices Usage History')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Usage')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    axs[2, 1].set_title(
        f'Path History Heatmap ')
    axs[2, 1].set_xlabel('Output Classes')
    axs[2, 1].set_ylabel('Epochs')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(f"./results/Energy Figs/r{rSetup}_p{init_punish}")
    plt.show()
