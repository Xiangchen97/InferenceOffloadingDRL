import matplotlib.pyplot as plt
import numpy as np


class TrainAnalysis(object):
    def __init__(self, result_path):
        super(TrainAnalysis, self).__init__()
        self.result_path = result_path
        self.avg_r_rec = []
        self.loss_p_rec = []
        self.loss_a_rec = []
        self.T_rec = []
        self.E_rec = []
        self.T_device_rec = []
        self.T_tran_rec = []
        self.T_server_rec = []
        self.T_queue_rec = []
        self.E_device_rec = []
        self.E_tran_rec = []
        self.E_server_rec = []
        self.queue = []
        self.hit_min_queue = []

        self.fontsize_s = 8
        self.fontsize_m = 12
        self.fontsize_l = 16

    def plot_loss(self):
        loss_p_axis = np.linspace(1, len(self.loss_p_rec), len(self.loss_p_rec))
        loss_a_axis = np.linspace(1, len(self.loss_a_rec), len(self.loss_a_rec))
        plt.subplot(2, 1, 1)
        plt.plot(loss_p_axis, self.loss_p_rec, color='aqua')
        plt.xlabel('learning_times')
        plt.ylabel('Loss')
        plt.title('partition net loss records')
        plt.subplot(2, 1, 2)
        plt.plot(loss_a_axis, self.loss_a_rec, color='orange')
        plt.xlabel('learning_times')
        plt.ylabel('Loss')
        plt.title('assignment net loss records')
        plt.savefig(self.result_path + '/LossRecord')
        plt.show()

    def plot_reward(self):
        reward_axis = np.linspace(1, len(self.avg_r_rec), len(self.avg_r_rec))
        plt.figure(dpi=400)
        plt.plot(reward_axis, self.avg_r_rec, color='limegreen', label='Average Reward')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_s)
        plt.ylabel('reward', fontsize=self.fontsize_s)
        plt.title('Reward Records', fontsize=self.fontsize_s)
        plt.savefig(self.result_path + '/RewardRecord')
        plt.show()

    def plot_cost(self):
        cost_axis = np.linspace(1, len(self.T_rec), len(self.T_rec))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Episode', fontsize=self.fontsize_l)
        ax1.set_ylabel('Time (Seconds)', color='red', fontsize=self.fontsize_l)
        ax1.plot(cost_axis, self.T_rec, color='red', label='Total Time', linewidth=3, marker='^')
        ax1.tick_params(axis='y', labelcolor='red', labelsize=self.fontsize_s)
        ax1.tick_params(axis='x', labelsize=self.fontsize_s)
        ax1.legend(loc=(0.68, 0.66), fontsize=self.fontsize_s)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Energy (J)', color='blue', fontsize=self.fontsize_l)  # we already handled the x-label with ax1
        ax2.plot(cost_axis, self.E_rec, color='blue', label='Total Energy', linewidth=3, marker='o')
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=self.fontsize_s)
        ax2.legend(loc=(0.68, 0.59), fontsize=self.fontsize_s)

        plt.savefig(self.result_path + '/TotalCostRecord')
        plt.show()

    def plot_time(self):
        time_axis = np.linspace(1, len(self.T_rec), len(self.T_rec))
        plt.figure(dpi=400)
        plt.plot(time_axis, self.T_rec, color='red', label='Overall Time')
        plt.plot(time_axis, self.T_device_rec, color='lightcoral', label='Device Processing Time')
        plt.plot(time_axis, self.T_tran_rec, color='firebrick', label='Transmission Time')
        plt.plot(time_axis, self.T_queue_rec, color='maroon', label='Queueing Time')
        plt.plot(time_axis, self.T_server_rec, color='peru', label='Server Processing Time')
        plt.legend(fontsize=self.fontsize_m)
        plt.xlabel('Episode', fontsize=self.fontsize_l)
        plt.ylabel('Time', fontsize=self.fontsize_l)
        plt.title('Time Consumption Breakdowns')

        plt.savefig(self.result_path + '/TimeBreakdowns')
        plt.show()

    def plot_energy(self):
        energy_axis = np.linspace(1, len(self.T_rec), len(self.T_rec))
        plt.figure(dpi=400)
        plt.plot(energy_axis, self.E_rec, color='blue', label='Overall Energy')
        plt.plot(energy_axis, self.E_device_rec, color='cornflowerblue', label='Device Processing Energy')
        plt.plot(energy_axis, self.E_tran_rec, color='deepskyblue', label='Transmission Energy')
        plt.plot(energy_axis, self.E_server_rec, color='blueviolet', label='Server Processing Energy')
        plt.legend(fontsize=self.fontsize_m)
        plt.xlabel('Episode', fontsize=self.fontsize_l)
        plt.ylabel('Energy', fontsize=self.fontsize_l)
        plt.title('Energy Consumption Breakdowns')

        plt.savefig(self.result_path + '/EnergyBreakdowns')
        plt.show()

    def plot_queue(self):
        queue_length = np.squeeze(np.array(self.queue, dtype=object))
        queue_axis = np.linspace(1, queue_length.shape[0], queue_length.shape[0])
        plt.figure(dpi=400)
        plt.plot(queue_axis, queue_length[:, 0], color='tab:blue', label='server 0')
        plt.plot(queue_axis, queue_length[:, 1], color='tab:orange', label='server 1')
        plt.plot(queue_axis, queue_length[:, 2], color='tab:green', label='server 2')
        plt.legend(fontsize=self.fontsize_m)
        plt.xlabel('Steps', fontsize=self.fontsize_l)
        plt.ylabel('Time', fontsize=self.fontsize_l)
        plt.title('Queue Length Dynamic')

        plt.savefig(self.result_path + '/QueueLength')
        plt.show()

    def plot_hit_min_queue(self):
        hmq_axis = np.linspace(1, len(self.hit_min_queue), len(self.hit_min_queue))
        plt.figure(dpi=400)
        plt.plot(hmq_axis, self.hit_min_queue, color='limegreen')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_s)
        plt.ylabel('Rate', fontsize=self.fontsize_s)
        plt.title('Shortest Queue Hit Rate', fontsize=self.fontsize_s)
        plt.savefig(self.result_path + '/HitMinQueue')
        plt.show()

    def plot_time_subplots(self):
        time_axis = np.linspace(1, len(self.T_rec), len(self.T_rec))
        plt.figure(dpi=400)

        plt.subplot(2, 2, 1)
        plt.plot(time_axis, self.T_device_rec, color='lightcoral', label='Device Processing Time')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_m)
        plt.ylabel('Time', fontsize=self.fontsize_m)
        plt.title('Device Processing Time', fontsize=self.fontsize_m)

        plt.subplot(2, 2, 2)
        plt.plot(time_axis, self.T_tran_rec, color='firebrick', label='Transmission Time')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_m)
        plt.ylabel('Time', fontsize=self.fontsize_m)
        plt.title('Transmission Time', fontsize=self.fontsize_m)

        plt.subplot(2, 2, 3)
        plt.plot(time_axis, self.T_queue_rec, color='maroon', label='Queueing Time')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_m)
        plt.ylabel('Time', fontsize=self.fontsize_m)
        plt.title('Queueing Time', fontsize=self.fontsize_m)

        plt.subplot(2, 2, 4)
        plt.plot(time_axis, self.T_server_rec, color='peru', label='Server Processing Time')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_m)
        plt.ylabel('Time', fontsize=self.fontsize_m)
        plt.title('Server Processing Time', fontsize=self.fontsize_m)

        plt.savefig(self.result_path + '/TimeSubplots')
        plt.show()

    def plot_energy_subplots(self):
        energy_axis = np.linspace(1, len(self.T_rec), len(self.T_rec))
        plt.figure(dpi=400)

        plt.subplot(2, 2, 1)
        plt.plot(energy_axis, self.E_device_rec, color='cornflowerblue', label='Device Processing Energy')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_m)
        plt.ylabel('Energy', fontsize=self.fontsize_m)
        plt.title('Device Processing Energy', fontsize=self.fontsize_m)

        plt.subplot(2, 2, 2)
        plt.plot(energy_axis, self.E_tran_rec, color='deepskyblue', label='Transmission Energy')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_m)
        plt.ylabel('Energy', fontsize=self.fontsize_m)
        plt.title('Transmission Energy', fontsize=self.fontsize_m)

        plt.subplot(2, 2, 3)
        plt.plot(energy_axis, self.E_server_rec, color='blueviolet', label='Server Processing Energy')
        plt.legend(fontsize=self.fontsize_s)
        plt.xlabel('episode', fontsize=self.fontsize_m)
        plt.ylabel('Energy', fontsize=self.fontsize_m)
        plt.title('Server Processing Energy', fontsize=self.fontsize_m)

        plt.savefig(self.result_path + '/EnergySubplots')
        plt.show()
