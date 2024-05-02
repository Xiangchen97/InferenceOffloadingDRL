import numpy as np
import math
from scipy.io import loadmat

DATA_PATH = '/home/lixiangchen/MyWorkspace/InferenceOffloading/results/'
# DATA_PATH = 'Y:/MyWorkspace/InferenceOffloading/results/'


class InfEnv(object):
    def __init__(self, n_device, n_server, task_id):
        super(InfEnv, self).__init__()
        self.gamma_device = 5
        self.gamma_server = 5 / 4
        self.f_device = 0.2e9
        self.f_server = 3e9
        self.kappa_device = 5 / (self.f_device ** 3)
        self.kappa_server = 5 / (self.f_server ** 3)
        self.pi = 1
        self.channel_capacity = 0.01e9
        self.omega = 1  # Time weight
        self.tau = 1    # Energy Weight
        self.n_device = n_device
        self.n_server = n_server
        self.device_energy_price = 20
        self.server_queue = np.zeros(self.n_server)
        self.task_id = task_id
        self.poisson_lambda = 0.003  # We define the time interval as 1s, the average number of task generated from each
        # device is captured by this parameter
        self.T_device = 0
        self.T_tran = 0
        self.T_queue = 0
        self.T_server = 0
        self.E_device = 0
        self.E_tran = 0
        self.E_server = 0
        self.T = 0
        self.E = 0
        self.hit_min_queue = 0
        self.state = 0
        self.reward = 0
        self.task_list = ['base', 'svhn', 'cifar10', 'resnet18', 'resnet34', 'resnet50']
        self.saved_activation_size = {'base': loadmat(DATA_PATH + 'base_model_parameter')['activation_size'],
                                      'svhn': loadmat(DATA_PATH + 'svhn_model_parameter')['activation_size'],
                                      'cifar10': loadmat(DATA_PATH + 'cifar10_model_parameter')['activation_size'],
                                      'resnet18': loadmat(DATA_PATH + 'resnet18_model_parameter')['activation_size'],
                                      'resnet34': loadmat(DATA_PATH + 'resnet34_model_parameter')['activation_size']}
        self.saved_computation_workload = {'base': loadmat(DATA_PATH + 'base_model_parameter')['computation_payload'],
                                           'svhn': loadmat(DATA_PATH + 'svhn_model_parameter')['computation_payload'],
                                           'cifar10': loadmat(DATA_PATH + 'cifar10_model_parameter')[
                                               'computation_payload'],
                                           'resnet18': loadmat(DATA_PATH + 'resnet18_model_parameter')[
                                               'computation_payload'],
                                           'resnet34': loadmat(DATA_PATH + 'resnet34_model_parameter')[
                                               'computation_payload']}
        self.saved_quantization_bitwidth = {
            'base': loadmat(DATA_PATH + 'base_model_parameter')['quantization_bitwidth'],
            'svhn': loadmat(DATA_PATH + 'svhn_model_parameter')['quantization_bitwidth'],
            'cifar10': loadmat(DATA_PATH + 'cifar10_model_parameter')[
                'quantization_bitwidth'],
            'resnet18': loadmat(DATA_PATH + 'resnet18_model_parameter')[
                'quantization_bitwidth'],
            'resnet34': loadmat(DATA_PATH + 'resnet34_model_parameter')[
                'quantization_bitwidth']}
        self.saved_parameter_size = {
            'base': loadmat(DATA_PATH + 'base_model_parameter')['parameter_size'],
            'svhn': loadmat(DATA_PATH + 'svhn_model_parameter')['parameter_size'],
            'cifar10': loadmat(DATA_PATH + 'cifar10_model_parameter')[
                'parameter_size'],
            'resnet18': loadmat(DATA_PATH + 'resnet18_model_parameter')[
                'parameter_size'],
            'resnet34': loadmat(DATA_PATH + 'resnet34_model_parameter')[
                'parameter_size']}

    def env_init(self):
        self.server_queue = np.zeros(self.n_server)

    def env_step(self, offloading_assignment, partition_point):  # This method is called upon each new task is generated
        # generate time interval according to poisson distribution
        task_interval = np.random.exponential(1 / self.poisson_lambda, size=1)
        self.server_queue -= task_interval
        self.server_queue = np.array([item if item >= 0 else 0.0 for item in self.server_queue])
        task_layer_num = self.saved_quantization_bitwidth[self.task_list[self.task_id]].shape[1]

        # generate state information
        MACs_per_sec_device = self.f_device / self.gamma_device
        MACs_per_sec_server = self.f_server / self.gamma_server

        # calculate reward function value
        self.T_device = sum(
            [self.saved_computation_workload[self.task_list[self.task_id]][0][i] * self.gamma_device / self.f_device for i in
             range(partition_point)])
        CommPld_w = sum([self.saved_parameter_size[self.task_list[self.task_id]][0][i] * self.saved_quantization_bitwidth[self.task_list[self.task_id]][0][i] for i in range(partition_point, task_layer_num, 1)])
        CommPld_x = self.saved_quantization_bitwidth[self.task_list[self.task_id]][0][partition_point] * self.saved_activation_size[self.task_list[self.task_id]][0][partition_point]
        self.T_tran = (CommPld_w + CommPld_x) / self.channel_capacity
        self.T_server = sum(
            [self.saved_computation_workload[self.task_list[self.task_id]][0][i] * self.gamma_server / self.f_server for i in
             range(partition_point, task_layer_num, 1)])
        self.T_queue = self.server_queue[offloading_assignment]
        self.hit_min_queue = 1 if offloading_assignment in list(np.where(self.server_queue == self.server_queue.min())[0]) else 0
        # update waiting time in the queue
        self.server_queue[offloading_assignment] += self.T_server
        self.T = self.T_queue + self.T_server + self.T_tran + self.T_device
        self.E_device = self.kappa_device * (self.f_device ** 2) * sum([self.saved_computation_workload[self.task_list[self.task_id]][0][i] for i in range(partition_point)]) * self.gamma_device
        self.E_server = self.kappa_server * (self.f_server ** 2) * sum([self.saved_computation_workload[self.task_list[self.task_id]][0][i] for i in range(partition_point, task_layer_num, 1)]) * self.gamma_server
        self.E_tran = self.T_tran * self.pi
        self.E = self.E_tran + self.E_device * self.device_energy_price + self.E_server
        cost = self.E * self.tau + self.T * self.omega
        # self.reward = -math.log2(cost) / 18
        # self.reward = -(np.log(self.T_queue + 1) / 3.88 - 1.94)
        # self.reward = self.hit_min_queue
        self.reward = -(np.log(self.T_device + 1) / 2.21 - 1.1)-(np.log(self.T_queue + 1) / 3.88 - 1.94)
        self.state = np.concatenate((np.squeeze(self.saved_computation_workload[self.task_list[self.task_id]] / 1e9,
                                                axis=0), np.squeeze(self.saved_activation_size[
                                                                        self.task_list[self.task_id]] / 1e4, axis=0),
                                     np.array([MACs_per_sec_device / 1e7]), np.array([MACs_per_sec_server / 1e9]),
                                     np.array([self.channel_capacity / 1e8]), np.array(self.server_queue)))
        return self.state, self.reward



