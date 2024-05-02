from InferenceOffloadingEnv import InfEnv
from DQN_Model import DQN
from scipy.io import loadmat
from ResultAnalysis import TrainAnalysis
import pickle
import os
from datetime import datetime

RESULT_PATH = '/home/lixiangchen/MyWorkspace/InferenceOffloadingDRL/Results/'
time_start = datetime.now()
time_start_formatted = time_start.strftime("%Y-%m-%d %H_%M_%S")
os.mkdir(RESULT_PATH + time_start_formatted + '/')

EPSILON_TRAIN = 0.9
MEMERY_CAPACITY = 100  # Memory Length
TASK_ID = 2
EPISODE = 1000  # Training Episodes
STEP_NUM = 25  # Number of steps per episode
SERVER_NUM = 3
LR = 0.001
BATCH_SIZE = 32
TASK_LIST = ['base', 'svhn', 'cifar10', 'resnet18', 'resnet34', 'resnet50']
DATA_PATH = '/home/lixiangchen/MyWorkspace/InferenceOffloadingDRL/Data/'

# Load model parameters
model_parameter_path = DATA_PATH + TASK_LIST[TASK_ID] + '_model_parameter.mat'
dict_model = loadmat(model_parameter_path)
model_layer_num = dict_model['quantization_bitwidth'].shape[1]

# Preparing model and Initializing Environment
dqn = DQN(n_states_p=model_layer_num * 2 + 6, n_states_a=model_layer_num * 2 + 6, n_actions_p=model_layer_num,
          n_actions_a=SERVER_NUM, memory_capacity=MEMERY_CAPACITY, init_lr=LR, batch_size=BATCH_SIZE)
env = InfEnv(n_device=5, n_server=SERVER_NUM, task_id=TASK_ID)
ta = TrainAnalysis()
with open(RESULT_PATH + time_start_formatted + '/EnvSettings.pkl', 'wb') as file:
    pickle.dump(env, file)
with open(RESULT_PATH + time_start_formatted + '/ModelSettings.pkl', 'wb') as file:
    pickle.dump(dqn, file)

# Start Training
print('Training DRL model with %d episodes ... ' % EPISODE)
for i_episode in range(EPISODE):
    # parameter initialization
    avg_r = 0
    loss_p = 0
    loss_a = 0
    T = 0
    E = 0
    T_device = 0
    T_tran = 0
    T_server = 0
    T_queue = 0
    E_device = 0
    E_tran = 0
    E_server = 0
    hit_min_queue = 0
    state, reward = env.env_step(offloading_assignment=0, partition_point=1)
    # adjustable epsilon and learning rate
    epsilon = EPSILON_TRAIN + i_episode * (1 - EPSILON_TRAIN) / (0.8 * EPISODE) if EPSILON_TRAIN + i_episode * (
                1 - EPSILON_TRAIN) / (0.8 * EPISODE) <= 1 else 1
    # Interacting and Training Steps
    for i_step in range(STEP_NUM):
        train_steps = 0
        action_p, action_a = dqn.choose_action(x=state, EPSILON=epsilon)
        state_, reward = env.env_step(offloading_assignment=action_a, partition_point=action_p)
        # Record the system metrics
        T += env.T
        E += env.E
        T_device += env.T_device
        T_tran += env.T_tran
        T_server += env.T_server
        T_queue += env.T_queue
        E_device += env.E_device
        E_tran += env.E_tran
        E_server += env.E_server
        hit_min_queue += env.hit_min_queue
        # Save to memory
        dqn.store_transition(state, action_p, action_a, reward, state_)
        avg_r += reward
        if dqn.memory_counter > MEMERY_CAPACITY:
            # Start updating model when memory is full
            train_steps += 1
            dqn.learn()
            loss_p += dqn.loss_p
            loss_a += dqn.loss_a
            if i_step == STEP_NUM - 1:
                ta.avg_r_rec.append(avg_r / STEP_NUM)
                ta.loss_p_rec.append(loss_p / train_steps)
                ta.loss_a_rec.append(loss_a / train_steps)
                # save system metrics
                ta.T_rec.append(T / STEP_NUM)
                ta.E_rec.append(E / STEP_NUM)
                ta.T_device_rec.append(T_device / STEP_NUM)
                ta.T_tran_rec.append(T_tran / STEP_NUM)
                ta.T_server_rec.append(T_server / STEP_NUM)
                ta.T_queue_rec.append(T_queue / STEP_NUM)
                ta.E_device_rec.append(E_device / STEP_NUM)
                ta.E_tran_rec.append(E_tran / STEP_NUM)
                ta.E_server_rec.append(E_server / STEP_NUM)
                ta.hit_min_queue.append(hit_min_queue / STEP_NUM)
                print('Episode %i, Step %i, p network loss: %f, a network loss: %f, average reward: %f' % (
                    i_episode, i_step, loss_p / train_steps, loss_a / train_steps, avg_r / STEP_NUM))
            dqn.adjust_learning_rate()
        state = state_
        ta.queue.append(env.server_queue)
print('Training Finished!')
print('Analyzing and Plotting Results ...')
ta.plot_loss()
ta.plot_reward()
ta.plot_cost()
ta.plot_time()
ta.plot_energy()
ta.plot_queue()
ta.plot_hit_min_queue()
ta.plot_time_subplots()
ta.plot_energy_subplots()


