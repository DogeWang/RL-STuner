import random
import toolkit.utility_function
import toolkit.sample_materialize
import toolkit.query_parser
import toolkit.query_rewriter
import toolkit.tools
import copy
import numpy as np
import math
import torch
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
storage_sample_number = toolkit.tools.get_storage_sample_number()
error_threshold = toolkit.tools.get_error_threshold()
max_workload_threshold = 30
max_state = 500


BATCH_SIZE = 10
TARGET_REPLACE_ITER = 20
MEMORY_CAPACITY = 100
N_STATES = 0
N_ACTIONS = 0
N_ACTIONS_TRUE = 0
EPSILON = 0.0
LEARNING_RATE = 0.0
DISCOUNT = 0.0


def init(env, epsilon, learning_rate, discount):
    global N_STATES, N_ACTIONS, EPSILON, LEARNING_RATE, DISCOUNT, N_ACTIONS_TRUE
    N_STATES, N_ACTIONS, N_ACTIONS_TRUE = env.state_n, env.action_n, env.action_true
    EPSILON, LEARNING_RATE, DISCOUNT = epsilon, learning_rate, discount


class Environment:
    def __init__(self, workload, abstract_sample_list):
        self.workload = workload
        self.state = list(np.zeros(max_state, int))
        self.state_n = len(self.state)
        self.action_n = len(self.state)
        self.action_true = len(abstract_sample_list)
        self.abstract_samples = abstract_sample_list
        self.active_samples = set()
        self.utility = 0
        self.init_workload = workload
        self.init_abstract_samples = abstract_sample_list

    def extend(self, query_list, sample_list):
        self.workload += query_list
        self.state += list(np.zeros(len(sample_list), int))
        self.abstract_samples += sample_list
        self.state_n = len(self.state)
        self.action_n = len(self.state)

    def step(self, action, exact=True):
        if self.state[action] == 0:
            self.state[action] = 1
            self.active_samples.add(self.abstract_samples[action])
            action_utility, sample_query, query_sample = toolkit.utility_function.calculate_action_utility(self.workload, self.active_samples, exact)
            reward = action_utility - self.utility
            self.utility = action_utility
        else:
            self.state[action] = 0
            for sample in self.active_samples:
                if sample.sample_name == self.abstract_samples[action].sample_name:
                    self.active_samples.remove(sample)
                    break
            action_utility, sample_query, query_sample = toolkit.utility_function.calculate_action_utility(self.workload, self.active_samples)
            reward = action_utility - self.utility
            self.utility = action_utility

        return self.state, reward, sample_query, query_sample

    def reset(self):
        self.__init__(self.init_workload, self.init_abstract_samples)
        return self.state


class Net(torch.nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_STATES, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, N_ACTIONS)
        )

    def forward(self, x):
        x = x.to(device)
        actions_value = self.net(x)
        return actions_value


class DQN(torch.nn.Module):
    def __init__(self,):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.loss_func = torch.nn.MSELoss()

    def epsilon_action(self, state):
        temp_state = state[:N_ACTIONS_TRUE]
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() <= EPSILON:
            actions_value = self.eval_net.forward(state)
            temp = torch.topk(actions_value, max_state, 1)[1][0].cpu()
            for i in range(len(temp)):
                action = temp[i].data.numpy()
                if state[0][action] == 0 and action < N_ACTIONS_TRUE:
                    break
        else:
            temp_list = []
            for i in range(len(temp_state)):
                if temp_state[i] == 0:
                    temp_list.append(i)
            action = random.choice(temp_list)
        return action

    def greedy_action(self, state):
        state = torch.unsqueeze(torch.tensor(state), 0)
        actions_value = self.eval_net.forward(state)
        action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_action = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_reward = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_next_state = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + DISCOUNT * q_next.max(1)[0].view(q_next.max(1)[0].numel(), 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def deep_q_nn(workload, sample_list, epochs=1000, epsilon=0.1, learning_rate=0.3, discount=0.5, exact=True, q_net=None):
    global EPSILON
    start = time.time()

    print('||==========================================================================================================================================================||')
    print('||                                                                                                                                                          ||')
    print('||                                                                           Training DQN                                                                   ||')
    print('||                                                                                                                                                          ||')
    print('||==========================================================================================================================================================||')

    env = Environment(workload, sample_list)
    init(env, epsilon, learning_rate, discount)
    stats_utility, stats_reward, materialized_samples_list = np.zeros(epochs), np.zeros(epochs), []
    dqn = DQN().to(device)
    if q_net:
        dqn = q_net
        epochs = 500
    else:
        epochs = 1000

    epochs_num = epochs / 10
    for i in range(epochs):
        EPSILON = 0.09 * math.ceil((i + 1) / epochs_num)
        if i == epochs - 1:
            EPSILON = 1
        state = env.reset()
        sample_size, step = 0, 0

        while sample_size < storage_sample_number and step < 20:
            action = dqn.epsilon_action(state)
            next_state, reward, sample_query, query_sample = env.step(action, exact)
            dqn.store_transition(state, action, reward, next_state)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            sample_size = len(env.active_samples)
            step += 1
            state = next_state
        stats_utility[i] += env.utility
        materialized_samples_list.append(env.active_samples)
        active_sample_set = env.active_samples
    # active_sample_set = materialized_samples_list[stats_utility.argmax()]
    tc_dqn = time.time() - start
    print('The time cost of training DQN (s): ', tc_dqn)
    print('The active samples of DQN is: ', end='')
    for sample in active_sample_set:
        print(sample.sample_name, end=' ')
    print()
    return active_sample_set


# lazy sample tuning strategy according to nearest_n Q_t
def lazy_sample_tuning(test_workload, active_sample_set, method, exact=True, train_workload=[]):
    global storage_sample_number
    storage_sample_number = toolkit.tools.get_storage_sample_number()
    train_workload, test_workload = copy.deepcopy(train_workload), copy.deepcopy(test_workload)
    toolkit.sample_materialize.construct_workload_2_samples(test_workload, train_workload)
    sample_list = copy.deepcopy(toolkit.sample_materialize.sample_list)
    unusable_query_list = train_workload + test_workload
    toolkit.utility_function.workload_length = len(unusable_query_list)

    # nearest_n = 0
    # if len(test_workload) > max_workload_threshold:
    #     nearest_n = 3

    if len(active_sample_set) > storage_sample_number:
        active_sample_set = deep_q_nn(unusable_query_list, sample_list, 1000, 0.1, 0.3, 0.5, exact)
        print('The active samples after tuning by ' + method + ' are: ', end='')
        for sample in active_sample_set:
            print(sample.sample_name, end=' ')
        print()
        return active_sample_set

    _, sample_query_1, query_sample_1 = toolkit.utility_function.calculate_action_utility(train_workload, active_sample_set, exact)
    if exact:
        _, sample_query_2, query_sample_2 = toolkit.utility_function.calculate_action_utility(test_workload, active_sample_set, True)
    else:
        sample_query_2, query_sample_2 = toolkit.query_rewriter.map_workload_sample_execut(test_workload, active_sample_set)
    sample_query_dict = {}
    for sample_1 in sample_query_1.keys():
        for sample_2 in sample_query_2.keys():
            if sample_1.sample_name == sample_2.sample_name:
                sample_query_dict[sample_1] = sample_query_1[sample_1] + sample_query_2[sample_2]
                break

    for sample_1 in sample_list[::-1]:
        for sample_2 in active_sample_set:
            if sample_1.sample_name == sample_2.sample_name:
                sample_list.remove(sample_1)
                break

    sample_utility_dict, query_re_dict = toolkit.utility_function.calculate_sample_utility(sample_query_dict, exact)
    init_utility = -len(unusable_query_list)
    for sample_1 in sample_query_dict.keys():
        sample_utility_dict[sample_1] = init_utility + len(sample_query_dict[sample_1]) + sample_utility_dict[sample_1]

    if (len(active_sample_set) + len(sample_list)) <= storage_sample_number:
        active_sample_set.update(sample_list)
    elif len(active_sample_set) >= storage_sample_number:
        min_value, min_sample = 1000000, None
        for key, value in sample_utility_dict.items():
            if value < min_value:
                min_sample = key
                min_value = value

        for sample_1 in list(sample_query_dict.keys())[::-1]:
            print('The number of query on', sample_1.sample_name, 'is', len(sample_query_dict[sample_1]))
            if len(sample_query_dict[sample_1]) <= 1:
                for sample_2 in active_sample_set:
                    if sample_2.sample_name == sample_1.sample_name:
                        active_sample_set.remove(sample_2)
                        active_sample_set.add(sample_2)
                        break
                del sample_query_dict[sample_1]

        for sample_1 in active_sample_set:
            if sample_1.sample_name == min_sample.sample_name:
                for sample_2 in sample_utility_dict.keys():
                    if sample_2.sample_name == min_sample.sample_name:
                        del sample_utility_dict[sample_2]
                        sample_list.append(sample_1)
                        break
                active_sample_set.remove(sample_1)
                break

        if (len(active_sample_set) + len(sample_list)) <= storage_sample_number:
            active_sample_set.update(sample_list)
        else:
            storage_sample_number -= len(active_sample_set)

            for sample_1, queries_1 in sample_query_dict.items():
                for sample_2 in active_sample_set:
                    if sample_1.sample_name == sample_2.sample_name:
                        for query in queries_1:
                            if query in unusable_query_list:
                                unusable_query_list.remove(query)
            for query, relative_error in query_re_dict.items():
                if relative_error > error_threshold and query not in unusable_query_list:
                    unusable_query_list.append(query)

            if not unusable_query_list:
                print('All query have a good result!!!')
                if len(active_sample_set) > storage_sample_number:
                    active_sample_set.update(random.sample(sample_list, storage_sample_number))
                else:
                    active_sample_set.update(sample_list)

                print('The active samples after tuning by ' + method + ' are: ', end='')
                for sample in active_sample_set:
                    print(sample.sample_name, end=' ')
                print()
                storage_sample_number = toolkit.tools.get_storage_sample_number()
                return active_sample_set

            new_active_samples = deep_q_nn(unusable_query_list, sample_list, 1000, 0.1, 0.3, 0.5, exact)

            active_sample_set.update(new_active_samples)
            storage_sample_number = toolkit.tools.get_storage_sample_number()
    else:
        storage_sample_number -= len(active_sample_set)

        for sample_1, queries_1 in sample_query_dict.items():
            for sample_2 in active_sample_set:
                if sample_1.sample_name == sample_2.sample_name:
                    for query in queries_1:
                        if query in unusable_query_list:
                            unusable_query_list.remove(query)
        for query, relative_error in query_re_dict.items():
            if relative_error > error_threshold and query not in unusable_query_list:
                unusable_query_list.append(query)
        if not unusable_query_list:
            print('All query have a good result!!!')
            if len(sample_list) > storage_sample_number:
                active_sample_set.update(random.sample(sample_list, storage_sample_number))
            else:
                active_sample_set.update(sample_list)

            print('The active samples after tuning by ' + method + ' are: ', end='')
            for sample in active_sample_set:
                print(sample.sample_name, end=' ')
            print()
            storage_sample_number = toolkit.tools.get_storage_sample_number()
            return active_sample_set
        new_active_samples = deep_q_nn(unusable_query_list, sample_list, 1000, 0.1, 0.3, 0.5, exact)

        active_sample_set.update(new_active_samples)
        storage_sample_number = toolkit.tools.get_storage_sample_number()

    print('The active samples after tuning by ' + method + ' are: ', end='')
    for sample in active_sample_set:
        print(sample.sample_name, end=' ')
    print()
    storage_sample_number = toolkit.tools.get_storage_sample_number()
    return active_sample_set
