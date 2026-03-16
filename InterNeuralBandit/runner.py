import os
import random
import csv
import time
import pickle

import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics.pairwise import rbf_kernel

from base.action import Action
from base.reward import Reward

from InterNeuralBandit.config import parse_args
from InterNeuralBandit.environment import LinearPayoffEnvironment
from InterNeuralBandit.environment import NonLinearCosPayoffEnvironment
from InterNeuralBandit.environment import NonLinearQuadPayoffEnvironment
from InterNeuralBandit.environment import NonLinearSquarePayoffEnvironment
from InterNeuralBandit.environment import NonLinearLogPayoffEnvironment
from InterNeuralBandit.environment import NonLinearExpPayoffEnvironment
from InterNeuralBandit.data_handling import FileData
from InterNeuralBandit.data_handling import HistoryData
from InterNeuralBandit.algorithms.inter_neural_ucb import InterNeuralUCB
from InterNeuralBandit.algorithms.inter_neural_ts import InterNeuralTS
from InterNeuralBandit.algorithms.neural_linear import NeuralLinear
from InterNeuralBandit.algorithms.ucb import LinearPayoffUCB
from InterNeuralBandit.algorithms.ucb import LinearPayoffTS
from InterNeuralBandit.algorithms.ucb import LinearPayoffUCBWithRP
from InterNeuralBandit.algorithms.ucb import KernelUCB
from InterNeuralBandit.algorithms.neural_ucb import *
from InterNeuralBandit.algorithms.exp3 import Exp3Bandit


def generate_synthetic_data(args):
    """create synthetic data"""

    if not os.path.exists('./data/synthetic_exp_dim%d.csv' % args.dim):
        for i in range(args.num_of_data):
            context = np.random.randn(args.dim, 1)
            context /= np.linalg.norm(context, axis=0)
            context = context.T
            env = NonLinearExpPayoffEnvironment(context, args.lin_para_seed)
            stochastic_reward = env.get_stochastic_reward(Action(0)).total_reward
            data = np.concatenate(([stochastic_reward], context[0, :]), axis=0).reshape((1, args.dim + 1))
            with open('./data/synthetic_exp_dim%d.csv' % args.dim, mode='ab') as file:
                np.savetxt(file, data, fmt='%f', delimiter=',')


    if not os.path.exists('./data/synthetic_cos_dim%d.csv' % args.dim):
        for i in range(args.num_of_data):
            context = np.random.randn(args.dim, 1)
            context /= np.linalg.norm(context, axis=0)
            context = context.T
            env = NonLinearCosPayoffEnvironment(context, args.lin_para_seed)
            stochastic_reward = env.get_stochastic_reward(Action(0)).total_reward
            data = np.concatenate(([stochastic_reward], context[0, :]), axis=0).reshape((1, args.dim + 1))
            with open('./data/synthetic_cos_dim%d.csv' % args.dim, mode='ab') as file:
                np.savetxt(file, data, fmt='%f', delimiter=',')

    if not os.path.exists('./data/synthetic_square_dim%d.csv' % args.dim):
        for i in range(args.num_of_data):
            context = np.random.randn(args.dim, 1)
            context /= np.linalg.norm(context, axis=0)
            context = context.T
            env = NonLinearSquarePayoffEnvironment(context, args.lin_para_seed)
            stochastic_reward = env.get_stochastic_reward(Action(0)).rewards[0]
            data = np.concatenate(([stochastic_reward], context[0, :]), axis=0).reshape((1, args.dim + 1))
            with open('./data/synthetic_square_dim%d.csv' % args.dim, mode='ab') as file:
                np.savetxt(file, data, fmt='%f', delimiter=',')

    # followings are currently not used in experiments
    '''
    if not os.path.exists('./data/synthetic_quad_dim%d.csv' % args.dim):
        for i in range(args.num_of_data):
            context = np.random.randn(args.dim, 1)
            context /= np.linalg.norm(context, axis=0)
            context = context.T
            cos_env = NonLinearQuadPayoffEnvironment(context, args.matrix_seed)
            stochastic_reward = cos_env.get_stochastic_reward(Action(0)).total_reward
            data = np.concatenate(([stochastic_reward], context[0, :]), axis=0).reshape((1, args.dim + 1))
            with open('./data/synthetic_quad_dim%d.csv' % args.dim, mode='ab') as file:
                np.savetxt(file, data, fmt='%f', delimiter=',')
                
    if not os.path.exists('./data/synthetic_linear_dim%d.csv' % args.dim):
        for i in range(args.num_of_data):
            context = np.random.randn(args.dim, 1)
            context /= np.linalg.norm(context, axis=0)
            context = context.T
            env = LinearPayoffEnvironment(context, args.lin_para_seed)
            stochastic_reward = env.get_stochastic_reward(Action(0)).total_reward
            data = np.concatenate(([stochastic_reward], context[0, :]), axis=0).reshape((1, args.dim + 1))
            with open('./data/synthetic_linear_dim%d.csv' % args.dim, mode='ab') as file:
                np.savetxt(file, data, fmt='%f', delimiter=',')

    if not os.path.exists('./data/synthetic_log_dim%d.csv' % args.dim):
        for i in range(args.num_of_data):
            context = np.random.randn(args.dim, 1)
            context /= np.linalg.norm(context, axis=0)
            context = context.T
            env = NonLinearLogPayoffEnvironment(context, args.lin_para_seed)
            stochastic_reward = env.get_stochastic_reward(Action(0)).total_reward
            data = np.concatenate(([stochastic_reward], context[0, :]), axis=0).reshape((1, args.dim + 1))
            with open('./data/synthetic_log_dim%d.csv' % args.dim, mode='ab') as file:
                np.savetxt(file, data, fmt='%f', delimiter=',')    
    '''


def inter_neural_ucb_runner(args, data_file, sample_seq, real_data=True, use_cnn=False):
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)
    historical_data = HistoryData(args.historical_data_size, data_file, feature_cols=args.feature_cols,
                                  reward_col=args.reward_col)
    inter_neural_ucb = InterNeuralUCB(historical_data, args, use_cnn)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_InterNeural_' + 'arm=' + str(args.num_of_arms) + '_p=' + str(args.latent_shape) + '_D=' + str(args.historical_data_size) + '_M=' + str(args.online_batch_size) + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start InterNeural...')
    print('=============================')

    step = 0
    cumu_regret = 0.0

    if args.pre_training_on:
        # start pre-training
        print('=4 Start Pre-training ...')
        print('=============================')
        for iteration in range(int(args.max_num_of_iter / 100)):
            inter_neural_ucb.init_lin_ucb()
            for _ in range(args.online_max_step):
                # online training
                # sample from historical data
                sampled_data = inter_neural_ucb.historical_data.sample(num=args.num_of_arms)
                contexts = sampled_data[:, args.feature_cols]
                rewards = sampled_data[:, args.reward_col]
                action = inter_neural_ucb.online_training(contexts, rewards)

            # offline training
            inter_neural_ucb.offline_training()
    else:
        inter_neural_ucb.pre_training()

    print('=5 Start Iterations ...')
    print('=============================')
    for iteration in range(args.max_num_of_iter):
        inter_neural_ucb.init_lin_ucb()
        # sample batched data M to alleviate cold start
        for _ in range(args.online_batch_size):
            sampled_data = inter_neural_ucb.historical_data.sample(num=args.num_of_arms)
            contexts = sampled_data[:, args.feature_cols]
            rewards = sampled_data[:, args.reward_col]
            action = inter_neural_ucb.online_training(contexts, rewards)
            # print(hy_neural_bandit.lin_ucb.linpara)

        # online training (LinUCB)
        #print('online')
        for _ in range(args.online_max_step):
            step += 1
            # get online data (simulated)
            sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
            contexts = sampled_data[:, args.feature_cols]
            rewards = sampled_data[:, args.reward_col]
            start_time = time.time()
            action = inter_neural_ucb.online_training(contexts, rewards)
            end_time = time.time()
            # append sampled data to offline historical data
            inter_neural_ucb.historical_data.append(np.array([sampled_data[action.actions[0]]]))
            #expected_reward = env.get_expected_reward(action)
            #optimal_reward = env.get_optimal_reward()
            regret = float(np.max(rewards) - rewards[action.actions, 0])
            cumu_regret += regret

            csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
            with open(dir, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(csv_row)

        # offline training
        inter_neural_ucb.offline_training()


def inter_neural_ts_runner(args, data_file, sample_seq, real_data=True, use_cnn=False):
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)
    historical_data = HistoryData(args.historical_data_size, data_file, feature_cols=args.feature_cols,
                                  reward_col=args.reward_col)
    inter_neural_ts = InterNeuralTS(historical_data, args, use_cnn)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_InterNeural_' + 'arm=' + str(args.num_of_arms) + '_p=' + str(args.latent_shape) + '_D=' + str(args.historical_data_size) + '_M=' + str(args.online_batch_size) + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start InterNeural...')
    print('=============================')

    step = 0
    cumu_regret = 0.0

    if args.pre_training_on:
        # start pre-training
        print('=4 Start Pre-training ...')
        print('=============================')
        for iteration in range(int(args.max_num_of_iter / 100)):
            inter_neural_ts.init_lin_ts()
            for _ in range(args.online_max_step):
                # online training
                # sample from historical data
                sampled_data = inter_neural_ts.historical_data.sample(num=args.num_of_arms)
                contexts = sampled_data[:, args.feature_cols]
                rewards = sampled_data[:, args.reward_col]
                action = inter_neural_ts.online_training(contexts, rewards)

            # offline training
            inter_neural_ts.offline_training()
    else:
        inter_neural_ts.pre_training()

    print('=5 Start Iterations ...')
    print('=============================')
    for iteration in range(args.max_num_of_iter):
        inter_neural_ts.init_lin_ts()
        # sample batched data M to alleviate cold start
        for _ in range(args.online_batch_size):
            sampled_data = inter_neural_ts.historical_data.sample(num=args.num_of_arms)
            contexts = sampled_data[:, args.feature_cols]
            rewards = sampled_data[:, args.reward_col]
            action = inter_neural_ts.online_training(contexts, rewards)
            # print(hy_neural_bandit.lin_ucb.linpara)

        # online training (LinUCB)
        #print('online')
        for _ in range(args.online_max_step):
            step += 1
            # get online data (simulated)
            sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
            contexts = sampled_data[:, args.feature_cols]
            rewards = sampled_data[:, args.reward_col]
            start_time = time.time()
            action = inter_neural_ts.online_training(contexts, rewards)
            end_time = time.time()
            # append sampled data to offline historical data
            inter_neural_ts.historical_data.append(np.array([sampled_data[action.actions[0]]]))
            #expected_reward = env.get_expected_reward(action)
            #optimal_reward = env.get_optimal_reward()
            regret = float(np.max(rewards) - rewards[action.actions, 0])
            cumu_regret += regret

            csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
            with open(dir, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(csv_row)

        # offline training
        inter_neural_ts.offline_training()


def lin_ucb_runner(args, data_file, sample_seq, dim_cut = 200):
    if dim_cut > args.dim:
        dim_cut = args.dim
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_' + 'linUCB' + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start LinUCB iterations ...')
    print('=============================')

    step = 0
    cumu_regret = 0.0
    lin_ucb = LinearPayoffUCB(np.array([[i for i in range(dim_cut)]]))

    for _ in range(args.total_run):
        step += 1
        sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
        reward_col = args.reward_col
        feature_cols = np.array([i for i in range(1, dim_cut + 1)])
        contexts = sampled_data[:, feature_cols]
        rewards = sampled_data[:, reward_col]
        lin_ucb.set_context(contexts)
        start_time = time.time()
        action = lin_ucb.pick_action(observation=None)
        lin_ucb.update_observation(action, Reward(rewards=rewards[action.actions]), observation=None)
        end_time = time.time()

        regret = float(np.max(rewards) - rewards[action.actions, 0])
        cumu_regret += regret

        csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
        with open(dir, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(csv_row)


def lin_ts_runner(args, data_file, sample_seq, dim_cut = 200):
    if dim_cut > args.dim:
        dim_cut = args.dim
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_' + 'LinTS' + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start LinTS iterations ...')
    print('=============================')

    step = 0
    cumu_regret = 0.0
    lin_ts = LinearPayoffTS(np.array([[i for i in range(dim_cut)]]))

    for _ in range(args.total_run):
        step += 1
        sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
        reward_col = args.reward_col
        feature_cols = np.array([i for i in range(1, dim_cut + 1)])
        contexts = sampled_data[:, feature_cols]
        rewards = sampled_data[:, reward_col]
        lin_ts.set_context(contexts)
        start_time = time.time()
        action = lin_ts.pick_action(observation=None)
        lin_ts.update_observation(action, Reward(rewards=rewards[action.actions]), observation=None)
        end_time = time.time()

        regret = float(np.max(rewards) - rewards[action.actions, 0])
        cumu_regret += regret

        csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
        with open(dir, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(csv_row)


def rp_runner(args, data_file, sample_seq):
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_' + 'RP' + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start RP iterations ...')
    print('=============================')

    step = 0
    cumu_regret = 0.0
    lin_ucb_RP = LinearPayoffUCBWithRP(np.array([[i for i in range(args.dim)]]), var=0.3, rp_dim =args.latent_shape)

    for _ in range(args.total_run):
        step += 1
        sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
        contexts = sampled_data[:, args.feature_cols]
        rewards = sampled_data[:, args.reward_col]
        lin_ucb_RP.set_context(contexts)
        start_time = time.time()
        action = lin_ucb_RP.pick_action(observation=None)
        lin_ucb_RP.update_observation(action, Reward(rewards=rewards[action.actions]), observation=None)
        end_time = time.time()

        regret = float(np.max(rewards) - rewards[action.actions, 0])
        cumu_regret += regret

        csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
        with open(dir, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(csv_row)


def e_greedy_runner(args, data_file, sample_seq, epsilon=0.05):
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)
    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')
    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_e-greedy_' + 'arm=' + str(args.num_of_arms) + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start e-greedy iterations ...')
    print('=============================')

    step = 0
    cumu_regret = 0.0
    avg_arm_reward = np.zeros(args.num_of_arms)
    action_counter = np.zeros(args.num_of_arms)
    for _ in range(args.total_run):
        step += 1
        sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
        contexts = sampled_data[:, args.feature_cols]
        rewards = sampled_data[:, args.reward_col]

        # e-greedy exploration
        start_time = time.time()
        prob = random.random()
        idx = np.argsort(avg_arm_reward)[-1] if prob > epsilon else random.randint(0, args.num_of_arms - 1)
        end_time = time.time()

        # update estimated average reward
        avg_arm_reward[idx] = (avg_arm_reward[idx] * action_counter[idx] + rewards[idx]) / (action_counter[idx] + 1)
        action_counter[idx] += 1

        action = Action(actions=idx)

        regret = float(np.max(rewards) - rewards[action.actions, 0])
        cumu_regret += regret

        csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
        with open(dir, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(csv_row)


def neural_linear_runner(args, data_file, sample_seq):
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)
    historical_data = HistoryData(args.historical_data_size, data_file, feature_cols=args.feature_cols,
                                  reward_col=args.reward_col)
    neural_linear = NeuralLinear(historical_data, args)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_neural-linear_' + 'p=' + str(args.latent_shape) + '_D=' + str(args.historical_data_size) + '_M=' + str(args.online_batch_size) + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start Neural Linear...')
    print('=============================')

    step = 0
    cumu_regret = 0.0
    # start pre-training
    print('=4 Start Pre-training ...')
    print('=============================')

    if args.pre_training_on:
        for iteration in range(int(args.max_num_of_iter / 100)):
            neural_linear.init_lin_ucb()
            for _ in range(args.online_max_step):
                # online training
                # sample from historical data
                sampled_data = neural_linear.historical_data.sample(num=args.num_of_arms)
                contexts = sampled_data[:, args.feature_cols]
                rewards = sampled_data[:, args.reward_col]
                action = neural_linear.linear_regression(contexts, rewards)

            # offline training
            neural_linear.update_dnn()
    else:
        neural_linear.pre_training()

    print('=5 Start Iterations ...')
    print('=============================')
    for iteration in range(args.max_num_of_iter):
        if step == args.total_run:
            break
        neural_linear.init_lin_ucb()
        # sample batched data M to alleviate cold start
        for _ in range(args.online_batch_size):
            sampled_data = neural_linear.historical_data.sample(num=args.num_of_arms)
            contexts = sampled_data[:, args.feature_cols]
            rewards = sampled_data[:, args.reward_col]
            action = neural_linear.linear_regression(contexts, rewards)
            # print(hy_neural_bandit.lin_ucb.linpara)

        # online training (LinUCB)
        #print('online')
        for _ in range(args.online_max_step):
            step += 1
            # get online data (simulated)
            sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
            contexts = sampled_data[:, args.feature_cols]
            rewards = sampled_data[:, args.reward_col]
            start_time = time.time()
            action = neural_linear.linear_regression(contexts, rewards)
            end_time = time.time()
            # append sampled data to offline historical data
            neural_linear.historical_data.append(np.array([sampled_data[action.actions[0]]]))

            regret = float(np.max(rewards) - rewards[action.actions, 0])
            cumu_regret += regret

            csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
            with open(dir, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(csv_row)

        # offline training
        neural_linear.update_dnn()


def neural_ucb_runner(args, data_file, sample_seq, dim_cut = 200):
    # bandit settings
    T = args.total_run
    n_arms = args.num_of_arms
    n_features = dim_cut
    noise_std = 0.1
    confidence_scaling_factor = noise_std
    n_sim = 1
    feature_cols = args.feature_cols
    reward_col = args.reward_col
    feature_cols = np.array([i for i in range(1, dim_cut + 1)])
    sample_cols = np.array([i for i in range(0, dim_cut + 1)])

    # network settings
    p = 0.2 # drop rates
    hidden_size = 100
    epochs = 100
    train_every = 10
    confidence_scaling_factor = 1.0
    use_cuda = False

    dataset = FileData(data_file, feature_cols=feature_cols, reward_col=reward_col)
    sampled_data = dataset.sample_by_index(idx=sample_seq)[:,:, sample_cols].reshape(T * n_arms, n_features + 1)
    #for _ in range(9):
    #    sampled_data = np.concatenate((sampled_data, dataset.sample(num=int(T * n_arms / 10))), axis=0)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_neuralUCB_' + 'arm=' + str(args.num_of_arms) + '_dim_' + str(dim_cut) + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start NeuralUCB...')
    print('=============================')

    contexts = sampled_data[:, feature_cols].reshape((T, n_arms, n_features))
    #print(contexts.shape)
    rewards = sampled_data[:, args.reward_col].reshape((T, n_arms))
    bandit = ContextualBandit(T, n_arms, n_features, contexts, rewards)
    model = NeuralUCB(bandit,
                      hidden_size=hidden_size,
                      reg_factor=1.0,
                      delta=0.1,
                      confidence_scaling_factor=confidence_scaling_factor,
                      training_window=100,
                      p=p,
                      learning_rate=args.offline_lr,
                      epochs=epochs,
                      train_every=train_every,
                      use_cuda=use_cuda,
                      log_output_dir=dir
                      )
    model.run()


def kernel_ucb_runner(args, data_file, sample_seq):
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)

    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')

    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_' + 'kernel' + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start KernelUCB iterations ...')
    print('=============================')

    step = 0
    cumu_regret = 0.0
    kernel_ucb = KernelUCB(np.array([[i for i in range(args.dim)]]), args.num_of_arms, gamma=0.01, eta=0.1, kernel=rbf_kernel)

    for _ in range(args.total_run):
        step += 1
        sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
        contexts = sampled_data[:, args.feature_cols]
        #print(contexts[0])
        rewards = sampled_data[:, args.reward_col]
        kernel_ucb.set_context(contexts)
        start_time = time.time()
        if step > 1:
            kernel_ucb.evaluate_arms()
        action = kernel_ucb.pick_action(observation=None)
        kernel_ucb.update_kernel_matrix(action, Reward(rewards=rewards[action.actions]))
        end_time = time.time()

        regret = float(np.max(rewards) - rewards[action.actions, 0])
        cumu_regret += regret

        csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
        with open(dir, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(csv_row)


def exp3_runner(args, data_file, sample_seq):
    dataset = FileData(data_file, feature_cols=args.feature_cols, reward_col=args.reward_col)
    exp3 = Exp3Bandit(gamma=0.1, arm_nums=args.num_of_arms)
    print('=============================')
    print('=1 Dataset {} is right ...'.format(args.reward_model))
    print('=============================')
    # log file preparation
    header = ['step', 'regret (separate)', 'regret (cumulative)', 'time']
    dir = args.save_log_dir + args.reward_model + '_exp3_' + 'arm=' + str(
        args.num_of_arms) + '_' + args.time + '.csv'
    with open(dir, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    print('=2 Log file created successfully ...')
    print('=============================')
    print('=3 Start EXP3 iterations ...')
    print('=============================')

    step = 0
    cumu_regret = 0.0
    for _ in range(args.total_run):
        step += 1
        sampled_data = dataset.sample_by_index(idx=sample_seq[step - 1])
        contexts = sampled_data[:, args.feature_cols]
        rewards = sampled_data[:, args.reward_col]

        start_time = time.time()
        action = exp3.pick_action(observation=None)
        #expected_reward = env.get_expected_reward(action)
        exp3.update_observation(action, Reward(rewards=rewards[action.actions]))
        end_time = time.time()

        #optimal_reward = env.get_optimal_reward()
        #regret = optimal_reward - expected_reward
        #cumu_regret += regret
        regret = float(np.max(rewards) - rewards[action.actions, 0])
        cumu_regret += regret

        csv_row = [str(step), str(regret), str(cumu_regret), str(end_time - start_time)]
        with open(dir, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(csv_row)


def get_sample_idx(args, num_of_data):
    sample_seq = np.zeros((2 * args.total_run, args.num_of_arms), dtype=np.int)
    for i in range(2 * args.total_run):
        sample_seq[i] = np.random.choice(num_of_data, args.num_of_arms)
    return sample_seq


def real_data_experiment(reward_model, save_log_dir, data_file, num_of_arms, dim, latent_shape, num_of_samples_per_class):
    args.reward_model = reward_model
    args.save_log_dir = save_log_dir
    data_file = data_file
    args.num_of_arms = num_of_arms
    args.dim = dim
    args.latent_shape = latent_shape
    args.feature_cols = np.array([i for i in range(1, args.dim + 1)])
    args.max_num_of_iter = int(args.total_run / args.online_max_step)
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    # sample_seq = get_sample_idx(args, num_of_data=70000)
    sample_seq = np.zeros((args.total_run, args.num_of_arms), dtype=np.int)
    for idx in range(args.total_run):
        arms = np.zeros(args.num_of_arms)
        for i in range(args.num_of_arms):
            arms[i] = random.randint(i * num_of_samples_per_class + 1, (i + 1) * num_of_samples_per_class - 1)
        np.random.shuffle(arms)
        sample_seq[idx] = arms
    sample_seq.reshape((args.total_run, args.num_of_arms))

    #lin_ucb_runner(args, data_file, sample_seq)
    #lin_ts_runner(args, data_file, sample_seq)
    #inter_neural_ts_runner(args, data_file, sample_seq)
    neural_linear_runner(args, data_file, sample_seq)

    '''
    for i in [1,2,3,4,5]:
        for j in [100, 200, 300, 400, 500]:
            args.latent_shape = int(args.dim / i)
            args.online_max_step = j
            args.max_num_of_iter = int(args.total_run / args.online_max_step)
            hybrid_neural_bandit_runner(args, data_file, sample_seq, real_data=True)
    args.latent_shape = latent_shape
    '''

    #exp3_runner(args, data_file, sample_seq)
    #rp_runner(args, data_file, sample_seq)
    e_greedy_runner(args, data_file, sample_seq)
    #neural_ucb_runner(args, data_file, sample_seq, dim_cut=20)
    #neural_linear_runner(args, data_file, sample_seq)
    #kernel_ucb_runner(args, data_file, sample_seq)


if __name__ == '__main__':
    # initialization
    args = parse_args()
    generate_synthetic_data(args)

    args.reward_col = np.array([0])
    args.feature_cols = np.array([i for i in range(1, args.dim + 1)])

    generate_synthetic_data(args)

    # comparisons on synthetic datasets

    '''
    for arms in [200]:
        # generate sample sequence
        args.dim = 500
        args.num_of_arms = arms
        sample_seq = get_sample_idx(args, args.num_of_data)

        for rwd, shape in [('cos', 100), ('square', 100), ('exp', 100)]:
            args.latent_shape = shape
            args.reward_model = rwd
            args.save_log_dir = './regret_stats/' + rwd + '/' + str(arms) + '/'
            if not os.path.exists(args.save_log_dir):
                os.makedirs(args.save_log_dir)
            data_file = './data/synthetic_' + args.reward_model + '_dim' + str(args.dim) + '.csv'
            lin_ucb_runner(args, data_file, sample_seq)
            lin_ts_runner(args, data_file, sample_seq)
            exp3_runner(args, data_file, sample_seq)
            rp_runner(args, data_file, sample_seq)
            e_greedy_runner(args, data_file, sample_seq)
            neural_linear_runner(args, data_file, sample_seq)
            #neural_ucb_runner(args, data_file, sample_seq)
            #kernel_ucb_runner(args, data_file, sample_seq)
            #inter_neural_ts_runner(args, data_file, sample_seq)
            inter_neural_ucb_runner(args, data_file, sample_seq)
    '''

    # comparisons on real-world datasets

    real_data_experiment(
        reward_model= 'MNIST',
        save_log_dir='./regret_stats/MNIST/',
        data_file='./data/mnist.data.csv',
        num_of_arms=10,
        dim=784,
        latent_shape=100,
        num_of_samples_per_class=1000
    )


    # find optimal latent feature dimension for HybridNeuralBandit
    '''
    for latent_shape in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for rwd in ['cos', 'square', 'exp']:
            args.latent_shape = latent_shape
            args.reward_model = rwd
            data_file = './data/synthetic_' + args.reward_model + '_dim' + str(args.dim) + '.csv'
            hybrid_neural_bandit_runner(args, data_file, sample_seq)
    
    for rwd, shape in [('cos', 40), ('square', 10), ('exp', 20)]:
        args.latent_shape = shape
        args.reward_model = rwd
        data_file = './data/synthetic/synthetic_' + args.reward_model + '_dim' + str(args.dim) + '.csv'
        if rwd == 'square':
            #lin_ucb_runner(args, data_file, sample_seq)
            #exp3_runner(args, data_file, sample_seq)
            #rp_runner(args, data_file, sample_seq)
            #e_greedy_runner(args, data_file, sample_seq)
            #neural_linear_runner(args, data_file, sample_seq)
            neural_ucb_runner(args, data_file, sample_seq)
            #kernel_ucb_runner(args, data_file, sample_seq)
    '''

    # efficiency test for NeuralUCB
    '''
    sample_seq = np.zeros((args.total_run, args.num_of_arms), dtype=np.int)
    for i in range(args.total_run):
    sample_seq[i] = np.random.choice(args.num_of_data, args.num_of_arms)
    for rwd in ['cos', 'square', 'exp']:
        args.reward_model = rwd
        data_file = './data/synthetic_' + args.reward_model + '_dim' + str(args.dim) + '.csv'
        neural_ucb_runner(args, data_file, sample_seq, dim_cut=100)
    '''


