import time
import torch
import argparse
import numpy as np

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')

mean = [0.2, 0.9, 0.5, 3, 1.1, 0.9, 2, 2.5, 1.6, 1.8]*30
var = [3, 2, 4, 3, 3.5, 5.5, 5, 3.5, 5, 3.5]*30
dim = 500
lin_para_seed = np.loadtxt('./seeds/lin_para_' + str(dim) + '_seed.txt', dtype=np.float, delimiter=',')
#matrix_seed = np.loadtxt('./seeds/matrix_' + str(dim) + '_seed.txt', dtype=np.float, delimiter=',')







def parse_args():
    parser = argparse.ArgumentParser("Hybrid Online Offline Neural Contextual Bandits")

    # systen info
    parser.add_argument("--time", type=str, default=time_now, help="algorithms and baselines")

    # environment
    parser.add_argument("--num_of_data", type=int, default=50000, help="the number of pieces of synthetic data to be generated")
    parser.add_argument("--dataset", type=str, default="None", help="the name of dataset")
    parser.add_argument("--num_of_users", type=int, default=1, help="number of users")
    parser.add_argument("--num_of_arms", type=int, default=10, help="number of arms")
    parser.add_argument("--dim", type=int, default=dim, help="dimension of contexts. must be times of 10")
    parser.add_argument("--latent_shape", type=int, default=10, help="dimension of latent features")
    parser.add_argument("--lin_para_seed", type=np.ndarray, default=lin_para_seed, help="the vector parameter to generate reward")
    #parser.add_argument("--matrix_seed", type=np.ndarray, default=matrix_seed, help="the matrix parameter to generate reward")
    parser.add_argument("--reward_model", type=str, default="cos", help="reward model: linear; cos; square; quad")
    parser.add_argument("--algo", type=str, default="linUCB", help="algorithms and baselines: linUCB; hybrid; RP; e-greedy;")


    # data generation
    parser.add_argument("--mean", type=list, default=mean, help="mean for data generation")
    parser.add_argument("--var", type=list, default=var, help="variance for data generation")
    parser.add_argument("--reward_col", type=np.array, default=np.array([0]), help="the column corresponding to rewards in dataset")
    parser.add_argument("--feature_cols", type=np.array, default=np.array([i for i in range(1, dim + 1)]),
                        help="the column corresponding to features in dataset")



    # core training parameters
    parser.add_argument("--device", default=device, help="torch device")
    parser.add_argument("--max_round", type=int, default=1001, help="maximum number of online/offline alternations")
    parser.add_argument("--pre_training_on", type=bool, default=True, help="turn on/off pre_training")
    parser.add_argument("--total_run", type=int, default=10000, help="number of runs for all algorithms")

    parser.add_argument("--offline_max_epoch", type=int, default=2000, help="maximum epoch")
    parser.add_argument("--online_max_step", type=int, default=100, help="maximum step of LinUCB")
    parser.add_argument("--offline_max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--offline_lr", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--historical_data_size", type=int, default=100, help="size of historical data set")
    parser.add_argument("--online_batch_size", type=int, default=100,
                        help="size of bathed data for online training to alleviate cold start")
    parser.add_argument("--offline_batch_size", type=int, default=100,
                        help="batch size for gradient descent for offline training")
    parser.add_argument("--UCB_coeff", type=float, default=1.0, help="coefficient for LinUCB")
    parser.add_argument("--max_num_of_iter", type=int, default=100, help="max number of offline/online iterations")

    # checkpointing
    parser.add_argument("--save_model", type=int, default=100, help="interval of episodes for saving the model")
    parser.add_argument("--start_save_model", type=int, default=100, help="the number of the epoches for saving the model")
    parser.add_argument("--start_save_log", type=int, default=1, help="interval of epoch for saving the log")
    parser.add_argument("--save_dir", type=str, default="models_output/", help="directory in which training state and model should be saved")
    parser.add_argument("--load_model_dir", type=str, default="none", help="directory in which training state and model are loaded")
    parser.add_argument("--save_log_dir", type=str, default="./regret_stats/",
                        help="directory to which log files are output")

    return parser.parse_args()