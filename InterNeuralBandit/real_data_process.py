import numpy as np
import pickle

# pre-process song dataset, predict the songs of released year 2019

'''
dataset = np.genfromtxt('./data/song.csv', delimiter=',', filling_values=0.0)
row, col = dataset.shape
contexts = dataset[:, [_ for _ in range(1, 91)]]
#contexts = contexts.T
#contexts /= np.linalg.norm(contexts, axis=0)
#contexts = contexts.T

for i in range(row):
    dataset[i, 0] = np.float(( 2009 - 1985 - (2009 - dataset[i, 0])) / 10)

dataset[:, [_ for _ in range(1, 91)]] = contexts

np.savetxt('./data/song2009.csv', dataset, fmt='%f', delimiter=',')
'''

# pre-process USCensus1990 dataset
# num_of_arms = 7
# dim = 68

'''
dataset = np.genfromtxt('./data/USCensus1990.csv', delimiter=',', filling_values=0.0)
row, col = dataset.shape
contexts = dataset[:, [_ for _ in range(1, col)]]
#contexts = contexts.T
#contexts /= np.linalg.norm(contexts, axis=0)
#contexts = contexts.T

for i in range(row):
    dataset[i, 0] = 1.0 if dataset[i, 35] == 0 else 0.0

dataset[:, [_ for _ in range(1, col)]] = contexts
np.savetxt('./data/USCensus1990.data.csv', dataset, fmt='%f', delimiter=',')
'''

# pre-process MNIST dataset
# num_of_arms = 10
# feature dim = 784
'''
dataset = np.genfromtxt('./data/mnist.csv', delimiter=',', filling_values=0.0)
row, col = dataset.shape
print(col)
contexts = dataset[:, [_ for _ in range(1, col)]]
contexts = contexts.T
contexts /= np.linalg.norm(contexts, axis=0)
contexts = contexts.T

for i in range(row):
    dataset[i, 0] = 1.0 if dataset[i, 0] == 0 else 0.0

dataset[:, [_ for _ in range(1, col)]] = contexts
np.savetxt('./data/mnist.data.csv', dataset, fmt='%f', delimiter=',')
'''


# pre-process CIFAR-10 dataset
# num_of_arms = 10
# feature dim = 1024 * 3 = 3072
'''
data = np.zeros((1, 3073))

for i in range(1, 2):
    with open('./data/cifar-10/data_batch_' + str(i), 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        labels = dict['labels']
        features = dict['data'].astype("float")
        print(np.shape(features))
        reward = np.zeros(10000)
        for j in range(10000):
            reward[j] = 1.0 if int(labels[j]) == 0 else 0
        batch = np.zeros((3073, 10000))
        batch[0] = reward
        features /= 255
        batch[1:, :] = features.T
        batch = batch.T
        data = np.concatenate((data, batch), axis=0)

np.savetxt('./data/cifar-10/cifar-10-1.csv', data, fmt='%f', delimiter=',')
'''

# pre-process font dataset
# num_of_arms = 20
# feature dim =  409
# link: http://archive.ics.uci.edu/ml/datasets/Character+Font+Images

'''
dataset = np.genfromtxt('./data/font.csv', delimiter=',', filling_values=0.0)
row, col = dataset.shape
contexts = dataset[:, [_ for _ in range(1, col)]]
contexts /= 255
dataset[:, [_ for _ in range(1, col)]] = contexts
np.savetxt('./data/font_data.csv', dataset, fmt='%f', delimiter=',')
'''

# pre-process madelon_train.data
# num_of_arms = 2
# feature dim =  500
# link: http://archive.ics.uci.edu/ml/datasets/Character+Font+Images


dataset = np.genfromtxt('./data/madelon_train.data.csv', delimiter=',', filling_values=0.0)
row, col = dataset.shape
contexts = dataset[:, [_ for _ in range(1, col)]]
contexts /= 1000
dataset[:, [_ for _ in range(1, col)]] = contexts
np.savetxt('./data/madelon.data.csv', dataset, fmt='%f', delimiter=',')
