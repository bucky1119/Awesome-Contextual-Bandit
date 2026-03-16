import numpy as np


class FileData(object):
    def __init__(self, file_path, feature_cols=None, reward_col=None):
        self.file = file_path
        self.dataset = np.genfromtxt(self.file, delimiter=',', filling_values=0.0)
        self.sample_num = len(self.dataset)
        self.feature_cols = feature_cols
        self.reward_col = reward_col

    # sample a batch of data
    def sample(self, num, feature_cols=None, label_col=None):
        assert num <= self.sample_num
        rcol = self.reward_col if label_col is None else label_col
        fcols = self.feature_cols if feature_cols is None else feature_cols
        idx = np.random.choice(len(self.dataset), num) # a vector indicating columns to be selected
        sampled_data = self.dataset[idx]
        #feature = sampled_data[:, fcols]
        #reward = sampled_data[:, rcol]
        return sampled_data

    def sample_by_index(self, idx, feature_cols=None, label_col=None):
        """
        return samples of given indices
        """
        rcol = self.reward_col if label_col is None else label_col
        fcols = self.feature_cols if feature_cols is None else feature_cols
        sampled_data = self.dataset[idx]
        #print(sampled_data.shape)
        return sampled_data

    def sample_feature_tensor(self, num, pkey_col=0, skey_col=1):
        idx = np.lexsort((self.dataset[:, skey_col], self.dataset[:,pkey_col]))
        pkey_num = len(np.unique(self.dataset[:,pkey_col]))
        assert len(self.dataset)%pkey_num==0
        skey_num = len(self.dataset)/pkey_num
        smp_ind = np.random.choice(skey_num, num, replace=False)
        feature = self.dataset[:,self.feature_cols]
        feature = feature[idx]
        feature = feature.reshape(pkey_num,skey_num,-1)
        feature = feature[:,smp_ind,:]
        return feature

    def get_features(self, feature_cols=None):
        fcols = self.feature_cols if feature_cols is None else feature_cols
        return self.dataset[:, fcols]


# historical data for offline training
class HistoryData(FileData):
    def __init__(self, size, file_path, feature_cols=None, reward_col=None):
        super().__init__(file_path, feature_cols, reward_col)
        # generate historical data by rewriting dataset by a sample
        self.dataset = self.sample(size, self.feature_cols, self.reward_col)

    # append new data
    def append(self, data):
        self.dataset = np.concatenate((self.dataset, data), axis=0)
