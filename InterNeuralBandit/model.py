import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, context_shape, latent_shape):
        super(Model, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        #self.linear_c1 = nn.Linear(context_shape, context_shape - int((context_shape - latent_shape) / 3))
        #self.linear_c2 = nn.Linear(context_shape - int((context_shape - latent_shape) / 3), context_shape - int(2 * (context_shape - latent_shape) / 3))
        #self.linear_c = nn.Linear(context_shape - int(2 * (context_shape - latent_shape) / 3), latent_shape)
        self.linear_c1 = nn.Linear(context_shape, context_shape)
        self.linear_c2 = nn.Linear(context_shape, context_shape)
        self.linear_c3 = nn.Linear(context_shape, context_shape)
        self.linear_c = nn.Linear(context_shape, latent_shape)
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, context):
        l1_output = self.LReLU(self.linear_c1(context))
        l2_output = self.LReLU(self.linear_c2(l1_output))
        l3_output = self.LReLU(self.linear_c3(l2_output))
        feature = self.linear_c(l3_output)
        #latent_feature = torch.div(feature.t(),  torch.norm(feature, dim=1))
        #return latent_feature.t()
        return feature

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 6,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(6,16,5)
        self.fc1 = torch.nn.Linear(16*5*5,120)
        self.fc2 = torch.nn.Linear(120,84)
        self.fc3 = torch.nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5) #卷积结束后将多层图片平铺batchsize行16*5*5列，每行为一个sample，16*5*5个特征
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
