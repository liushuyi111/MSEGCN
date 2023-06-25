import torchvision.models as models
from torch.nn import Parameter
# from gcn_util import *
import torch
import torch.nn as nn
from scipy import io as scio
import torch.nn.functional as F
class GraphConvolution(nn.Module):

    def __init__(self, in_features=300, out_features=63):
        super(GraphConvolution, self).__init__()
        # self.config = config
        # self.in_features = torch.Tensor(in_features) #63*300
        # self.out_features = torch.Tensor(out_features) #63*63
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.in_features, self.out_features)) #300*63

        self.relu = nn.LeakyReLU(0.2)
        # if bias:
        #     self.bias = Parameter(torch.Tensor(1, 1, 63))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        # support = torch.matmul(input, self.weight)  #Inp :63*300
        # output = torch.matmul(adj, support)   #63*300
        #
        output = torch.matmul(adj, input)
        output = torch.matmul(output,self.weight)
        output = F.sigmoid(output)

        return output





if __name__ == '__main__':
    config = dict()
    inp = scio.loadmat('/media/HardDisk/lsy/meipai_dataset/LabelVectorized.mat')['labelVector']
    inp  = torch.Tensor(inp);
    adj = scio.loadmat('/media/HardDisk/lsy/meipai_dataset/shuffle_data/adj_less.mat')['adj_less']
    adj = torch.Tensor(adj)
    config['in_features'] = 300
    config['out_features'] = 63

    # inp = torch.rand(63,300)
    # adj = torch.rand(63,63)
    net = GraphConvolution()
    output = net(inp, adj)

    print(output)

    print('OK')
