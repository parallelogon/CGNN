
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import numpy as np

# Loss function for graphical outputs.
def loss(yHat,y):
    mse = nn.MSELoss()
    return mse(yHat.reshape([-1,1]),y.reshape([-1,1]))

# allows for the application of a graph filter which examines multiple features
def apply_mimo_graph_filter(A,Z,h,k,Fin,Fout):
    u_k = Z
    u = 0
    _,n_samples = Z.shape

    chunks = int(n_samples/Fin)
    for i in range(k):
        u_in = torch.cat(torch.chunk(u_k, chunks, dim = 1),dim = 0)
        u_in = torch.cat( (u_in @ h[:,:,i].type(torch.double)).chunk(chunks), dim = 1)

        u += u_in
        u_k = A@u_k

    return u

# Each layer is a graph convolution, here called a model
class Model(nn.Module):
    def __init__(self,A,k, Fin = 1, Fout = 10, sigma = lambda x: x):
        super(Model, self).__init__()

        # initialize weights and layer parameters
        self.k = k
        h = torch.randn((Fin,Fout,k))
        self.h = nn.parameter.Parameter(h)
        self.Fin = Fin
        self.Fout = Fout
        self.reset_parameters()

        # Store Adjacency matrix A
        self.A = torch.tensor(A)
        self.sigma = sigma

    def reset_parameters(self):
        stdv = 1./np.sqrt(self.k)
        self.h.data.uniform_(-stdv, stdv)

    def forward(self, z):
        out= apply_mimo_graph_filter(self.A, z, self.h, self.k, self.Fin, self.Fout)
        out = self.sigma(out)
        return out

# main Multi Layer Graphical Neural Network object.  Initialized with 
# A - adjacency matrix
# K - list of number of diffusions/diffusion parameters for each layer.
#   with L layers len(K) = L
# F - 3d array with format n_features_in x n_features_out x layer.
#   Must have n_features_out (layer L - 1) = n_features_in (layer L)
# Sigma - default linear, a pointwise nonlinearity can be chosen (nn.ReLU,
#   nn.Tanh, nn.LeakyReLU or a user defined function)

class MLGNN(nn.Module):
    def __init__(self, A, K, F, sigma = lambda x: x):
        super(MLGNN, self).__init__()
        layers = []
        L = len(K)
        assert L == F.shape[2]

        for layer in range(L):
            layers.append(Model(A,K[layer],F[0,0,layer],F[0,1,layer], sigma))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
