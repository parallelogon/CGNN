import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import distance_matrix
import torch
from copy import deepcopy

"""
 Channel class, this is a test class for a later implementation.
 Not ultimately used.

class Channel():
    N0 = 10**-6

    def __init__(self, d0, gamma):
        self.d0 = d0
        self.gamma = gamma

    def pathloss(self,d, **kwargs):

        if len(kwargs.keys()) == 0:
            return (self.d0/d)**self.gamma
        else:
            assert 's' in kwargs.keys() and 'Q' in kwargs.keys()
            s = kwargs['s']; Q = kwargs['Q'];
            
            L = max(d.shape)
            d = d.reshape((1,-1))
            return (self.d0/d)**self.gamma * torch.Tensor(Q,L).exponential_(s)/s #torch.random.exponential(s, size = (Q,L))/s

    def c(self, p,h):
        assert len(p) == len(h)
        return torch.log(1 + p*h/self.N0)

channel = Channel(1, 2.2)

d = torch.arange(1,100)

fig1 = plt.figure()
plt.plot(d,torch.log(channel.pathloss(d)))
fig1.show()

realization = channel.pathloss(d, s = 2, Q = 100)

fig2 = plt.figure()
plt.plot(d.numpy(), (torch.max(realization, axis = 0))[0].numpy())
plt.plot(d.numpy(), torch.min(realization, axis = 0)[0].numpy())
fig2.show()

high = []
low = []
for dist in range(len(d)):
    p = 0.05*torch.ones(realization[dist,:].shape)
    quality = channel.c(p, realization[dist,:])
    high.append(torch.max(quality))
    low.append(torch.min(quality))

fig3 = plt.figure()
plt.plot(d,high)
plt.plot(d,low)
plt.show()
"""


class Net():

    """
    Net() class which holds all the information of a wireless network with area
    wx*wy, max receiver distance wc, and density of transmitters/receivers rho
    calculate the power transmitted gives a pathloss_matrix (h) s.t. h_{i,j} is
    the power transmitted from transmitter i to receiver j.  Sampled Q times,
    the power transmitted is assumed to follow an exp. distribution
    with factors d0 and gamma found experimentally
    """

    def __init__(self, wx, wy, wc, rho):
        n = int(wx*wy*rho)
        self.n = n
        self.transmitters = torch.hstack(
            (torch.Tensor(n, 1).uniform_(0, wx), torch.Tensor(n, 1).uniform_(0, wy)))

        theta = torch.Tensor(n, 1).uniform_(0., 2*np.pi)

        self.receivers = self.transmitters + \
            torch.Tensor(n, 1).uniform_(0, wc) * \
            (torch.hstack((torch.cos(theta), torch.sin(theta))))

    def pathloss_matrix(self, d0, gamma):
        self.d0 = d0
        self.gamma = gamma

        a = self.transmitters
        b = self.receivers
        d = torch.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

        self.d = d
        self.H = normalize_matrix((d0/d)**gamma)
        return self

    def sample(self, s, Q):
        self.samples = self.H[None, :, :] * \
            torch.Tensor(Q, self.n, self.n).exponential_(s)/s
        return self

# Helper function to stabilize training


def normalize_matrix(A):
    eigenvalues = np.linalg.eigvals(A)
    return A / np.max(eigenvalues.real)


# Channel capacity is quantity to be maximized
def capacity(Q_nets, P, N0=1e-6):

    # project columnQ_nets of P onto
    BATCHSIZE, n, _ = Q_nets.shape

    # extract diagonals of hp
    num = torch.diagonal(Q_nets, dim1=-2, dim2=-1)*(P.squeeze())

    # subtract diagonals from matrix multiply transmitter j on receiver i
    den = torch.einsum('ijk,ijl->ikl', Q_nets, P).squeeze() - num + N0
    return torch.log(1 + num/den)


# net samples should inheret attributes from Net() and the following parameters
# determine the size of the area / number of nodes in the graph
wx = 200
wy = 100
wc = 50
rho = 0.005
n = int(rho*wx*wy)
net = Net(wx, wy, wc, rho)
d0 = 1
gamma = 2.2
s = 2

# Batchsize, generate samples on the fly
BATCHSIZE = 100

# graphs the net with black lines connecting receivers and transmitters
fig3 = plt.figure()

plt.scatter(net.receivers[:, 0], net.receivers[:, 1], c='red')
plt.scatter(net.transmitters[:, 0], net.transmitters[:, 1], c='blue')

for i in range(net.receivers.shape[0]):
    plt.plot([net.transmitters[i, 0], net.receivers[i, 0]],
             [net.transmitters[i, 1], net.receivers[i, 1]], c='black', lw=.5)

plt.show()


class Layer(nn.Module):
    def __init__(self, k, Fin, Fout, sigma):
        super(Layer, self).__init__()

        self.k = k

        A = torch.rand(k, Fin, Fout)
        self.A = nn.parameter.Parameter(A)

        self.Fin = Fin
        self.Fout = Fout

        self.sigma = sigma

    def forward(self, H, x):
        z = self.apply_filter(H, x, self.A)
        x = self.sigma(z)
        return x

    def apply_filter(self, H, X, A):
        k, _, _ = A.shape

        u = 0
        for i in range(k):
            HZA = X @ A[i, :, :]
            u = u + HZA
            X = torch.einsum('ijk,ikl->ijl', H, X)

        return u


class REGNN(nn.Module):
    def __init__(self, K, F, sigma=lambda x: x):
        super(REGNN, self).__init__()

        layers = []

        assert len(K) == len(F) - 1, "Incorrect number of layers"

        for layer in range(len(K)):
            layers.append(Layer(K[layer], F[layer], F[layer+1], sigma))

        # cannot use sequential, as it only allows one parameter input
        self.layers = nn.ModuleList(layers)

    def forward(self, H, x):
        for layer in self.layers:
            x = layer(H, x)
        return x

# negative quantity to be maximized, upper bounded at zero


def loss(Hh, P, mu):
    return -capacity(Hh, P).mean() + mu*P.mean()

# reformulated as a pure minimization problem with constraint mu
def loss_constrined(H,P,mu, P_MAX):
    return -capacity(H,P).mean() + mu.T @ (P.mean(axis = 0) - P_MAX)

net.pathloss_matrix(d0, gamma)


# Initialize the model (R)andom (E)dge (G)raphical (N)eural (N)etwork
F = [1, 1, 1, 1, 1, 1]
K = [5, 5, 5, 5, 5]
H = net.sample(s, BATCHSIZE).samples
filter = REGNN(K, F, nn.ReLU())

# Hyperparameters
LR = 0.01  # learning rate
EPOCHS = 10
N_REALIZATIONS = 10**4 # number of samples total per epoch
N_BATCHES = N_REALIZATIONS//BATCHSIZE
MU = 0.01 #torch.tensor([0.01])  # penalty on total power allocated
LOSS_TRAIN = []
LOSS_VALID = []

P_MAX = 10e-3*torch.ones(net.n,1)
mu = torch.zeros(net.n, 1)
mu.requires_grad = True

# Initalize the network
net = Net(wx, wy, wc, rho).pathloss_matrix(d0, gamma)
val_net = Net(wx//2, wy//2, wc//2, rho)
val_net.pathloss_matrix(d0, gamma)

# Use adam optimizer with base values for betas, two different steps one for the 
# original problem and one for the constraints
optimizerPrimal = optim.Adam(filter.parameters(), LR)
optimizerDual = optim.Adam([mu], LR/10)

best_model = []
last_model = []

for epoch in range(EPOCHS):
    print("")
    print("Epoch %d" % (epoch+1))

    for batch in range(N_BATCHES):

        # sample BATCHSIZE randomly generated samples from the random net
        H = net.sample(s, BATCHSIZE).samples[:]
        P = torch.ones(BATCHSIZE, net.n, 1)

        if batch % 20 == 0:
            print("")
            print("    (E: %2d, B: %3d)" % (epoch+1, batch+1), end=' ')
            print("")

        # zero gradient
        filter.zero_grad()

        # power values
        P = filter(H, P)

        # loss value
        #lval = loss(H, P, MU)
        lval = loss_constrined(H, P, mu, P_MAX)

        # deriv
        lval.backward()

        # optimize step
        optimizerPrimal.step()
        optimizerDual.step()

        # clamp mu values
        with torch.no_grad():
            mu = mu.clamp(min = 0)

        # track
        LOSS_TRAIN += [lval.item()]

        if batch % 20 == 0:
            # validate the model
            H = val_net.sample(s,BATCHSIZE//2).samples
            P = torch.ones(BATCHSIZE//2, val_net.n, 1)
            P = filter(H,P)
            lval_valid = capacity(H, P).mean().item()
            
            if len(LOSS_VALID) == 0:
                best_model = deepcopy(filter)

            elif lval_valid > max(LOSS_VALID):
                best_model = deepcopy(filter)

            LOSS_VALID.append(lval_valid)

            print("\t Filter: %6.4f [T]" % (
                    lval.item()) + " %6.4f [V]" % (
                    lval_valid))
print("")

fig= plt.plot()
plt.plot(LOSS_TRAIN)

plt.show()

def test_net(net, filter, s, Q):
    H =net.sample(s, Q).samples
    P = torch.ones(BATCHSIZE, net.n, 1)
    P = filter(H,P)
    return(capacity(H, P).mean())

test_score_last = test_net(net, filter, s, BATCHSIZE)
test_score_best = test_net(net, best_model, s, BATCHSIZE)

print("Final Score: %6.4f (last)" % (test_score_last) + " %6.4f (best)" % test_score_best)