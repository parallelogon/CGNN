import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
from MLCGNN import *
# load the data, randomly permute, and split in to
# train and test in order to generate a generalizable
# graph structure
X = np.load('movies_cleaned.npy')

total,m = X.shape

n_train = np.floor(total*.90).astype('int')
n_val = np.ceil(.1*n_train).astype('int')
n_train = n_train - n_val

I = np.random.permutation(total)
X = X[I,:]

X = X[:total,:]
n_train = np.floor(total*.90).astype('int')
n_val = np.ceil(.1*n_train).astype('int')
n_train = n_train - n_val

X_train = X[:n_train,:]
X_val = X[n_train:n_train+n_val]
X_test = X[n_train+n_val:,]


# TODO: this graph should be an object
# build a graph using correlation structure a better way
# would be to use a hash function which maps close points 
# close to one another, or an aproach similar to a t-sne?
cov = np.zeros((m,m))
for movie1 in range(m):
    for movie2 in range(movie1,m):
        # find users which have rated both movies
        mutual = (X_train[:,movie1] != 0)*(X_train[:,movie2]!=0)

        # mean for movie 1
        mu_1 = np.mean(X_train[mutual,movie1])

        # mean for movie 2
        mu_2 = np.mean(X_train[mutual,movie2])

        # calculate covariance between movie1 movie 2
        cov[movie1, movie2] = 1/len(mutual) * (X_train[mutual,movie1] - mu_1).T @ (X_train[mutual, movie2] - mu_2)

# calculate correlation
for movie1 in range(m):
    for movie2 in range(movie1,m):
        cov[movie1,movie2] = cov[movie1,movie2]/np.sqrt(cov[movie1,movie1]*cov[movie2,movie2])
        cov[movie2, movie1] = cov[movie1, movie2]

for i in range(m):
    cov[i,i] = 0

# sparsify the graph for easier computation by only taking the largest
# 40 entries in each row.
inds = np.array([i for i in range(m)])
for row in range(m):
    cov[:,np.argsort(cov[1,:])[::-1][40:]] = 0

#normalize the graph by largest eigenvalues
eigs = np.linalg.eig(cov)
W = cov/np.max(np.abs(eigs[0]))

# index 110
# only consider users who have rated 'contact', index 110
# transform the data to practice predicting the missing rating.
# any index in the range [0,m] can be chosen.
index = 110

def data_transform_index(X,index):
    X_U = X[X[:,index] != 0]
    x = []
    y = []

    for user in X_U:
        for movie_index in range(len(user)):
            y_sparse = np.zeros(len(X_U[1,:]))
            y_sparse[movie_index] = user[movie_index]
            x.append(user - y_sparse)
            y.append(y_sparse)

    y = torch.tensor(y)#p.array(y)
    x = torch.tensor(x)#np.array(x)
    return(x,y)

X_train_contact,y_train_contact = data_transform_index(X_train,index)
X_val_contact,y_val_contact = data_transform_index(X_val,index)
X_test_contact,y_test_contact = data_transform_index(X_test, index)

n_train = X_train_contact.shape[0]
n_val = X_val_contact.shape[0]
n_test = X_test_contact.shape[0]

# Initializing the model
F = np.array([[[1,64],[64,1]]])
graphFilter = MLGNN(W, [8,1], F,nn.ReLU())

# Hyperparameters, training EPOCHS, BATCHSIZE, and Learning Rate
EPOCHS = 30
BATCHSIZE = 200
LR = 0.05

# Use adam optimizer with base values for betas
optimizer = optim.Adam(graphFilter.parameters(), lr=LR)

# Allocate number for batch size, in the event that the dataset size is not
# an even multiple of batchsize make uneven batches by adding to final batch
if n_train < BATCHSIZE:
    n_batches = 1
    BATCHSIZE = n_train #- n_test
elif n_train % BATCHSIZE != 0:
    n_batches = np.ceil(n_train/BATCHSIZE).astype(np.int16)
    BATCHSIZE = [BATCHSIZE] * n_batches
    while sum(BATCHSIZE) != n_train:
        BATCHSIZE[-1] -= 1
else:
    n_batches = np.int(n_train/BATCHSIZE)
    BATCHSIZE = [BATCHSIZE] * n_batches


batch_index = np.cumsum(BATCHSIZE).tolist()
batch_index = [0] + batch_index

epoch = 0

loss_train = []
loss_val = []

# Main training loop
while epoch < EPOCHS:
    random_permutation = np.random.permutation(n_train)
    id = [int(i) for i in random_permutation]

    print("")
    print("Epoch %d" % (epoch+1))

    batch = 0
    while batch < n_batches:
        #find the batch index
        this_batch = id[batch_index[batch]:batch_index[batch+1]]

        X_batch = X_train_contact[this_batch,:]
        y_batch = y_train_contact[this_batch,:]

        if (epoch * n_batches + batch) % 5 == 0:
            print("")
            print("    (E: %2d, B: %3d)" % (epoch+1, batch+1),end = ' ')
            print("")

        # zero gradients inbetween training rounds
        graphFilter.zero_grad()

        # obtain outputs
        #y_hat = graphFilter(X_batch.T).T
        y_hat = graphFilter(X_batch.T).T

        # compute loss
        loss_value_train = loss(y_hat, y_batch)

        # compute gradients
        loss_value_train.backward()

        # optimize
        optimizer.step()

        loss_train += [loss_value_train.item()]

        if (epoch * n_batches + batch) % 50 == 0:
            with torch.no_grad():
                # Obtain the output of the GNN
                y_hat_val = graphFilter(X_val_contact.T).T

            # Compute loss
            loss_value_val = loss(y_hat_val, y_val_contact)
            
            loss_val += [loss_value_val.item()]

            print("\t Graph Filter: %6.4f [T]" % (
                    loss_value_train) + " %6.4f [V]" % (
                    loss_value_val))
            
            # Saving the best model so far
            if len(loss_val) > 1:
                if loss_value_val <= min(loss_val):
                    best_model =  copy.deepcopy(graphFilter)
            else:
                best_model =  copy.deepcopy(graphFilter)    
        batch+=1
    
    epoch+=1

print("")

plt.subplot(1,2,1)
plt.plot(loss_train)
plt.subplot(1,2,2)
plt.plot(loss_val)

y_hat_test_best = best_model(X_test_contact.T).T
y_hat_test_last = graphFilter(X_test_contact.T).T

print("Test Loss %f (Best) %f (last)\n" % (loss(y_hat_test_best,y_test_contact),loss(y_hat_test_last,y_test_contact)))