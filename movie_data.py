import numpy as np

# import the data
path = "./ml-100k/u.data"
data = np.genfromtxt(path, dtype= None)


# format the data such that X_{i,j} = user rating i of movie j
# movies with no rating are set to zero

unique_users = np.sort(np.unique(data[:,0]))
unique_movies = np.unique(data[:,1])
X = np.zeros((len(unique_users),len(unique_movies)))

for user in range(len(unique_users)):
    for movie,rating in data[data[:,0] == unique_users[user],1:3]:
        X[user,movie-1] = rating

# drop columns in which the number of reviews (nonzero entries) are less
# than 150
# 
contact = np.cumsum(np.sum(X != 0, axis = 0) >= 150)[257]
print("Index of the movie 'contact' %d"%contact)
X = X[:,np.sum(X != 0, axis = 0) >= 150]

# save the modified dataset
outfile = "movies_cleaned"
np.save(outfile, X)

