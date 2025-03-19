import numpy as np
from numpy.linalg import norm
from collections import Counter

# Formulas for distances calculations
def manhattan_distance(x_1, x_2):
    d = np.sum(np.abs(x_1 - x_2))
    return d

def euclidean_distance(x_1, x_2):
    d = np.sqrt(np.sum((x_1 - x_2)**2))
    return d

def cosine_distance(x_1, x_2):
    d = 1 - np.dot(x_1, x_2)/(norm(x_1)*norm(x_2))
    return d

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, Y_train):
        # In k-nearest neighbours the fitting is just saving
        # the training data to the calculate the distance of
        # the data to predict
        self.X_train = X_train
        self.Y_train = Y_train

    def gather_distances(self, x_0, d_formula=euclidean_distance):
        distances = []
        for x in self.X_train:
            distances.append(d_formula(x_0, x))
        
        return distances
    
    def get_neighbours(self, distances):
        distances = np.array(distances)
        sorted_distances = np.argsort(distances)
        neighbours = self.Y_train[sorted_distances[:self.k]] # We took the k-nearest neighbours

        neighbours = np.array(neighbours) # Convert neighbours to a np.array
        return neighbours
    
    def most_common(self, neighbours): # Receives a np.array
        k_neighbours = Counter(neighbours) 
        most_common, most_common_count = most_common.k_neighbours(1)[0]

        #I want to know how many neighbours share the top
        top_commons = len(count for count in k_neighbours.values() if count == most_common_count)
        if top_commons == 1:
            return most_common # If there is one top neighbours we return it
        else:
            return most_common(k_neighbours[:-1]) # Otherwise we slice the last neighbour and count again
    
    def predict(self, x_0):
        predicted_class = self.most_common(self.get_neighbours(x_0))
        return predicted_class


