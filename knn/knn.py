import numpy as np
from numpy.linalg import norm
from collections import Counter

# Formulas for distances calculations
def manhattan_distance(x_1, x_2):
    np.sum(np.abs(x_1 - x_2))

def euclidean_distance(x_1, x_2):
    np.sqrt(np.sum((x_1 - x_2)**2))

def cosine_distance(x_1, x_2):
    1 - np.dot(x_1, x_2)/(norm(x_1)*norm(x_2))

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        # In k-nearest neighbours the fitting is just saving
        # the training data to the calculate the distance of
        # the data to predict
        self.X_train = X_train
        self.y_train = y_train

    def get_neighbours(self, x_0, d_formula=euclidean_distance):
        distances = []
        for x_train in self.X_train:
            distances.append(d_formula(x_0, x_train))
        
        distances = np.array(distances)
        sorted_distances = np.argsort(distances)
        neighbours = sorted_distances[:self.k] # We took the k-nearest neighbours

