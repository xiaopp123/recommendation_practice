#encoding=utf-8
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    def __init__(self, path):
        self.train_matrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.test_ratings = self.load_rating_file_as_list(path + '.test.rating')
        self.test_negatives = self.load_negative_file(path + ".test.negative")

    def load_negative_file(self, file_name):
        negativeList = []
        with open(file_name, "r") as fr:
            for line in fr:
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
        return negativeList
    
    def load_rating_file_as_list(self, file_name):
        ratingList = []
        with open(file_name, "r") as fr:
            for line in fr:
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
        return ratingList

    def load_rating_file_as_matrix(self, file_name):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(file_name, "r") as fr:
            for line in fr:
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
        #print(num_users, num_items)
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(file_name, "r") as fr:
            for line in fr:
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
        return mat

#data_set = Dataset('neural_collaborative_filtering/Data/ml-1m')
