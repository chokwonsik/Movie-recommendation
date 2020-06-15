# -*-coding: utf 8

import os
import matplotlib
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

warnings.filterwarnings("ignore")

rating_file_path = "movie/ratings.dat"
movie_file_path = "movie/movies.dat"
user_file_path = "movie/users.dat"

rating_data = pd.read_csv(rating_file_path, names= ['user_id', 'movie_id', 'rating', 'time'], delimiter="::")
movie_data = pd.read_csv(movie_file_path, names= ['movie_id', 'title', 'genre'], delimiter='::')
user_data = pd.read_csv(user_file_path, names= ['user_id', 'gender', 'age', 'occupation', 'zipcode '], delimiter='::')
print(rating_data.head())
# print(movie_data.head())
# print(user_data.head())

print("-----------------------")


reader = Reader(rating_scale(1,5))

data = Dataset.load_from_df(rating_data[['user_id', 'movie_id', 'rating']], reader)

train_data = data.build_full_trainset()
model = SVD (n_factors=8, n_epochs=20)
model.fit(train_data)