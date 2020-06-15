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
# print(rating_data.head())
# print(movie_data.head())
# print(user_data.head())

# print("-----------------------")
#
# print("total number of movie in data :", len(movie_data['movie_id'].unique()))
#
# movie_data['year'] = movie_data['title'].apply(lambda x: x[-5:-1])
# print(movie_data['year'].value_counts().head(10))

# print(movie_data['genre'].value_counts()[:50])

# unique_genre_dict = {}
# for index, row in movie_data.iterrows():
# 
#     genre_combination = row['genre']
#     parser_genre = genre_combination.split("|")
# 
#     for genre in parser_genre:
#         if genre in unique_genre_dict:
#             unique_genre_dict[genre] += 1
#         else:
#             unique_genre_dt(movie_rate_count, bins=200ict[genre] = 1
# 
# print(unique_genre_dict)
# 
# plt.rcParams['figure.figsize'] = [20, 16]
# plt.bar(list(unique_genre_dict.keys()), list(unique_genre_dict.values()), alpha=0.8)
# plt.title('Popular genre in movies')
# plt.ylabel('Count of Genre ', fontsize =12)
# plt.xlabel('Genre ', fontsize = 12)
# plt.show()
# 

movie_rate_count = rating_data.groupby('movie_id')['rating'].count().values
plt.rcParams['figure.figsize'] = [8,8]
fig = plt.his