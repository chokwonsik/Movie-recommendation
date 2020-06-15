# -*-coding: utf 8

import os
import time

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

rating_data = pd.read_csv(rating_file_path, names=['user_id', 'movie_id',
                                                   'rating', 'time'], delimiter="::")
movie_data = pd.read_csv(movie_file_path, names=['movie_id', 'title',
                                                 'genre'], delimiter='::')
user_data = pd.read_csv(user_file_path, names=['user_id', 'gender', 'age',
                                               'occupation', 'zipcode '], delimiter='::')

rating_table = rating_data[['user_id', 'movie_id', 'rating']] \
    .set_index(["user_id", "movie_id"]).unstack()
# print(rating_table.head(10))

# SVD 라이브러리를 사용하기 위한 학습 데이터를 생성합니다.
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(rating_data[['user_id', 'movie_id', 'rating']], reader)
train_data, test_data = train_test_split(data, test_size=0.2)

# SVD 모델을 학습합니다.
train_start = time.time()
model = SVD(n_factors=8, lr_all=0.005, reg_all=0.02, n_epochs=100)
print(model.fit(train_data))
train_end = time.time()
print("trainning time of model: %.2f seconds" % (train_end - train_start))

predictions = model.test(test_data)

# 테스트 데이터의 RMSE를 출력합니다.
print("RMSE of test dataset in SVD model: ", accuracy.rmse((predictions)))

# user_id가 4인 유저의 영화 평가 데이터입니다.
target_user_id = 4
target_user_data = rating_data[rating_data['user_id'] == target_user_id]
target_user_data.head(5)

# user_id 4인 유저가 평가한 영화 히스토리 정보를 추출합니다.
target_user_movie_rating_dict = {}

for index, row in target_user_data.iterrows():
    movie_id = row['movie_id']
    target_user_movie_rating_dict[movie_id] = row['rating']

print("유저의 영화 평가 점수:", target_user_movie_rating_dict)

# 타겟 유저(user_id가 4인 유저)가 보지 않은 영화 정보를 테스트 데이터로 생성합니다.
test_data = []
for index, row in movie_data.iterrows():
    movie_id = row['movie_id']
    rating = 0
    if movie_id in target_user_movie_rating_dict:
        continue
    test_data.append((target_user_id, movie_id, rating))
# 타겟 유저의 평점 점수를 예측합니다.
target_user_predictions = model.test(test_data)


# 예측된 점수 중, 타켓 유저의 영화별 점수를 target_user_movie_predict_dict로 저장합니다.
def get_user_predicted_ratings(predictions, user_id, user_history):
    target_user_movie_predict_dict = {}
    for uid, mid, rating, predicted_rating, _ in predictions:
        if user_id == uid:
            if mid not in user_history:
                target_user_movie_predict_dict[mid] = predicted_rating
    return target_user_movie_predict_dict


target_user_movie_predict_dict = get_user_predicted_ratings \
    (predictions=target_user_predictions,
     user_id=target_user_id,
     user_history=target_user_movie_rating_dict)

print("타겟 유저 예측 점수 : ", target_user_movie_predict_dict)

# target_user_movie_predict_dict 에서 예측된 점수 중 , 타겟 유저의 Top 10 영화를 선정합니다
target_user_top10_predicted = sorted(target_user_movie_predict_dict.items(),
                                     key=operator.itemgetter(1), reverse=True)[:10]

# 예측된 Top 10 영화
print("예측된 Top 10 영화: ", target_user_top10_predicted)

# 타이틀 정보로 출력하기 위해 movie_id마다 movie_title을 딕셔너리 형태로 저장합니다.
movie_dict = {}
for index, row in movie_data.iterrows():
    movie_id = row['movie_id']
    movie_title = row['title']
    movie_dict[movie_id] = movie_title

# 앞서 계산한 Top 10영화에 movie_title을 매핑하여 출력합니다.
for predicted in target_user_top10_predicted:
    movie_id = predicted[0]
    predicted_rating = predicted[1]
    print("타겟 관람x :", movie_dict[movie_id], ":", predicted_rating)

# 타겟 사용자의 기존 선호 영화와 비교합니다.
target_user_top10_real = sorted(target_user_movie_rating_dict.items(),
                                key=operator.itemgetter(1), reverse=True)[:10]
for real in target_user_top10_real:
    movie_id = real[0]
    real_Rating = real[1]
    print("타겟 관람o :", movie_dict[movie_id], ",", real_Rating)

# 예측 점수와 실제 점수를 영화 타이틀에 매핑합니다.
origin_rating_list = []
predicted_rating_list = []
movie_title_list = []
idx = 0
# for movie_id, origin_rating in target_user_movie_rating_dict.items():
#     idx = idx + 1
#     origin_rating = origin_rating
#     try:
#         predicted_rating = round((target_user_movie_predict_dict[movie_id], 3))
#     except KeyError:
#         print("KeyError")
#     movie_title = movie_dict[movie_id]
#     print("movie", str(idx), ":", movie_title, "-", origin_rating, "/", predicted_rating)
#     origin_rating_list.append(origin_rating)
#     predicted_rating_list.append(predicted_rating)
#     movie_title_list.append(str(idx))

for movie_id, predicted_rating in target_user_movie_predict_dict.items():
    idx = idx + 1
    predicted_rating = round(predicted_rating, 2)
    origin_rating = target_user_movie_rating_dict[movie_id]
    movie_title = movie_dict[movie_id]
    print("movie", str(idx), ":", movie_title, "-", origin_rating, "/", predicted_rating)
    origin_rating_list.append(origin_rating)
    predicted_rating_list.append(predicted_rating)
    movie_title_list.append(str(idx))



# 실제 점수와 예측 점수를 리스트로 추출합니다.
origin = origin_rating_list
predicted = predicted_rating_list

# 영화의 개수만큼 bar 그래프의 index 개수를 생성합니다.
plt.rcParams['figure.figsize'] = (10, 6)
index = np.arange(len(movie_title_list))
bar_width = 0.2

# 실제 점수와 예측 점수를 bar 그래프로 출력합니다.
rects1 = plt.bar(index, origin, bar_width,
                 color='orange',
                 label='Origin')
rects2 = plt.bar(index + bar_width, predicted, bar_width,
                 color='green',
                 label='Predicted')
plt.xticks(index, movie_title_list)
plt.legend()
print(plt.show())
