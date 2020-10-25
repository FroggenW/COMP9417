
# author: Chang Wang
# 20t2 COMP9417 GROUP PROJECT
# data pre-processing and SVM classfier

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('data.csv')
#print(data.dtypes)

def str_trans(data):
    str_list = data.tolist()
    new_list = []
    for i in range(len(str_list)):
        if str_list[i] not in new_list:
            new_list.append(str_list[i])
    trans_str = {}
    for index in range(len(new_list)):
        trans_str[new_list[index]] = index
    # print(data)
    # print(dict)
    int_list = []
    # print('11', list)
    for index_1 in range(len(str_list)):
        int_list.append(trans_str[str_list[index_1]])
    return int_list

action_type = data['action_type']

combined_shot_type = data['combined_shot_type']
game_event_id = data['game_event_id']
game_id = data['game_id']
lat = data['lat']
loc_x = data['loc_x']
loc_y = data['loc_y']
lon = data['lon']
minutes_remaining = data['minutes_remaining']
period = data['period']
playoffs = data['playoffs']
season = data['season']
seconds_remaining = data['seconds_remaining']
shot_distance = data['shot_distance']
shot_made_flag = data['shot_made_flag']
shot_type = data['shot_type']
#print('shot_type:', shot_type.unique())
shot_zone_area = data['shot_zone_area']
shot_zone_basic = data['shot_zone_basic']
shot_zone_range = data['shot_zone_range']
team_id = data['team_id']
#print('team id:', team_id.unique())
team_name = data['team_name']
game_date = data['game_date']
matchup = data['matchup']
opponent = data['opponent']
shot_id = data['shot_id']

columes = ['action_type','combined_shot_type','loc_x','loc_y','period','playoffs','time_remaining', 'shot_distance',
           'shot_type','shot_zone_area','shot_zone_basic','game_date', 'home_away','opponent', 'shot_made_flag']

df = pd.DataFrame(index=shot_id, columns=columes)
# print(shot_id)
action_type_list = str_trans(action_type)
df['action_type'] = action_type_list
combined_list = str_trans(combined_shot_type)
df['combined_shot_type'] = combined_list
df['loc_x'] = loc_x.values
df['loc_y'] = loc_y.values
df['period'] = period.values
df['playoffs'] = playoffs.values

time_remaining = minutes_remaining.values * 60 + seconds_remaining.values
df['time_remaining'] = time_remaining

df['shot_distance'] = shot_distance.values
shot_type_list = str_trans(shot_type)
df['shot_type'] = shot_type_list
area_list = str_trans(shot_zone_area)
df['shot_zone_area'] = area_list
basic_list = str_trans(shot_zone_basic)
df['shot_zone_basic'] = basic_list

timedel = pd.to_datetime(game_date.values) - pd.to_datetime(game_date.min())
daydel = timedel.days.values
df['game_date'] = daydel

home_away = [0 if '@' in m else 1 for m in matchup]
df['home_away'] = home_away

opponent_list = str_trans(opponent)
df['opponent'] = opponent_list
df['shot_made_flag'] = shot_made_flag.values

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.head())
min_max_scaler = preprocessing.MinMaxScaler()
df_minmax = min_max_scaler.fit_transform(df)
#print(df_minmax)

df = pd.DataFrame(df_minmax, index=shot_id, columns=columes)
#print(df)


notnull = df['shot_made_flag'].notnull()
isnull = ~ notnull
train_set = df[notnull]
predict_set = df[isnull]

X_train = train_set.drop('shot_made_flag',axis=1).values
Y_train = train_set['shot_made_flag'].values
X_test = predict_set.drop('shot_made_flag',axis=1).values
Y_test = predict_set['shot_made_flag'].values

#clf_svm = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose=1)

clf_svm = LinearSVC(C=0.01, class_weight='balanced')
clf_svm.fit(X_train, Y_train)
predict_result = clf_svm.predict(X_test)
out = pd.DataFrame(index=predict_set.index, columns=['shot_made_flag'])
out['shot_made_flag'] = predict_result
out.to_csv('submission.csv')
print(out.head())

#print(predict_result)
def skGridSearchCv(X_train, X_test, y_train, y_test):
    param_grid = {"gamma": [0.01,0.1,1],
                 "C": [0.01,0.1,1]}
    print("Parameters:{}".format(param_grid))

    grid_search = GridSearchCV(SVC(),param_grid,cv=10)
    print(1)
    grid_search.fit(X_train, y_train)
    print(2)
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

def gridSearchCv(X_train, X_test, y_train, y_test):
    best_score = 0.0
    best_parameters = {}
    best_parameters['C'] = 0
    list = []
    #print(best_parameters)
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        svm = LinearSVC(C=C, max_iter=1000000, class_weight='balanced')
        print(C)
        scores = cross_val_score(svm, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
        #print(scores)
        score = scores.mean()
        print(score)
        list.append(score)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C}
    print("Best parameters:{}".format(best_parameters)) # 0.01
    return list

list = gridSearchCv(X_train, X_test, Y_train, Y_test)
# plt.plot(list)
# plt.show
