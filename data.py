
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')

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
shot_zone_area = data['shot_zone_area']
shot_zone_basic = data['shot_zone_basic']
shot_zone_range = data['shot_zone_range']
team_id = data['team_id']
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

notnull = df['shot_made_flag'].notnull()
isnull = ~ notnull
train_set = df[notnull]
predict_set = df[isnull]
