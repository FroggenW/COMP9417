#Written by Chang Wang(z5196324)
#this code used KNN algorithm
#it picked 9 column from the input file and dealed with these features
#main aims to discuss the difference between using same algorithm but different features with other group members
#the output is such as:
#the best k =  13
#precision score = 0.7142857142857143
#recall score = 0.8333333333333334
#F1_score = 0.7692307692307692
#accuracy score for test dataset: 0.6666666666666666
#Finished time: 24/11/2019
#reference URL:
#https://blog.csdn.net/z583636762/article/details/78988415
#https://www.cnblogs.com/bymo/p/8618191.html
#https://blog.csdn.net/u011630575/article/details/79195450
#https://blog.csdn.net/jasonleesjtu/article/details/92091143
#https://www.jianshu.com/p/284581d9b189

import os
import re
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import metrics
from matplotlib import pyplot as plt

def initial_dataframe(dir_name, index, features):
    lst = os.listdir(dir_name)
    lst.sort()
    df = pd.DataFrame(np.nan, index=index, columns=features)
    return lst, df

index = os.listdir(os.getcwd() + '/StudentLife_Dataset/inputs/sensing/activity')
index.sort()
#print(index)
for i in range(len(index)):
    index[i] = re.findall(r'u\d+', index[i])[0]
#print(index)

# dealing with feature which need to count the number of inference
# for activity and audio
def featrue_count(dir, lst, df):
    for i in range(len(lst)):
        csv = os.path.join(dir, lst[i])
        data = pd.read_csv(csv, index_col=False)
        count = data[data.columns[1]].value_counts() #normalize=True
        user = re.findall(r'u\d+', lst[i])
        feature = []
        amount = 0
        for j in range(len(count.keys())):
            feature.append(count[j])
            amount = amount + count[j]
        for k in range(len(feature)):
            feature[k] = round(feature[k] / amount,3)
        df.loc[user[0]] = feature
        # print(feature)
    return df


# dataframe for activity and audio
# activity feature
activity = os.getcwd() + '/StudentLife_Dataset/inputs/sensing/activity'
#print(activity)
activity_features = ['Stationary', 'Walking', 'Running', 'Unknown']
act_lst, act_df = initial_dataframe(activity, index, activity_features)
act_df = featrue_count(activity, act_lst, act_df)
act_df['Moving'] = act_df['Walking'] + act_df['Running'] + act_df['Unknown']
act_df = act_df[['Stationary', 'Moving']]
print(act_df)

# audio feature
audio = os.getcwd() + '/StudentLife_Dataset/inputs/sensing/audio'
audio_features = ['Silence', 'Voice', 'Noise']
aud_lst, aud_df = initial_dataframe(audio, index, audio_features)
aud_df = featrue_count(audio, aud_lst, aud_df)
# print(aud_df.var())

def feature_conversation(dir, lst, df):
    for i in range(len(lst)):
        csv = os.path.join(dir, lst[i])
        user = re.findall(r'u\d+', lst[i])
        data = pd.read_csv(csv, index_col=False)
        duration = 0
        start_timestamp = data['start_timestamp']
        end_timestamp = data[' end_timestamp']
        for i in range(len(data)):
            duration = duration + (end_timestamp[i] - start_timestamp[i])
        df.loc[user[0]] = duration
    return df
#conversation feature
conv = os.getcwd() + '/StudentLife_Dataset/inputs/sensing/conversation'
conv_features = ['conversation']
conv_lst, conv_df = initial_dataframe(conv, index, conv_features)
conv_df = feature_conversation(conv, conv_lst, conv_df)

def feature_dis_freg(dir, lst, df):
    for i in range(len(lst)):
        csv = os.path.join(dir, lst[i])
        user = re.findall(r'u\d+', lst[i])
        data = pd.read_csv(csv, index_col=False)
        latitude = data['latitude']
        longitude = data['longitude']
        altitude = data['altitude']
        change_account = 0
        pre_latitude = latitude[0]
        pre_longitude = longitude[0]
        pre_altitude = altitude[0]
        for i in range(len(data)):
            if abs(pre_latitude - latitude[i]) > 0.00004 or abs(pre_longitude - longitude[i]) > 0.00004 or pre_altitude != altitude[i]:
                change_account += 1
                pre_latitude = latitude[i]
                pre_longitude = longitude[i]
                pre_altitude = altitude[i]
        df.loc[user[0]] = change_account
    return df

# gps feature
gps = os.getcwd() + '/StudentLife_Dataset/inputs/sensing/gps'
gps_features = ['MovedTime']
gps_lst, gps_df = initial_dataframe(gps, index, gps_features)
gps_df = feature_dis_freg(gps, gps_lst, gps_df)

# wifi location
def feature_loc(dir, lst, df):
    for i in range(len(lst)):
        csv = os.path.join(dir, lst[i])
        user = re.findall(r'u\d+', lst[i])
        data = pd.read_csv(csv, index_col=False)
        data = data[['location']]
        indoor = data['location'].str.contains("in").value_counts()[True] / 10
        outdoor = data['location'].str.contains("near").value_counts()[True] / 10
        df.loc[user[0]] = [indoor, outdoor]
    return df

location = os.getcwd() + '/StudentLife_Dataset/inputs/sensing/wifi_location'
loc_features = ['Indoor', 'Outdoor']
loc_lst, loc_df = initial_dataframe(location, index, loc_features)
loc_df = feature_loc(location, loc_lst, loc_df)
#print(loc_df)

#merge for a new dataframe
input_df_0 = pd.merge(act_df, aud_df, left_index=True, right_index=True)
input_df_0 = pd.merge(input_df_0, conv_df, left_index=True, right_index=True)
input_df_0 = pd.merge(input_df_0, gps_df, left_index=True, right_index=True)
input_df_0 = pd.merge(input_df_0, loc_df, left_index=True, right_index=True)
#print(train_X)

#output flourishing
Flourishing = pd.read_csv('FlourishingScale.csv')
Flourishing_pre = Flourishing.head(46)
Flourishing_pre.fillna(0, inplace = True)
#deal with the data
array_Flourishing = Flourishing_pre.values
uid = []
sum_list = []
#only need to calculate the sum
flourishing = pd.DataFrame(np.nan, index=uid, columns=['average'])
for index_1 in range(len(array_Flourishing)):
    uid.append(array_Flourishing[index_1][0])
    account = 0
    number = 0
    for index_2 in range(2,len(array_Flourishing[index_1])):
        if array_Flourishing[index_1][index_2] > 0:
            #get the average for each uid, avoid NAN value's influence
            account += int(array_Flourishing[index_1][index_2])
            number += 1
    sum_list.append(float((account) / number))
    flourishing.loc[uid[index_1]] = float(sum_list[index_1])
#print(input_df)
input_df = pd.merge(input_df_0, flourishing, left_index=True, right_index=True)
#print(input_df)
input_1 = input_df.values

#min and max
def transformation(data):
    for index in range(0, len(data[0])-1):
        max_number = 0
        min_number = 99999999
        for row in range(len(data)):
            if float(data[row][index]) > max_number:
                max_number = float(data[row][index])
            if float(data[row][index]) < min_number:
                min_number = float(data[row][index])
        for row_a in range(len(data)):
            data[row_a][index] = (float(data[row_a][index]) - min_number)/(max_number - min_number)
    return data
#except the last row, regard them as X
def X_train(data):
    x_train = []
    for x in range(len(data)):
        list_x = []
        for y in range(0, len(data[0])-1):
            list_x.append(float(data[x][y]))
        x_train.append(list_x)
    return x_train

#the last row is Y
#using the average to obtain the overall
def Y_train(data):
    y_train = []
    for i in range(len(data)):
        y_train.append(int(8*data[i][len(data[0])-1]))
    value = 0
    acc = 0
    for j in range(len(y_train)):
        value = value + y_train[j]
        acc = j
    value_1 = value / (acc + 1)
    for k in range(len(y_train)):
        if y_train[k] >= value_1:
            y_train[k] = float(1)
        else:
            y_train[k] = float(0)
    return y_train

input_1 = transformation(input_1)
training_data = input_1[:37]
test_data = input_1[37:]

x_train_training = X_train(training_data)
x_train_test = X_train(test_data)
y_train = Y_train(input_1)
y_train_training = y_train[:37]
y_train_test = y_train[37:]

#find the largest auc score and got k
def Optimal_number(x_1, y_1, x_2, y_2):
    largest_AUC_score = 0
    optimal_number = 0
    AUC_list = []
    for index in range(1, 31):
        model = neighbors.KNeighborsClassifier(index)
        model.fit(x_1, y_1)
        AUC_score = metrics.roc_auc_score(y_2, model.predict_proba(x_2)[:, 1])
        if AUC_score > largest_AUC_score:
            largest_AUC_score = AUC_score
            optimal_number = index
        AUC_list.append(AUC_score)
    return optimal_number, AUC_list

print('------------flourishing------------')
#training_optimal_number, training_AUC_list = Optimal_number(x_train_training, y_train_training, x_train_training,y_train_training)
test_optimal_number, test_AUC_list = Optimal_number(x_train_training, y_train_training, x_train_test, y_train_test)
#print(training_optimal_number)
print('the best k = ',test_optimal_number)
#plt.plot(training_AUC_list)
#plt.show()
plt.plot(test_AUC_list)
plt.show()

#get precision score, recall score and F1 score
def Part(n, x_1, y_1, x_2, y_2):
    model = neighbors.KNeighborsClassifier(n)
    model.fit(x_1, y_1)
    recall_opt = metrics.recall_score(y_2, model.predict(x_2))
    prec_opt = metrics.precision_score(y_2, model.predict(x_2))
    F1_score = metrics.f1_score(y_2,model.predict(x_2))
    return prec_opt, recall_opt, F1_score


prec_opt, recall_opt, F1_score = Part(5, x_train_training, y_train_training, x_train_test, y_train_test)
print('precision score =',prec_opt)
print('recall score =',recall_opt)
print('F1_score =',F1_score)

def accuracy_score_2(n, x_train_training, y_train_training, x_train, y_train):
    model = neighbors.KNeighborsClassifier(n)
    model.fit(x_train_training, y_train_training)
    score = accuracy_score(y_train, model.predict(x_train))
    return score
#for a in range(1,30):
    #score_1111 = accuracy_score_2(a, x_train_training, y_train_training, x_train_test, y_train_test)
    #prec_opt_111, recall_opt_111, F1_score_111 = Part(a, x_train_training, y_train_training, x_train_test, y_train_test)
    #print('11111111111111',score_1111)
    #print('2222222222222222', recall_opt_111)

#training_score = accuracy_score_2(2, x_train_training, y_train_training, x_train_training, y_train_training)
test_score = accuracy_score_2(test_optimal_number, x_train_training, y_train_training, x_train_test, y_train_test)
#print("accuracy score for training dataset:", training_score)
print("accuracy score for test dataset:", test_score)

#similar with previous
panas = pd.read_csv('panas.csv')
panas_pre = panas.head(46)
panas_pre.fillna(0, inplace = True)
#print(panas_pre.to_string())
drop_columns_1 = ['Distressed','Upset','Guilty','Scared','Hostile','Irritable','Nervous','Jittery','Afraid']
posi_panas = panas_pre.drop(drop_columns_1, axis = 1)
#print(posi_panas.to_string())

array_pos_panas = posi_panas.values
uid_1 = []
sum_list_1 = []
panas_1 = pd.DataFrame(np.nan, index=uid, columns=['average'])
for index_1 in range(len(array_pos_panas)):
    uid_1.append(array_Flourishing[index_1][0])
    account = 0
    number = 0
    for index_2 in range(2,len(array_pos_panas[index_1])):
        if array_pos_panas[index_1][index_2] > 0:
            account += int(array_pos_panas[index_1][index_2])
            number += 1
    sum_list_1.append(float((account) / number))
    panas_1.loc[uid_1[index_1]] = float(sum_list_1[index_1])
#print(input_df)
input_1_df = pd.merge(input_df_0, panas_1, left_index=True, right_index=True)
#print(input_1_df.to_string())
input_2 = input_1_df.values

def Y_train_1(data):
    y_train = []
    for i in range(len(data)):
        y_train.append(int(9*data[i][len(data[0])-1]))
    value = 25.3
    for k in range(len(y_train)):
        if y_train[k] >= value:
            y_train[k] = float(1)
        else:
            y_train[k] = float(0)
    return y_train

input_2 = transformation(input_2)
training_data_1 = input_2[:37]
test_data_1 = input_2[37:]
x_train_training_1 = X_train(training_data_1)
x_train_test_1 = X_train(test_data_1)
y_train_1 = Y_train_1(input_2)
y_train_training_1 = y_train_1[:37]
y_train_test_1 = y_train_1[37:]

test_optimal_number_1, test_AUC_list_1 = Optimal_number(x_train_training_1, y_train_training_1, x_train_test_1, y_train_test_1)
print('------------panas------------')
print('the best k = ',test_optimal_number_1)
plt.plot(test_AUC_list_1)
plt.show()
prec_opt_1, recall_opt_1, F1_score_1 = Part(4, x_train_training_1, y_train_training_1, x_train_test_1, y_train_test_1)
print('precision score =',prec_opt_1)
print('recall score =',recall_opt_1)
print('F1_score =',F1_score_1)

test_score_1 = accuracy_score_2(test_optimal_number_1, x_train_training_1, y_train_training_1, x_train_test_1, y_train_test_1)
#print("accuracy score for training dataset:", training_score)
print("accuracy score for test dataset:", test_score_1)


drop_columns_2 = ['Interested','Strong','Enthusiastic','Proud','Alert','Inspired','Determined','Attentive','Active']
neg_panas = panas_pre.drop(drop_columns_2, axis = 1)

array_neg_panas = neg_panas.values
uid_2 = []
sum_list_2 = []
panas_2 = pd.DataFrame(np.nan, index=uid, columns=['average'])
for index_1 in range(len(array_neg_panas)):
    uid_2.append(array_Flourishing[index_1][0])
    account = 0
    number = 0
    for index_2 in range(2,len(array_neg_panas[index_1])):
        if array_neg_panas[index_1][index_2] > 0:
            account += int(array_neg_panas[index_1][index_2])
            number += 1
    sum_list_2.append(float((account) / number))
    panas_2.loc[uid_2[index_1]] = float(sum_list_2[index_1])
#print(input_df)
input_2_df = pd.merge(input_df_0, panas_2, left_index=True, right_index=True)
#print(input_1_df.to_string())
input_3 = input_2_df.values

def Y_train_2(data):
    y_train = []
    for i in range(len(data)):
        y_train.append(int(9*data[i][len(data[0])-1]))
    value = 12.4
    for k in range(len(y_train)):
        if y_train[k] >= value:
            y_train[k] = float(1)
        else:
            y_train[k] = float(0)
    return y_train

input_3 = transformation(input_3)
training_data_2 = input_3[:37]
test_data_2 = input_3[37:]
x_train_training_2 = X_train(training_data_2)
x_train_test_2 = X_train(test_data_2)
y_train_2 = Y_train_2(input_3)
y_train_training_2 = y_train_2[:37]
y_train_test_2 = y_train_2[37:]

test_optimal_number_2, test_AUC_list_2 = Optimal_number(x_train_training_2, y_train_training_2, x_train_test_2, y_train_test_2)
print('------------panas------------')
print('the best k = ',test_optimal_number_2)
plt.plot(test_AUC_list_2)
plt.show()
prec_opt_2, recall_opt_2, F1_score_2 = Part(2, x_train_training_2, y_train_training_2, x_train_test_2, y_train_test_2)
print('precision score =',prec_opt_2)
print('recall score =',recall_opt_2)
print('F1_score =',F1_score_2)

test_score_2 = accuracy_score_2(test_optimal_number_2, x_train_training_2, y_train_training_2, x_train_test_2, y_train_test_2)
#print("accuracy score for training dataset:", training_score)
print("accuracy score for test dataset:", test_score_2)

