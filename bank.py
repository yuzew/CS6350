# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:09:29 2022

@author: puyuz
"""

#%%
import pandas as pd
import DecisionTree as dt

# load train data
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = {'age': int, 
        'job': str, 
        'marital': str, 
        'education': str,
        'default': str,
        'balance': int,
        'housing': str,
        'loan': str,
        'contact': str,
        'day': int,
        'month': str,
        'duration': int,
        'campaign': int,
        'pdays': int,
        'previous': int,
        'poutcome': str,
        'y': str}
# load train data
train_data =  pd.read_csv('E:/cs6350/hw1/bank/train.csv', names=columns, dtype=types)
train_size = len(train_data.index)
## process data
# convert numeric to binary
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for c in numeric_features:
    median = train_data[c].median()
    train_data[c] = train_data[c].apply(lambda x: 0 if x < median else 1)
# replace unknowns
unknown_features = ['job', 'education', 'contact', 'poutcome']
for c in unknown_features:
    order = train_data[c].value_counts().index.tolist()
    if order[0] != 'unknown':
        replace = order[0]
    else:
        replace = order[1]
    train_data[c] = train_data[c].apply(lambda x: replace if x == 'unknown' else x)
#load test data
test_data =  pd.read_csv('E:/cs6350/hw1/bank/test.csv', names=columns, dtype=types)
test_size = len(test_data.index)
for c in numeric_features:
    median = test_data[c].median()
    test_data[c] = test_data[c].apply(lambda x: 0 if x < median else 1)

for c in unknown_features:
    order = test_data[c].value_counts().index.tolist()
    if order[0] != 'unknown':
        replace = order[0]
    else:
        replace = order[1]
    test_data[c] = test_data[c].apply(lambda x: replace if x == 'unknown' else x)

# set features and label
features = {'age': [0, 1],  # converted to binary
        'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], 
        'marital': ['married','divorced','single'], 
        'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],
        'balance': [0, 1],  # converted to binary
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [0, 1],  # converted to binary,
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1],  # converted to binary
        'campaign': [0, 1],  # converted to binary
        'pdays': [0, 1],  # converted to binary
        'previous': [0, 1],  # converted to binary
        'poutcome': ['unknown', 'other', 'failure', 'success']}
label = {'y': ['yes', 'no']}


train_acc = [[0 for x in range(16)] for y in range(3)]
test_acc = [[0 for x in range(16)] for y in range(3)]
acc = [[0 for x in range(6)] for y in range(16)]

for feature_selection in range(3):
    for max_depth in range(16):
        print('m:', feature_selection, ' d:', max_depth)
        # ID3
        dt_generator = dt.ID3(feature_selection=feature_selection, max_depth=max_depth+1)
        # get decision tree
        decision_tree = dt_generator.generate_decision_tree(train_data, features, label)
        # train acc
        # predict
        train_data['py']= dt_generator.classify(decision_tree, train_data)
        train_acc[feature_selection][max_depth] = train_data.apply(lambda row: 1 if row['y'] == row['py'] else 0, axis=1).sum() / train_size
        acc[max_depth][feature_selection * 2] = train_acc[feature_selection][max_depth]
        # test acc
        # predict
        test_data['py']= dt_generator.classify(decision_tree, test_data)
        test_acc[feature_selection][max_depth] = test_data.apply(lambda row: 1 if row['y'] == row['py'] else 0, axis=1).sum() / test_size
        acc[max_depth][feature_selection * 2 + 1] = test_acc[feature_selection][max_depth]


with open('bank_reuslts.txt', 'w') as f:
    f.write(str(train_acc))
    f.write(str(test_acc))
    f.write(str(acc))
    
print(acc)    
#print('train_acc:', train_acc,file='train_acc')
# print('test_acc:', test_acc)


