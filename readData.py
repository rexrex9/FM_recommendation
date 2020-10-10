#https://grouplens.org/datasets/movielens/100k/

import os
import numpy as np

base_path='data/ml-100k'
train_path = os.path.join(base_path,'ua.base')
test_path = os.path.join(base_path,'ua.test')
user_path = os.path.join(base_path,'u.user')
item_path = os.path.join(base_path,'u.item')
occupation_path = os.path.join(base_path,'u.occupation')


def get1or0(r):
    return 1.0 if r>3 else 0.0

def __read_rating_data(path):
    dataSet={}
    with open(path,'r') as f:
        for line in f.readlines():
            d=line.strip().split('\t')
            dataSet[(int(d[0]),int(d[1]))]=[get1or0(int(d[2]))]
    return dataSet

def __read_item_hot():
    items={}
    with open(item_path,'r',encoding='ISO-8859-1') as f:
        for line in f.readlines():
            d=line.strip().split('|')
            items[int(d[0])]=np.array(d[5:],dtype='float64')
    return items


def __read_occupation_hot():
    occupations = {}
    with open(occupation_path,'r') as f:
        names=f.read().strip().split('\n')
    length=len(names)
    for i in range(length):
        l=np.zeros(length,dtype='float64')
        l[i]=1
        occupations[names[i]]=l
    return occupations

def __read_user_hot():
    users={}
    gender_dict={'M':1,'F':0}
    occupation_dict = __read_occupation_hot()
    with open(user_path,'r') as f:
        for line in f.readlines():
            d=line.strip().split('|')
            a=np.array([int(d[1]), gender_dict[d[2]]])
            users[int(d[0])]=np.append(a,occupation_dict[d[3]])
    return users

def read_dataSet(user_dict,item_dict,path):
    X, Y = [], []
    ratings = __read_rating_data(path)
    for k in ratings:
        X.append(np.append(user_dict[k[0]], item_dict[k[1]]))
        Y.append(ratings[k])
    return X,Y


def read_data():
    user_dict = __read_user_hot()
    item_dict=__read_item_hot()
    trainX,trainY=read_dataSet(user_dict,item_dict,train_path)
    testX,testY=read_dataSet(user_dict,item_dict,test_path)

    return trainX,trainY,testX,testY

if __name__ == '__main__':
    trainX, trainY, testX, testY=read_data()
    print(trainX[:5])


    print(len(trainX[0]))
