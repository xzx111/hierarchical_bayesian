#!/bin/env python
#-*-coding:utf8-*-
import sys
import os
import random
import numpy as np

def obtain_data(file, brand_min=0):
    """
    :param file:  the path of data file
    :param fi:  feature information,
      fi=0: popularity
      fi=1: cid + popularity
      fi=2: cid + popularity + title
      fi=3: cid + popularity + title + attributes
    :param brand_min: the minimum number of exposure for each brand
    :return:   X, r, User_id, Brand_id, d, U, B
    """

    r = []
    user_uniq_name = []
    user_name = []

    sku_uniq_name = []
    sku_name = []

    X = []

    brand_uniq_name = []
    brand_name = []

    s = ''
    with open(file, 'rb') as f:
        for case in f.readlines():
            case = case.decode('UTF-8')
            if case == s:
                continue
            s = case

            case = case.strip().split('\t')

            if case[0] not in user_uniq_name:
                user_uniq_name += [case[0]]
            user_name += [case[0]]

            if case[1] not in sku_uniq_name:
                sku_uniq_name += [case[1]]
            sku_name += [case[1]]

            if case[2] == 'expose':
                r += [0]
            else:
                r += [1]

            if case[6] not in brand_uniq_name:
                brand_uniq_name += [case[6]]
            brand_name += [case[6]]

            #if fi == 0:
            #    X += [[np.log(float(case[-5]))] + [np.log(float(case[-4]))]]
            #elif fi == 1:
            #    #pop_cid =
            #    X += [[float(x) for x in case[-6].strip('[]').split(', ')] + \
            #          [np.log(float(case[-5]))] + [np.log(float(case[-4]))]]
            #elif fi == 2:
            #    pop_cid = -np.log(float(case[-3]))
            #    X += [[float(x) for x in case[-6].strip('[]').split(', ')] + \
            #          [np.log(float(case[-5]))] + [np.log(float(case[-4]))] + \
            #          [float(x) for x in case[4].split()]]
            #else:
            #    pop_cid = -np.log(float(case[-3]))
            #X += [[np.log(float(case[-5]))] + [np.log(float(case[-4]))] + \
            #        [float(x) for x in case[-6].strip('[]').split(', ')] + \
            #        [float(x) for x in case[4].split()] + \
            #        [float(x) for x in case[-1].strip('[]').split(', ')]]
            X += [[float(x) for x in case[4].split()] + \
                  [float(x) for x in case[-1].strip('[]').split(', ')]]



    User_id = np.array([user_uniq_name.index(x) for x in user_name])
    Brand_id = np.array([brand_uniq_name.index(x) for x in brand_name])
    Sku_id = np.array([sku_uniq_name.index(x) for x in sku_name])
    X = np.array(X)
    r = np.array(r)

    brand_name = np.array(brand_name)
    if brand_min!=0:
        delete_ind = []
        delete_brand = []
        for brand in brand_uniq_name:
            brand_index = np.where(brand_name == brand)[0].tolist()
            if len(brand_index)<brand_min:
                delete_ind += brand_index
                delete_brand += [brand]
                brand_uniq_name.remove(brand)

        X = np.delete(X, delete_ind, 0)
        r = np.delete(r, delete_ind)
        User_id = np.delete(User_id, delete_ind)
        Sku_id = np.delete(Sku_id, delete_ind)

        brand_name = np.delete(brand_name, delete_ind)
        Brand_id = np.array([brand_uniq_name.index(x) for x in brand_name])

    return X, r, User_id, Brand_id, Sku_id, X.shape[1], len(user_uniq_name), len(brand_uniq_name), len(sku_uniq_name)

def sample_train_data(X, r, User_id, Brand_id, Sku_id, U, prop=0.8):
    train_ind = []
    for i in range(U):
        id_i = np.where(User_id == i)[0]
        n_i = len(id_i)
        if n_i<40:
            train_ind += id_i.tolist()
        else:
            train_ind += np.random.choice(id_i, int(round(n_i*prop)), replace=False).tolist()

    train_ind = np.random.permutation(train_ind)
    X_train = X[train_ind, :]
    r_train = r[train_ind]
    User_id_train = User_id[train_ind]
    Brand_id_train = Brand_id[train_ind]
    Sku_id_train = Sku_id[train_ind]

    X_test = np.delete(X, train_ind, 0)
    r_test = np.delete(r, train_ind)
    User_id_test = np.delete(User_id, train_ind)
    Brand_id_test = np.delete(Brand_id, train_ind)
    Sku_id_test = np.delete(Sku_id, train_ind)

    return X_train, r_train, User_id_train, Brand_id_train, Sku_id_train, X_test, r_test, User_id_test, Brand_id_test, Sku_id_test

def sample_data_out(X, r, User_id, Brand_id, Sku_id, U, prop=0.8):
    train_ind = []
    nsku = len(np.unique(Sku_id))
    sku_train_id = np.random.permutation(range(nsku))[:round(nsku*prop)]
    for i in sku_train_id:
        id_i = np.where(Sku_id == i)[0]
        train_ind += id_i.tolist()

    #train_ind = np.random.permutation(train_ind)
    X_train = X[train_ind, :]
    r_train = r[train_ind]
    User_id_train = User_id[train_ind]
    Brand_id_train = Brand_id[train_ind]
    Sku_id_train = Sku_id[train_ind]

    X_test = np.delete(X, train_ind, 0)
    r_test = np.delete(r, train_ind)
    User_id_test = np.delete(User_id, train_ind)
    Brand_id_test = np.delete(Brand_id, train_ind)
    Sku_id_test = np.delete(Sku_id, train_ind)

    return X_train, r_train, User_id_train, Brand_id_train, Sku_id_train, X_test, r_test, User_id_test, Brand_id_test, Sku_id_test

def splitData(X, r, User_id, Brand_id, Sku_id, prop=0.8):
    print('load dataset')
    perm = np.random.permutation(len(X))
    N_train = int(np.floor(len(X) * prop))

    X_train, X_test = np.split(X[perm],   [N_train])
    r_train, r_test = np.split(r[perm], [N_train])
    User_id_train, User_id_test = np.split(User_id[perm], [N_train])
    Brand_id_train, Brand_id_test = np.split(Brand_id[perm], [N_train])
    Sku_id_train, Sku_id_test = np.split(Sku_id[perm], [N_train])
    return X_train, r_train, User_id_train, Brand_id_train, Sku_id_train, X_test, r_test, User_id_test, Brand_id_test, Sku_id_test
