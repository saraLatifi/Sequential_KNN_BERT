from __future__ import print_function
from collections import defaultdict


def data_partition(fname, original_setting):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    train_elems = {}
    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append((i,t))


    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]  # training set: all interactions of each user, except the last two
            user_valid[user] = []
            user_valid[user].append(User[user][-2])  # validation set: the second last interactions of each user
            user_test[user] = []
            user_test[user].append(User[user][-1])  # test set: the last interactions of each user

        if not original_setting:
            # Add all elements in train and validation to train_elems
            for i in user_train[user]+user_valid[user]:
                train_elems[i] = 1

    if not original_setting:
        # Remove test information for the users if item not in train_elems
        for user in User:
            if len(user_test[user]) > 0:
                if user_test[user][0] not in train_elems:
                    #del user_test[user]  # 
                    user_test[user] = []

    return [user_train, user_valid, user_test, usernum, itemnum]
