# -*- coding: UTF-8 -*-
import os
from util import *
import argparse
import csv

def main():

    parser = argparse.ArgumentParser(description='Print resutls.')
    parser.add_argument('-d', '--dataset_name', dest='dataset_name')
    parser.add_argument('-s', '--signature', dest='signature')
    parser.add_argument('-l', '--train_length', dest='train_length')
    parser.add_argument('-t', '--test_length', dest='test_length')
    parser.add_argument('-o', '--overlap', dest='overlap')
    # parser.add_argument('-os', '--original_setting', dest='original_setting')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    version = args.signature
    overlap = args.overlap
    if not overlap:
        overlap = '50'
    overlap = 1 - float(overlap)/100
    print (overlap)
    output_dir = 'data/'

    if not os.path.isdir(output_dir):
        print(output_dir + ' is not exist')
        print(os.getcwd())
        exit(1)

    # dataset = data_partition(output_dir + dataset_name + '.txt', original_setting=args.original_setting)
    dataset = data_partition(output_dir + dataset_name + '.txt', original_setting=True)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    length = int(args.train_length)
    test_length = int(args.test_length)

    # iCreate sessions for the train_tr set
    sequence_user_train_tr = {}
    sequence_id = 0
    for u in user_train:
        sequence_user_train_tr[u] = {}
        step = int(length*overlap)
        if step ==0:
            step = 1
        if len(user_train[u]) <= length:
            sequence_user_train_tr[u][sequence_id] = [(j,e) for e,j in enumerate(user_train[u][-length:])]
            sequence_id += 1
        else:
            for i in range(0, len(user_train[u]) - (length-step), step):
                curr_session = user_train[u][i:i+length]
                if len(curr_session) == length: 
                    sequence_user_train_tr[u][sequence_id] = [(j,i+e) for e,j in enumerate(curr_session)]
                else:
                    sequence_user_train_tr[u][sequence_id] = [(j,len(user_train[u])-length+e) for e,j in enumerate(user_train[u][-length:])]
                sequence_id += 1

    # Add item in validation to train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])  # user_train contains all interactions of each user, except the last one

    last_user_train = {}
    sequence_user_train = {}
    sequence_id = 0
    for u in user_train:
        sequence_user_train[u] = {}
        step = int(length*overlap)
        if step ==0:
            step = 1
        if len(user_train[u]) <= length:
            sequence_user_train[u][sequence_id] = [(j,e) for e,j in enumerate(user_train[u][-length:])]
            sequence_id += 1
        else:
            for i in range(0, len(user_train[u]) - (length-step), step):
                curr_session = user_train[u][i:i+length]
                if len(curr_session) == length: 
                    sequence_user_train[u][sequence_id] = [(j,i+e) for e,j in enumerate(curr_session)]
                else:
                    sequence_user_train[u][sequence_id] = [(j,len(user_train[u])-length+e) for e,j in enumerate(user_train[u][-length:])]
                sequence_id += 1
        if len(user_train[u]) <= (test_length):
            last_user_train[u] = [(j, e) for e,j in enumerate(user_train[u][-test_length:])]
        else:
            last_user_train[u] = [(j, len(user_train[u])-test_length+e) for e,j in enumerate(user_train[u][-test_length:])]

    for idx, u in enumerate(user_train):
        if idx < 10:
            print(user_train[u])
            print(user_valid[u])
            print(user_test[u])

    order_nums = {}
    output_filename = output_dir + 'sequence_aware_datasets/prepared/split_seq/' + dataset_name + "_" + version
    print('begin to save train_full data')
    print('train_full:{}'.format(output_filename))
    order_nums = write_data_csv(output_filename+'_train_full.txt', sequence_user_train, order_nums, True)  # write_data_csv(output_filename, data, order_nums, is_training_data)
    
    write_data_csv(output_filename+'_train_tr.txt', sequence_user_train_tr, order_nums, True)  

    write_data_csv(output_filename + '_train_valid.txt', last_user_train, order_nums, False)

    for u in user_train:
        if u in user_test:
            last_user_train[u].extend([(user_test[u][0], last_user_train[u][-1][1]+1)])  # user_train contains all interactions of each user, except the last one

    print('begin to save test data')
    print('test:{}'.format(output_filename))
    write_data_csv(output_filename + '_test.txt', last_user_train, order_nums, False)

    print('done.')


def write_data_csv(output_filename, data, order_nums, is_training_data):
    '''
    Write the result array to a csv file, if a result folder is defined in the configuration
        --------
        results : dict
            Dictionary of all results res[algorithm_key][metric_key]
        iteration; int
            Optional for the window mode
        extra: string
            Optional string to add to the file name
    '''

    print('start printing '+output_filename)
    with open(output_filename, 'w') as output_file:
        csv_writer = csv.writer(output_file)
        if is_training_data:  
            csv_writer.writerow(['UserId', 'SessionId', 'ItemId', 'Time'])
            for user, seq in data.items():
                #curr_order = 0
                for seq_id, items in seq.items():
                    for item, curr_order in items:
                        #curr_order += 1
                        csv_writer.writerow([user, seq_id, item, curr_order])
                        order_nums[user] = curr_order
        else:
            csv_writer.writerow(['UserId', 'ItemId', 'Time'])
            for user, items in data.items():
                #curr_order = 0  # for every user the Time of the first interaction will be 1
                # if user in order_nums:
                #     curr_order = order_nums[user]
                for item, curr_order in items:
                    #curr_order += 1
                    csv_writer.writerow([user, item, curr_order])

    print("printing is done for: "+output_filename)
    return order_nums


if __name__ == "__main__":
    main()
