# -*- coding: UTF-8 -*-
import os
from util_timestamp import *
import argparse
import csv

def main():

    parser = argparse.ArgumentParser(description='Print resutls.')
    parser.add_argument('-d', '--dataset_name', dest='dataset_name')
    parser.add_argument('-s', '--signature', dest='signature')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    version = args.signature
    output_dir = 'data/'

    if not os.path.isdir(output_dir):
        print(output_dir + ' is not exist')
        print(os.getcwd())
        exit(1)

    dataset = data_partition(output_dir + dataset_name + '.txt', original_setting=True)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    cc = 0.0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)

    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))

    print('len_train:{}, len_valid:{}, len_test:{}, usernum:{}, itemnum:{}'.
        format(
        len(user_train),
        len(user_valid), len(user_test), usernum, itemnum))

    for idx, u in enumerate(user_train):
        if idx < 10:
            print(user_train[u])
            print(user_valid[u])
            print(user_test[u])

    output_filename = output_dir + 'sequence_aware_datasets/prepared/timestamps/extended_test_set/' + dataset_name + "_" + version
    print('begin to save train data')
    print('train_valid:{}'.format(output_filename))
    write_data_csv(output_filename + '_train_tr.txt', user_train, True)

    # user_train_valid = user_train.copy()  # user_train_valid contains all interactions of each user, except the last two [BERT-KNN]
    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])  # user_train contains all interactions of each user, except the last one

    print('begin to save train_full data')
    print('train_full:{}'.format(output_filename))
    write_data_csv(output_filename+'_train_full.txt', user_train, True)  # write_data_csv(output_filename, data, order_nums, is_training_data)

    print('begin to save validation data')
    print('valid:{}'.format(output_filename))
    write_data_csv(output_filename + '_train_valid.txt', user_train, False)

    for u in user_train:
        if u in user_test:
            user_train[u].extend(user_test[u])  # user_train contains all interactions of each user, except the last one

    print('begin to save test data')
    print('test:{}'.format(output_filename))
    write_data_csv(output_filename + '_test.txt', user_train, False)

    print('done.')


def write_data_csv(output_filename, data, is_training_data):
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
        if is_training_data:  # train_full_data & train_valid_data
            csv_writer.writerow(['UserId', 'SessionId', 'ItemId', 'Time'])
        else:
            csv_writer.writerow(['UserId', 'ItemId', 'Time'])
        for user, items in data.items():
            if is_training_data:  # train_full_data & train_valid_data
                for item in items:
                    csv_writer.writerow([user, user, item[0], item[1]])
            else:  # valid_data & test_data
                for item in items:
                    csv_writer.writerow([user, item[0], item[1]])

    print("printing is done for: "+output_filename)


if __name__ == "__main__":
    main()
