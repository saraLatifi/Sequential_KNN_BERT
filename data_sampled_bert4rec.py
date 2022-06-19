# -*- coding: UTF-8 -*-
import os
from util import *
import argparse
import csv
import random
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description='Print results.')
    parser.add_argument('-d', '--dataset_name', dest='dataset_name')
    parser.add_argument('-s', '--signature', dest='signature')  # percentage of sampled users
    args = parser.parse_args()

    dataset_name = args.dataset_name
    signature = args.signature
    sample_percentage = int(signature)

    output_dir = 'data/'
    data = pd.read_csv(output_dir + dataset_name + '.txt', sep=' ')

    users = list(set(data.iloc[:, 0]))  # first column: user
    random.seed(10)
    sample_size = len(users) * (sample_percentage / 100)
    users_sampled = random.sample(users, int(sample_size))
    data = data[data.iloc[:, 0].isin(users_sampled)]

    output_filename = output_dir + dataset_name + "_sampled_" + signature + '.txt'
    print('begin to save sampled data')

    with open(output_filename, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=' ')
        for index, row in data.iterrows():
            user = row[0]
            item = row[1]
            csv_writer.writerow([user, item])

if __name__ == "__main__":
    main()
