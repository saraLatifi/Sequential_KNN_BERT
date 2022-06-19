import csv
import json
import ast
import time
import datetime


def proprocess(input_file, output_file, with_repetitions=False, first_item=True):
    f = open(input_file)
    username_id = {}
    a = {}
    b = {}
    """
    for i in f.readlines():
        review = ast.literal_eval(i)
        if "user_id" in review:
            username_id[review['username']] = review['user_id']
    f = open(input_file)
    """
    for i in f.readlines():
        #print (i)
        review = ast.literal_eval(i)
        #review = json.loads(i)
        user_id = "username"
        """
        user_id = "user_id"
        if "user_id" not in review:
            if review['username'] in username_id:
                review['user_id'] = username_id[review['username']]
            else:
                user_id = "username"
        else:
            username_id[review['username']] = review['user_id']
        """
        if review[user_id] not in a:
            a[review[user_id]] = []
        #d = review['date']
        d = time.mktime(datetime.datetime.strptime(review['date'],"%Y-%m-%d").timetuple())
        ignore = False
        if not with_repetitions:
            copy_items = []
            for j in a[review[user_id]]:
                if j[0] == review['product_id']:
                    if first_item:
                        ignore = True
                else:
                    copy_items.append(j)
            if first_item == False:
                a[review[user_id]] = copy_items
        if with_repetitions or not ignore:
            a[review[user_id]].append((review['product_id'], int(d)))
            if review['product_id'] not in b:
                b[review['product_id']] = []
            b[review['product_id']].append(review[user_id])

    for i in b.keys():
        if len(b[i]) < 5:
            for j in b[i]:
                old_list = a[j]
                new_list = []
                for k in old_list:
                    if k[0] != i:
                        new_list.append(k)
                a[j] = new_list

    users = {}
    items = {}
    with open(output_file, mode='w', newline='') as w_file:
        csv_writer = csv.writer(w_file, delimiter=',')

        for i in a.keys():
            if len(a[i]) >= 5:
                for k,t in sorted(a[i], key=lambda x: x[1]):
                    if len(b[k]) >= 5:
                        if k not in items:
                            items[k] = len(items)+1
                        if i not in users:
                            users[i] = len(users)+1
                        csv_writer.writerow([users[i], items[k], t])
                        #csv_writer.writerow([users[i], items[k]])


if __name__ == "__main__":
    f = 'data/raw/steam_new.json'
    output_file = 'data/steam_timestamps.txt'
    with_repetitions = False
    # output_file = 'data/steam_timestamps_repeated.txt'
    # with_repetitions = True
    proprocess(f, output_file, with_repetitions)
