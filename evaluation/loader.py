import time
import numpy as np
import pandas as pd


def load_data(path, file, rows_train=None, rows_test=None, train_eval=False):
    '''
    Loads a tuple of training and test set with the given parameters.

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file.
        This option will automatically filter the test set to only retain items included in the training set.
    rows_test : int or None
        Number of rows to load from the test set file.
    slice_num :
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1).
        The result is cached for the execution of multiple experiments.
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)

    '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'

    density_appendix = ''

    if (rows_train == None):
        train = pd.read_csv(path + file + train_appendix + '.txt' + density_appendix, sep=',',
                            dtype={'ItemId': np.int64})
        # train = pd.DataFrame()
        # for chunk in pd.read_csv(path + file + train_appendix + '.txt' + density_appendix, sep=',',
        #                          dtype={'ItemId': np.int64}, chunksize=1000):
        #     train = pd.concat([train, chunk], ignore_index=True)
    else:
        train = pd.read_csv(path + file + train_appendix + '.txt' + density_appendix, sep=',',
                            dtype={'ItemId': np.int64}, nrows=rows_train)
        # train = pd.DataFrame()
        # for chunk in pd.read_csv(path + file + train_appendix + '.txt' + density_appendix, sep=',',
        #                     dtype={'ItemId': np.int64}, nrows=rows_train, chunksize=1000):
        #     train = pd.concat([train, chunk], ignore_index=True)
    if (rows_test == None):
        test = pd.read_csv(path + file + test_appendix + '.txt' + density_appendix, sep=',',
                           dtype={'ItemId': np.int64})
        # test = pd.DataFrame()
        # for chunk in pd.read_csv(path + file + test_appendix + '.txt' + density_appendix, sep=',',
        #                    dtype={'ItemId': np.int64}, chunksize=1000):
        #     test = pd.concat([test, chunk], ignore_index=True)
    else:
        test = pd.read_csv(path + file + test_appendix + '.txt' + density_appendix, sep=',',
                           dtype={'ItemId': np.int64}, nrows=rows_test)
        # test = pd.DataFrame()
        # for chunk in pd.read_csv(path + file + test_appendix + '.txt' + density_appendix, sep=',',
        #                    dtype={'ItemId': np.int64}, nrows=rows_test, chunksize=1000):
        #     test = pd.concat([test, chunk], ignore_index=True)

    test = test[np.in1d(test.ItemId, train.ItemId)]

    sequence_lengths = test.groupby('UserId').size()
    test = test[np.in1d(test.Order, sequence_lengths[sequence_lengths > 1].index)]

    test.sort_values(['UserId', 'Order'], inplace=True)
    train.sort_values(['UserId', 'Order'], inplace=True)

    # output

    print('Loaded train set\n\tEvents: {}\n\tUsers: {}\n\tItems: {}\n'.
          format(len(train), train.UserId.nunique(), train.ItemId.nunique()))

    print('Loaded test set\n\tEvents: {}\n\tUsers: {}\n\tItems: {}\n'.
          format(len(test), test.UserId.nunique(), test.ItemId.nunique()))

    # data_start = datetime.fromtimestamp(train.Time.min(), timezone.utc)
    # data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    #
    # print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
    #       format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(),
    #              data_end.date().isoformat()))
    #
    # data_start = datetime.fromtimestamp(test.Time.min(), timezone.utc)
    # data_end = datetime.fromtimestamp(test.Time.max(), timezone.utc)
    #
    # print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
    #       format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(),
    #              data_end.date().isoformat()))

    check_data(train, test)

    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)

def load_data_session(path, file, train_eval=False):
    '''
    Loads a tuple of training and test set with the given parameters.

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file.
        This option will automatically filter the test set to only retain items included in the training set.
    rows_test : int or None
        Number of rows to load from the test set file.
    slice_num :
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1).
        The result is cached for the execution of multiple experiments.
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)

    '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'

    train = pd.read_csv(path + file + train_appendix + '.txt', sep=',', dtype={'ItemId': np.int64})
    # train = pd.DataFrame()
    # for chunk in pd.read_csv(path + file + train_appendix + '.txt', sep=',', dtype={'ItemId': np.int64}, chunksize=1000):
    #     train = pd.concat([train, chunk], ignore_index=True)
    test = pd.read_csv(path + file + test_appendix + '.txt', sep=',', dtype={'ItemId': np.int64})
    # test = pd.DataFrame()
    # for chunk in pd.read_csv(path + file + test_appendix + '.txt', sep=',', dtype={'ItemId': np.int64}, chunksize=1000):
    #     test = pd.concat([test, chunk], ignore_index=True)

    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)


def load_buys(path, file):
    '''
    Load all buy events from the youchoose file, retains events fitting in the given test set and merges both data sets into one

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt

    Returns
    --------
    out : pandas.DataFrame
        test with buys

    '''

    print('START load buys')
    st = time.time()
    sc = time.clock()

    # load csv
    buys = pd.read_csv(path + file + '.txt', sep=',', dtype={'ItemId': np.int64})

    print('END load buys ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return buys


def check_data(train, test):
    if 'ItemId' in train.columns and 'UserId' in train.columns:

        new_in_test = set(test.ItemId.unique()) - set(train.ItemId.unique())
        if len(new_in_test) > 0:
            print('WAAAAAARRRNIIIIING: new items in test set')

        session_min_train = train.groupby('UserId').size().min()
        if session_min_train == 0:
            print('WAAAAAARRRNIIIIING: sequence length 1 in train set')

        session_min_test = test.groupby('UserId').size().min()
        if session_min_test == 0:
            print('WAAAAAARRRNIIIIING: sequence length 1 in train set')

        users_train = train.UserId.unique()
        users_test = test.UserId.unique()

        if not all(users_train[i] <= users_train[i + 1] for i in range(len(users_train) - 1)):
            print('WAAAAAARRRNIIIIING: train users not sorted by id')
            if 'Time' in train.columns:
                train.sort_values(['UserId', 'Time'], inplace=True)
            else:
                train.sort_values(['UserId', 'Order'], inplace=True)
            print(' -- corrected the order')

        if not all(users_test[i] <= users_test[i + 1] for i in range(len(users_test) - 1)):
            print('WAAAAAARRRNIIIIING: test users not sorted by id')
            if 'Time' in test.columns:
                test.sort_values(['UserId', 'Time'], inplace=True)
            else:
                test.sort_values(['UserId', 'Order'], inplace=True)
            print(' -- corrected the order')

    else:
        print('data check not possible due to individual column names')


def rename_cols(df):
    names = {}

    names['order'] = 'Order'

    names['item_id'] = 'ItemId'
    names['user_id'] = 'UserId'
    names['created_at'] = 'Time'

    names['itemId'] = 'ItemId'
    names['userId'] = 'UserId'
    names['eventdate'] = 'Time'

    names['itemid'] = 'ItemId'
    names['visitorid'] = 'UserId'
    names['timestamp'] = 'Time'

    names['product_id'] = 'ItemId'
    names['user_id'] = 'UserId'
    names['event_time'] = 'Time'

    for col in list(df.columns):
        if col in names:
            df[names[col]] = df[col]
            del df[col]

    return df