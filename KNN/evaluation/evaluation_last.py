import time
import numpy as np
import statistics  # DEBUG
import os  # DEBUG

from evaluation.evaluation import Evaluation
from evaluation.dataloader import Dataloader
    
def evaluate_sequences(pr, test_data, train_data, items=None, negative_sampling=True, use_pop_random=True, negative_sample_size=100, user_key='UserId', item_key='ItemId', time_key='Time', pred_time=None, conf=None, key=None):
    '''
    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : list of tuples
        (metric_name, value)
    
    '''
    
    actions = len(test_data)
    sequences = len(test_data[user_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sequences, ' sequences')

    st = time.time()
    # scalability
    pred_time.reset()

    export_csv = 'results/statistical_significance/' + conf['data']['name'] + '/all_preds_' + conf['key'] + '_' + conf['data']['name'] + '.csv'
    eval = Evaluation(negative_sampling=negative_sampling, use_pop_random=use_pop_random, negative_sample_size=negative_sample_size, export_csv=export_csv)

    # data = Dataloader('bert4rec',
    #                   user_history_filename="../data/sequence_aware_datasets/prepared/ml-1m-mp1.0-sw0.5-mlp0.2-df10-mpps40-msl200.his",
    #                                          vocab_filename="../data/sequence_aware_datasets/prepared/ml-1m-mp1.0-sw0.5-mlp0.2-df10-mpps40-msl200.vocab")
    data = Dataloader('baseline', train_data=train_data, test_data=test_data, item_key=item_key)  # data = Dataloader('baseline', train_data=test_data, item_key=item_key)

    test_data.sort_values([user_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()  # items_to_predict = test_data[item_key].unique()

    # prev_iid, prev_sid = -1, -1
    prev_iid, prev_uid = -1, -1

    # rank_list = []  # DEBUG
    # score_list = []  # DEBUG
    for i in range(len(test_data)):
        if not test_data[item_key].values[i] in items_to_predict:
            count += 1
            # print(str(count) + " - item id: " + str(test_data[item_key].values[i]))
        else:
        # if True:
            if not test_data[item_key].values[i] in items_to_predict:
                print("item id: " + str(test_data[item_key].values[i]))
            if count % 1000 == 0:
                print('eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0 ), ' % in',(time.time()-st), 's')

            uid = test_data[user_key].values[i]
            iid = test_data[item_key].values[i]
            ts = test_data[time_key].values[i]
            if prev_uid != uid:
                prev_uid = uid
            else:
                next_user = test_data[user_key].values[i + 1] if i + 1 < len(test_data) else -1
                last = next_user is -1 or next_user != uid

                # pred_time.start_predict()  # scalability
                if last:  # if it is the last interaction in the sequence, we should make the measurements

                    rated = train_data[train_data[user_key] == uid][item_key].values
                    target_item = iid

                    if eval.is_negative_sampling():  # negative sampling
                        item_idx = data.negative_sampling([target_item], rated, use_pop_random=eval.is_pop_random(), sample_size=eval.get_sample_size())
                    else:  # normal evaluation
                        item_idx = data.get_items_id().copy().tolist()  # all items


                    pred_time.start_predict()  # scalability
                    preds = pr.predict_next(uid, prev_iid, predict_for_item_ids=np.array(item_idx), timestamp=ts, skip=(not last), user_history=rated)
                    pred_time.stop_predict()  # scalability
                    # --------------------------
                    preds[np.isnan(preds)] = 0
                    #             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                    preds.sort_values(ascending=False, inplace=True)
                    # --------------------------
                    target_index = item_idx.index(target_item)
                    predictions = -preds.loc[item_idx]
                    # print("DEBUG: target item's score: "+str(preds.loc[target_item]))
                    # score_list.append(preds.loc[target_item])  # DEBUG
                    rank = predictions.argsort().argsort().iloc[target_index]  # or: rank = predictions.argsort().argsort().iloc[target_item]
                    # print("DEBUG: target item's rank: " + str(rank))
                    # rank_list.append(rank)  # DEBUG

                    sorted_pred = predictions.sort_values()  # as values are negated (-preds), the smallest ones will be kept

                    eval.after_run(rank, sorted_pred, data.get_popularity())
                else:  # just to update the internal data structures of the model
                    preds = pr.predict_next(uid, prev_iid, predict_for_item_ids=items_to_predict, timestamp=ts, skip=(not last))
                # pred_time.stop_predict()  # scalability

        prev_iid = iid
        count += 1

    
    print( 'END evaluation' )
    # write_results_csv(rank_list, score_list)  # DEBUG
    # write_prediction_times_csv(key, pred_time.result(), conf)  # scalability
    pred_time.write_prediction_times_csv(pred_time.result(), key, conf=conf)  # scalability

    res = eval.end()
    return res

# def write_prediction_times_csv(key, results, conf):
#     '''
#     Write the result array to a csv file, if a result folder is defined in the configuration
#         --------
#         results : tuple of tuples
#     '''
#
#     if 'results' in conf and 'folder' in conf['results']:
#
#         export_csv = conf['results']['folder'] + 'prediction_times/' + 'test_' + conf['type'] + '_' + conf['key'] + '_' + conf['data']['name'] + '.csv'
#
#         ensure_dir(export_csv)
#
#         file = open(export_csv, 'w+')
#         file.write('Metrics;')
#
#         file.write(results[0][0])  # prediction time
#         file.write(';')
#         file.write(results[0][1])  # prediction time cpu
#         file.write(';')
#         file.write('\n')
#
#         file.write(key)  # key (algorithm)
#         file.write(';')
#
#         file.write(str(results[1][0]))  # value
#         file.write(';')
#         file.write(str(results[1][1]))  # value (cpu)
#         file.write(';')
#         file.write('\n')

# DEBUG
def write_results_csv(ranks, scores):
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

    # ranks
    mean = statistics.mean(ranks)
    median = statistics.median(ranks)
    variance = statistics.variance(ranks,mean)

    export_csv = "results/statistics/statistics_eval.csv"

    ensure_dir(export_csv)

    file = open(export_csv, 'w+')
    file.write('statistics;')
    file.write('mean;')
    file.write('median;')
    file.write('variance;')
    file.write('\n')

    file.write('ranks;')
    file.write(str(mean))
    file.write(';')
    file.write(str(median))
    file.write(';')
    file.write(str(variance))
    file.write(';')
    file.write('\n')

    # scores
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    variance = statistics.variance(scores, mean)

    file.write('scores;')
    file.write(str(mean))
    file.write(';')
    file.write(str(median))
    file.write(';')
    file.write(str(variance))
    file.write(';')
    file.write('\n')

# DEBUG
def ensure_dir(file_path):
    '''
    Create all directories in the file_path if non-existent.
        --------
        file_path : string
            Path to the a file
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)