import numpy as np
import sys
import statistics
import os

class Evaluation:

    def __init__(self, negative_sampling=True, use_pop_random=True, negative_sample_size=100, export_csv=None):
        self.negative_sampling = negative_sampling
        self.use_pop_random = use_pop_random
        self.negative_sample_size = negative_sample_size
        self.valid_user = 0.0
        # --- START --- [BERT-KNN]
        # [Paper] Considering we only have one ground truth item for each user:
        # HR@k is equivalent to Recall@k and proportional to Precision@k
        # MRR is equivalent to Mean Average Precision (MAP)
        # --- END ---
        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ap = 0.0  # Mean Reciprocal Rank (MRR) [BERT-KNN]
        self.pop_1 = 0.0
        self.pop_5 = 0.0
        self.pop_10 = 0.0
        self.num_1 = 0.0
        self.num_5 = 0.0
        self.num_10 = 0.0

        self.rank_list = []  # DEBUG

        # statistically significance test
        # ensure_dir
        directory = os.path.dirname(export_csv)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.file = open(export_csv, 'w+')
        self.file.write('User;')
        self.file.write("HR@1;")
        self.file.write("NDCG@5;")
        self.file.write("HR@5;")
        self.file.write("NDCG@10;")
        self.file.write("HR@10;")
        self.file.write("MRR;")
        self.file.write('\n')



    def after_run(self, rank, sorted_predictions, popularity):
        self.valid_user += 1

        if self.valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

        # statistically significance test
        hit_1, ndcg_5, hit_5, ndcg_10, hit_10, ap = 0, 0, 0, 0, 0, 0
        if rank < 1:
            self.ndcg_1 += 1
            self.hit_1 += 1
            # statistically significance test
            hit_1 = 1
        if rank < 5:
            self.ndcg_5 += 1 / np.log2(rank + 2)
            self.hit_5 += 1
            # statistically significance test
            ndcg_5 = 1 / np.log2(rank + 2)
            hit_5 = 1
        if rank < 10:
            self.ndcg_10 += 1 / np.log2(rank + 2)
            self.hit_10 += 1
            # statistically significance test
            ndcg_10 = 1 / np.log2(rank + 2)
            hit_10 = 1

        self.ap += 1.0 / (rank + 1)
        # statistically significance test
        ap = 1.0 / (rank + 1)

        idx = sorted_predictions[:1].index
        if idx[0] in popularity.keys():
            self.pop_1 += popularity[idx[0]]
            self.num_1 += 1

        idx = sorted_predictions[:5].index
        for x in idx:
            if x in popularity.keys():
                self.pop_5 += popularity[x]
                self.num_5 += 1

        idx = sorted_predictions[:10].index
        for x in idx:
            if x in popularity.keys():
                self.pop_10 += popularity[x]
                self.num_10 += 1

        self.rank_list.append(rank)  # DEBUG

        # statistically significance test
        self.file.write(str(self.valid_user))
        self.file.write(';')
        self.file.write(str(hit_1))
        self.file.write(";")
        self.file.write(str(ndcg_5))
        self.file.write(";")
        self.file.write(str(hit_5))
        self.file.write(";")
        self.file.write(str(ndcg_10))
        self.file.write(";")
        self.file.write(str(hit_10))
        self.file.write(";")
        self.file.write(str(ap))
        self.file.write(";")
        self.file.write('\n')

    def end(self):
        print(
            "ndcg@1:{}, hit@1:{}， ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}, pop@1:{}, pop@5:{}, pop@10:{}".
                format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
                       self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
                       self.ndcg_10 / self.valid_user,
                       self.hit_10 / self.valid_user, self.ap / self.valid_user,
                       self.valid_user, self.pop_1 / self.num_1,
                       self.pop_5 / self.num_5, self.pop_10 / self.num_10))


        # return ("Precision@" + str(self.length) + ": "), (self.hit / self.test)
        self.print_statistics(self.rank_list)  # DEBUG

        return ("ndcg@1; hit@1; ndcg@5; hit@5; ndcg@10; hit@10; ap; valid_user; pop@1; pop@5; pop@10;"), (str(self.ndcg_1 / self.valid_user)+";"+
                str(self.hit_1 / self.valid_user)+";"+
                str(self.ndcg_5 / self.valid_user)+";"+
                str(self.hit_5 / self.valid_user)+";"+
                str(self.ndcg_10 / self.valid_user)+";"+
                str(self.hit_10 / self.valid_user)+";"+
                str(self.ap / self.valid_user)+";"+
                str(self.valid_user)+";"+
                str(self.pop_1 / self.num_1)+";"+
                str(self.pop_5 / self.num_5)+";"+
                str(self.pop_10 / self.num_10)+";")

    def is_negative_sampling(self):
        return self.negative_sampling

    def is_pop_random(self):
        return self.use_pop_random

    def get_sample_size(self):
        return self.negative_sample_size

    # DEBUG
    def print_statistics(self, ranks):
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
        variance = statistics.variance(ranks, mean)

        print("ranks:: mean:{}， median:{}, variance:{}".
              format(mean, median, variance))

