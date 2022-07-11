import numpy as np
import pickle
from collections import Counter

class Dataloader:

    def __init__(self, modelType, user_history_filename=None, vocab_filename=None, train_data=None, test_data=None, user_key="UserId", item_key="ItemId"):
        np.random.seed(12345)
        if modelType == 'bert4rec':
            print("modelType: bert4rec")
            self.user_history_filename = user_history_filename
            self.vocab_filename = vocab_filename
            self.vocab = None
            self.values = None

            if self.user_history_filename is not None:
                print('load user history from :' + self.user_history_filename)
                with open(self.user_history_filename, 'rb') as input_file:
                    self.user_history = pickle.load(input_file)

            if self.vocab_filename is not None:
                print('load vocab from :' + self.vocab_filename)
                with open(self.vocab_filename, 'rb') as input_file:
                    self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            self.values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            if self.values is not None:
                sum_value = np.sum([x for x in self.values])
                # print(sum_value)
                self.probability = [value / sum_value for value in self.values]
                # to calculate the popularity bias
                values_list = []
                for val in self.values:
                    values_list.append(val)

                self.popularity = dict()
                for i in range(len(self.values)):
                    item = self.ids[i]
                    count = values_list[i]
                    self.popularity[item] = count / sum_value
        elif modelType == 'baseline':
            print("modelType: baseline")
            self.user_key = user_key
            self.item_key = item_key
            self.user_history = train_data
            self.vocab = train_data[self.item_key]  # previously: test_data[self.item_key]
            self.values = Counter(self.vocab)
            self.ids = self.vocab.unique()
            if self.values is not None:
                sum_value = np.sum([count for item, count in self.values.items()])
                self.probability = [count / sum_value for item, count in self.values.items()]
                # to calculate the popularity bias
                self.popularity = dict()
                for item, count in self.values.items():
                    self.popularity[item] = count/sum_value
        else:
            print("Type of the model is not supported!")

    def negative_sampling(self, item_idx, rated, use_pop_random=True, sample_size=100):
        # here we need more consideration
        size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
        if use_pop_random:
            if self.vocab is not None:
                while len(
                        item_idx) < (sample_size+1):  # select 100 negative samples that user has never interacted with them [BERT-KNN]
                    sampled_ids = np.random.choice(self.ids, (sample_size+1), replace=False, p=self.probability)
                    sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                    item_idx.extend(sampled_ids[:])
                item_idx = item_idx[:(sample_size+1)]  # first item is "target item" [BERT-KNN]
        else:
            # print("evaluation random -> ")
            for _ in range(sample_size):  # select 100 negative samples [BERT-KNN]
                t = np.random.randint(1, size_of_prob)
                while t in rated:  # user should have never interacted with the negative sample [BERT-KNN]
                    t = np.random.randint(1, size_of_prob)
                item_idx.append(t)

        # locate target item in a random position in item_idx list
        rnd_pos = np.random.randint(0, len(item_idx))
        target_item = item_idx[0]
        item_idx[0] = item_idx[rnd_pos]
        item_idx[rnd_pos] = target_item
        return item_idx

    def get_user_history(self):
        return self.user_history

    def get_history(self, user_id):
        history = self.user_history[self.user_history[self.user_key] == user_id]
        return history

    def get_items_id(self):
        return self.ids

    def get_popularity(self):
        return self.popularity