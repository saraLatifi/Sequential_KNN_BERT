# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
import numpy as np
import sys
import pickle
import pandas as pd

from evaluation.dataloader import Dataloader
from evaluation.evaluation import Evaluation
import evaluation.scalability as runningTimes  # scalability

import statistics  # DEBUG

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("signature", 'default', "signature_name")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

# flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

# flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_pop_random", True, "use pop random negative samples")
flags.DEFINE_string("vocab_filename", None, "vocab filename")
flags.DEFINE_string("user_history_filename", None, "user history filename")
flags.DEFINE_bool("negative_sampling_evaluation", True, "use negative samples for evaluation")
flags.DEFINE_integer("negative_sample_size", 100, "use 100 negative samples for evaluation")

class EvalHooks(tf.train.SessionRunHook):
    def __init__(self):
        tf.logging.info('run init')

    def begin(self):
        export_csv = FLAGS.checkpointDir + '/statistical_significance/' + 'all_preds_bert4rec.csv'
        self.eval = Evaluation(negative_sampling=FLAGS.negative_sampling_evaluation, use_pop_random=FLAGS.use_pop_random, negative_sample_size=FLAGS.negative_sample_size, export_csv=export_csv)

        self.vocab = None

        self.data = Dataloader('bert4rec', user_history_filename=FLAGS.user_history_filename, vocab_filename=FLAGS.vocab_filename)

        self.score_list = []  # DEBUG

        self.prediction_time = runningTimes.Time_usage_testing()  # scalability
        self.prediction_time.init()  # scalability

    def end(self, session):
        # DEBUG ---------------
        mean = statistics.mean(self.score_list)
        median = statistics.median(self.score_list)
        # variance = statistics.variance(self.score_list)
        # variance = statistics.variance(self.score_list, mean)

        # print("scores:: mean:{}， median:{}, variance:{}".
        #       format(mean, median, variance))
        print("scores:: mean:{}， median:{}".
              format(mean, median))
        # --------------- DEBUG
        # write_prediction_times_csv(self.prediction_time.result())  # scalability
        self.prediction_time.write_prediction_times_csv(self.prediction_time.result(), 'bert4rec', checkpointDir=FLAGS.checkpointDir)  # scalability
        self.eval.end()

    def before_run(self, run_context):
        # tf.logging.info('run before run')
        # print('*** before_run ***')
        variables = tf.get_collection('eval_sp')
        self.prediction_time.start_predict()  # scalability
        return tf.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        # tf.logging.info('run after run')
        # print('*** after run ***')
        masked_lm_log_probs, input_ids, masked_lm_ids, info = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, FLAGS.max_predictions_per_seq, masked_lm_log_probs.shape[1]))
        #         print("loss value:", masked_lm_log_probs.shape, input_ids.shape,
        #               masked_lm_ids.shape, info.shape)
        # self.prediction_time.stop_predict(batch_size=len(input_ids), neural_model=True)  # scalability
        self.prediction_time.stop_predict(batch_size=len(input_ids))  # scalability
        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            target_item = masked_lm_ids[idx][0]  # BERT-KNN, target item's id
            rated.add(target_item)
            self.user_history = self.data.get_user_history()  # handle with dataloader [BERT-KNN]
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            # here we need more consideration
            masked_lm_log_probs_elem = masked_lm_log_probs[idx, 0]

            if self.eval.is_negative_sampling():  # negative sampling
                item_idx = self.data.negative_sampling([target_item], rated, use_pop_random=self.eval.is_pop_random(), sample_size=self.eval.get_sample_size())
            else:  # normal evaluation [BERT-KNN]
                item_idx = self.data.get_items_id().copy()  # all items

            target_index = item_idx.index(target_item)
            predictions = -masked_lm_log_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[target_index]

            self.score_list.append(masked_lm_log_probs_elem[target_item])  # DEBUG

            sorted_pred = pd.Series(predictions).sort_values()  # as values are negated (-masked_lm_log_probs_elem), the smallest ones will be kept

            self.eval.after_run(rank, sorted_pred, self.data.get_popularity())


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name,
                                                         features[name].shape))

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)

        #         all_user_and_item = model.get_embedding_table()
        #         item_ids = [i for i in range(0, item_size + 1)]
        #         softmax_output_embedding = tf.nn.embedding_lookup(all_user_and_item, item_ids)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config,
            model.get_sequence_output(),
            model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
                          masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss,
                                                    [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            tf.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.add_to_collection('eval_sp', input_ids)
            tf.add_to_collection('eval_sp', masked_lm_ids)
            tf.add_to_collection('eval_sp', info)

            eval_metrics = metric_fn(masked_lm_example_loss,
                                     masked_lm_log_probs, masked_lm_ids,
                                     masked_lm_weights)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" %
                             (mode))

        return output_spec

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info":
                tf.FixedLenFeature([1], tf.int64),  # [user]
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

            # `cycle_length` is the number of parallel files that get read.
            # cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            # d = d.apply(
            #    tf.contrib.data.parallel_interleave(
            #        tf.data.TFRecordDataset,
            #        sloppy=is_training,
            #        cycle_length=cycle_length))
            # d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)

        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS.checkpointDir = FLAGS.checkpointDir + FLAGS.signature
    print('checkpointDir:', FLAGS.checkpointDir)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
        train_input_files.extend(tf.gfile.Glob(input_pattern))

    test_input_files = []
    if FLAGS.test_input_file is None:
        test_input_files = train_input_files
    else:
        for input_pattern in FLAGS.test_input_file.split(","):
            test_input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** train Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tf.logging.info("*** test Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.checkpointDir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    if FLAGS.vocab_filename is not None:
        with open(FLAGS.vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        item_size=item_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": FLAGS.batch_size
        })

    training_time = runningTimes.Time_usage_training()  # scalability
    training_time.init()  # scalability
    # prediction_time = runningTimes.Time_usage_testing()  # scalability
    # prediction_time.init()  # scalability

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)

        training_time.start()  # scalability
        estimator.train(
            input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
        # training_time.stop(neural_model=True)  # scalability
        training_time.stop()  # scalability
        # write_training_times_csv(training_time.result())  # scalability
        training_time.write_training_times_csv(training_time.result(), 'bert4rec', checkpointDir=FLAGS.checkpointDir)  # scalability

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = input_fn_builder(
            input_files=test_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        # prediction_time.start_predict()  # scalability
        # tf.logging.info('special eval ops:', special_eval_ops)
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=None,
            hooks=[EvalHooks()])
        # prediction_time.stop_predict()  # scalability

        output_eval_file = os.path.join(FLAGS.checkpointDir,
                                        "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            tf.logging.info(bert_config.to_json_string())
            writer.write(bert_config.to_json_string() + '\n')
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

# def ensure_dir(file_path):
#     '''
#     Create all directories in the file_path if non-existent.
#         --------
#         file_path : string
#             Path to the a file
#     '''
#     directory = os.path.dirname(file_path)
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# def write_training_times_csv(results):
#     '''
#     Write the result array to a csv file, if a result folder is defined in the configuration
#         --------
#         results : tuple
#     '''
#
#     export_csv = FLAGS.checkpointDir + '/training_times/' + 'all_preds.csv'
#
#     ensure_dir(export_csv)
#
#     file = open(export_csv, 'w+')
#     file.write('Metrics;')
#
#     file.write(results[0])  # Training time
#     file.write(';')
#     file.write('\n')
#
#     file.write("bert4rec")
#     file.write(';')
#
#     file.write(str(results[1]))  # value
#     file.write(';')
#     file.write('\n')



# def write_prediction_times_csv(results):
#     '''
#     Write the result array to a csv file, if a result folder is defined in the configuration
#         --------
#         results : tuple of tuples
#     '''
#
#     export_csv = FLAGS.checkpointDir + '/prediction_times/' + 'all_preds.csv'
#
#     ensure_dir(export_csv)
#
#     file = open(export_csv, 'w+')
#     file.write('Metrics;')
#
#     file.write(results[0][0])  # prediction time
#     file.write(';')
#     file.write(results[0][1])  # prediction time cpu
#     file.write(';')
#     file.write('\n')
#
#     file.write("bert4rec")
#     file.write(';')
#
#     file.write(str(results[1][0]))  # value
#     file.write(';')
#     file.write(str(results[1][1]))  # value (cpu)
#     file.write(';')
#     file.write('\n')




if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("checkpointDir")
    flags.mark_flag_as_required("user_history_filename")
    tf.app.run()