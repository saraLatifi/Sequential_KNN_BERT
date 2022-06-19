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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import operator
import collections
import os

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps,
                     use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate +
                         is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=5.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) +
                      tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) +
                      tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name



def make_early_stopping_hook(estimator,
                             should_stop_fn,
                             run_every_secs=60,
                             run_every_steps=None):
  """Creates early-stopping hook.
  Returns a `SessionRunHook` that stops training when `should_stop_fn` returns
  `True`.
  Usage example:
  ```python
  estimator = ...
  hook = early_stopping.make_early_stopping_hook(
      estimator, should_stop_fn=make_stop_fn(...))
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```
  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.
  Args:
    estimator: A `tf.estimator.Estimator` instance.
    should_stop_fn: `callable`, function that takes no arguments and returns a
      `bool`. If the function returns `True`, stopping will be initiated by the
      chief.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.
  Returns:
    A `SessionRunHook` that periodically executes `should_stop_fn` and initiates
    early stopping if the function returns `True`.
  Raises:
    TypeError: If `estimator` is not of type `tf.estimator.Estimator`.
    ValueError: If both `run_every_secs` and `run_every_steps` are set.
  """
  if not isinstance(estimator, tf.estimator.Estimator):
    raise TypeError('`estimator` must have type `tf.estimator.Estimator`. '
                    'Got: {}'.format(type(estimator)))

  if run_every_secs is not None and run_every_steps is not None:
    raise ValueError('Only one of `run_every_secs` and `run_every_steps` must '
                     'be set.')

  if estimator.config.is_chief:
    return _StopOnPredicateHook(should_stop_fn, run_every_secs, run_every_steps)
  else:
    return _CheckForStoppingHook()


def stop_if_no_decrease_hook(estimator,
                             metric_name,
                             max_steps_without_decrease,
                             eval_dir=None,
                             min_steps=0,
                             run_every_secs=60,
                             run_every_steps=None):

  return _stop_if_no_metric_improvement_hook(
      estimator=estimator,
      metric_name=metric_name,
      max_steps_without_improvement=max_steps_without_decrease,
      higher_is_better=False,
      eval_dir=eval_dir,
      min_steps=min_steps,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)

def read_eval_metrics(eval_dir):
  """Helper to read eval metrics from eval summary files.
  Args:
    eval_dir: Directory containing summary files with eval metrics.
  Returns:
    A `dict` with global steps mapping to `dict` of metric names and values.
  """
  eval_metrics_dict = {}
  for event in _summaries(eval_dir):
    if not event.HasField('summary'):
      continue
    metrics = {}
    for value in event.summary.value:
      if value.HasField('simple_value'):
        metrics[value.tag] = value.simple_value
    if metrics:
      eval_metrics_dict[event.step] = metrics
  return collections.OrderedDict(
      sorted(eval_metrics_dict.items(), key=lambda t: t[0]))

def _stop_if_no_metric_improvement_hook(
    estimator, metric_name, max_steps_without_improvement, higher_is_better,
    eval_dir, min_steps, run_every_secs, run_every_steps):
  """Returns hook to stop training if given metric shows no improvement."""

  if eval_dir is None:
    eval_dir = estimator.eval_dir()

  is_lhs_better = operator.gt if higher_is_better else operator.lt
  increase_or_decrease = 'increase' if higher_is_better else 'decrease'

  def stop_if_no_metric_improvement_fn():
    """Returns `True` if metric does not improve within max steps."""

    eval_results = read_eval_metrics(eval_dir)

    best_val = None
    best_val_step = None
    for step, metrics in eval_results.items():
      tf.logging.info('Early stopping running.'+str(step)+" "+str(metrics))
      if step < min_steps:
        continue
      val = metrics[metric_name]
      if best_val is None or is_lhs_better(val, best_val):
        best_val = val
        best_val_step = step
      if step - best_val_step >= max_steps_without_improvement:
        tf.logging.info(
            'No %s in metric "%s" for %s steps, which is greater than or equal '
            'to max steps (%s) configured for early stopping.',
            increase_or_decrease, metric_name, step - best_val_step,
            max_steps_without_improvement)
        return True

    tf.logging.info('Early stopping running.'+str(max_steps_without_improvement)+" "+str(best_val)+" "+str( best_val_step))
    return False

  return make_early_stopping_hook(
      estimator=estimator,
      should_stop_fn=stop_if_no_metric_improvement_fn,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


class _StopOnPredicateHook(tf.estimator.SessionRunHook):#tf.train.SessionRunHook):
  """Hook that requests stop when `should_stop_fn` returns `True`."""

  def __init__(self, should_stop_fn, run_every_secs=60, run_every_steps=None):
    if not callable(should_stop_fn):
      raise TypeError('`should_stop_fn` must be callable.')

    tf.logging.info('Early stopping init.')
    self._should_stop_fn = should_stop_fn
    self._timer = tf.estimator.SecondOrStepTimer(
        every_secs=run_every_secs, every_steps=run_every_steps)
    self._global_step_tensor = None
    self._stop_var = None
    self._stop_op = None

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    self._stop_var = _get_or_create_stop_var()
    self._stop_op = tf.assign(self._stop_var, True)

  def before_run(self, run_context):
    del run_context
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)
      if self._should_stop_fn():
        tf.logging.info('Requesting early stopping at global step %d',
                        global_step)
        run_context.session.run(self._stop_op)
        run_context.request_stop()


class _CheckForStoppingHook(tf.estimator.SessionRunHook):#tf.train.SessionRunHook):
  """Hook that requests stop if stop is requested by `_StopOnPredicateHook`."""

  def __init__(self):
    self._stop_var = None

  def begin(self):
    self._stop_var = _get_or_create_stop_var()

  def before_run(self, run_context):
    del run_context
    return tf.train.SessionRunArgs(self._stop_var)

  def after_run(self, run_context, run_values):
    should_early_stop = run_values.results
    if should_early_stop:
      tf.logging.info('Early stopping requested, suspending run.')
      run_context.request_stop()

def _summaries(eval_dir):
  """Yields `tensorflow.Event` protos from event files in the eval dir.
  Args:
    eval_dir: Directory containing summary files with eval metrics.
  Yields:
    `tensorflow.Event` object read from the event files.
  """
  tf.logging.info('Early stopping summary.'+eval_dir)
  if tf.gfile.Exists(eval_dir):
    for event_file in tf.gfile.Glob(
        os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
      for event in tf.train.summary_iterator(event_file):
        yield event

def _get_or_create_stop_var():
  with tf.variable_scope(
      name_or_scope='signal_early_stopping',
      values=[],
      reuse=tf.AUTO_REUSE):
    return tf.get_variable(
        name='STOP',
        shape=[],
        dtype=tf.bool,
        initializer=tf.constant_initializer(False),
        collections=[tf.GraphKeys.GLOBAL_VARIABLES],
        trainable=False)
