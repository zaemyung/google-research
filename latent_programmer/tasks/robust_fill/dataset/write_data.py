# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Write supervised training tasks to TFRecord dataset."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import os
import random
import sys
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer.tasks.robust_fill import dsl
from latent_programmer.tasks.robust_fill import sample_random
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens
from latent_programmer.tasks.robust_fill.dataset import experiment as exp_module


sys.path.append('../../../../')
gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_work_units', 1, 'Total number of work units.')
flags.DEFINE_integer('seed', None, 'Fixed random seed.')

flags.DEFINE_integer('num_tasks', 100000, 'Number of tasks to write.')
flags.DEFINE_integer('num_strings_per_task', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_expressions', 10,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('min_expressions', 1,
                     'Maximum number of expressions in program.')
flags.DEFINE_integer('max_input_length', 20,
                     'Maximum number of characters in input strings.')

flags.DEFINE_string('save_dir', '/tmp/decomposition/robust_fill',
                    'Directory to save results to.')

flags.DEFINE_boolean('split_program', False,
                     'Whether to split program by parial program.')

flags.DEFINE_enum('split', None, ['train', 'valid', 'test', 'finetune'],
                  'Which split of the dataset to generate.')
flags.DEFINE_enum('experiment', 'NONE', [e.name for e in exp_module.Experiment],
                  'Kind of experiment (see document for descriptions).')


def _bytes_feature(strs):
  """Returns a bytes_list Feature from a list of strings."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[str.encode(s) for s in strs]))


def serialize_entire_program_example(task, token_id_table):
  """Creates a tf.Example message for the entire program."""
  program_string = ''
  if FLAGS.split_program:
    for expr in task.program.expressions:
      program_string += ' '.join(map(str, expr.encode(token_id_table)))
      program_string += '|'
    program_string = program_string[:-1]
  else:
    program_string = ' '.join(
        map(str, task.program.encode(token_id_table)[:-1]))

  feature = {
      'inputs': _bytes_feature(task.inputs),
      'outputs': _bytes_feature(task.outputs),
      'program_encoding': _bytes_feature([program_string]),
  }

  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def serialize_decomposition_examples(task, token_id_table):
  """Creates tf.Example messages for decomposition."""
  # TODO(kshi): If we want to include length-2 programs in the subprogram
  # synthesizer's training data, we'll need to create a separate dataset for
  # that, since we don't want such data in the spec decomposer model's training
  # data.
  output_parts = [[expr(inp) for expr in task.program.expressions]
                  for inp in task.inputs]
  assert all(''.join(parts) == out
             for parts, out in zip(output_parts, task.outputs))

  results = []
  for i, expr in enumerate(task.program.expressions):
    outputs = [''.join(parts[i:]) for parts in output_parts]
    next_part = [parts[i] for parts in output_parts]
    program_part_string = ' '.join(map(str, expr.encode(token_id_table)))
    feature = {
        'inputs': _bytes_feature(task.inputs),
        'outputs': _bytes_feature(outputs),
        'next_part': _bytes_feature(next_part),
        'program_part': _bytes_feature([program_part_string]),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    results.append(example_proto.SerializeToString())

  return results


def generate_task_for_experiment(experiment, is_train):
  """Generates a random task for a given experiment and dataset split."""
  if experiment == exp_module.Experiment.SWITCH_CONCEPT_ORDER.name:
    # Handle this case separately because it's the most different from the rest.
    return sample_random.random_task_switch_concept_order(
        max_k=3,
        max_input_tokens=5,
        max_input_length=FLAGS.max_input_length,
        num_examples=FLAGS.num_strings_per_task,
        min_expressions=2,
        max_expressions=6,
        is_train=is_train)

  # Still pass in max_expressions, min_expressions, sampler_pool,
  # valid_length_fn, and keep_fn.
  random_task_partial = functools.partial(
      sample_random.random_task,
      max_k=3,
      max_input_tokens=5,
      max_input_length=FLAGS.max_input_length,
      num_examples=FLAGS.num_strings_per_task)

  valid_num_expressions_fn = None
  keep_fn = None

  if experiment == exp_module.Experiment.LENGTH_1_6_TO_7_10.name:
    min_expressions = 1 if is_train else 7
    max_expressions = 6 if is_train else 10
    sampler_pool = sample_random.SAMPLER_POOL_ALL

  elif experiment == exp_module.Experiment.LENGTH_6_TO_1_10.name:
    min_expressions = 6 if is_train else 1
    max_expressions = 6 if is_train else 10
    sampler_pool = sample_random.SAMPLER_POOL_ALL
    if not is_train:
      valid_num_expressions_fn = lambda n: n != 6

  elif experiment == exp_module.Experiment.LENGTH_1_TO_2_6.name:
    min_expressions = 1 if is_train else 2
    max_expressions = 1 if is_train else 6
    sampler_pool = sample_random.SAMPLER_POOL_ALL

  elif experiment == exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS.name:
    min_expressions = 2
    max_expressions = 6
    if is_train:
      sampler_pool = random.choice([sample_random.ALL_SUBSTRING,
                                    sample_random.SAMPLER_POOL_MODIFY_OR_CONST])
    else:
      sampler_pool = [sample_random.ALL_SUBSTRING,
                      sample_random.SAMPLER_POOL_MODIFY_OR_CONST]
      keep_fn = lambda c: (  # pylint: disable=g-long-lambda
          any(isinstance(e, dsl.Substring) for e in c.expressions) and
          any(isinstance(e, (dsl.Modification, dsl.ConstStr))
              for e in c.expressions))

  elif experiment == exp_module.Experiment.COMPOSE_NEW_OP.name:
    if is_train:
      if random.random() < 0.25:
        min_expressions = 1
        max_expressions = 1
        sampler_pool = sample_random.SAMPLER_POOL_ONLY_COMPOSE
      else:
        min_expressions = 2
        max_expressions = 6
        sampler_pool = sample_random.SAMPLER_POOL_NO_COMPOSE
    else:
      min_expressions = 2
      max_expressions = 6
      sampler_pool = sample_random.SAMPLER_POOL_ALL
      keep_fn = lambda c: any(isinstance(e, dsl.Compose) for e in c.expressions)

  elif experiment == exp_module.Experiment.EXTEND_OP_FUNCTIONALITY.name:
    min_expressions = 1
    max_expressions = 6
    sampler_pool = (sample_random.SAMPLER_POOL_NO_COMPOSE_SUBSTRING if is_train
                    else sample_random.SAMPLER_POOL_ALL)
    if not is_train:
      keep_fn = lambda c: any(  # pylint: disable=g-long-lambda
          isinstance(e, dsl.Compose) and
          isinstance(e.modification_or_substring, dsl.Substring)
          for e in c.expressions)

  else:
    raise ValueError('Unhandled experiment name: {}'.format(experiment))

  if is_train:
    # These are only used for test.
    assert valid_num_expressions_fn is None and keep_fn is None

  return random_task_partial(
      max_expressions=max_expressions,
      min_expressions=min_expressions,
      sampler_pool=sampler_pool,
      valid_num_expressions_fn=valid_num_expressions_fn,
      keep_fn=keep_fn)


def main(_):
  tf.enable_v2_behavior()

  if FLAGS.seed is not None:
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

  _, token_id_table = dsl_tokens.build_token_tables()

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  shard_id = 0
  total_shards = 1

  entire_programs_fname = os.path.join(
      FLAGS.save_dir,
      'entire_programs_{}.tf_records-{:05d}-of-{:05d}'.format(
          FLAGS.split, shard_id, total_shards))
  decomposition_data_fname = os.path.join(
      FLAGS.save_dir,
      'decomposition_data_{}.tf_records-{:05d}-of-{:05d}'.format(
          FLAGS.split, shard_id, total_shards))

  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(entire_programs_fname) as entire_programs_writer, \
      tf.io.TFRecordWriter(decomposition_data_fname) as decomposition_data_writer:
    for i in range(FLAGS.num_tasks):
      if FLAGS.experiment == exp_module.Experiment.NONE.name:
        task = sample_random.random_task(
            max_expressions=FLAGS.max_expressions,
            min_expressions=FLAGS.min_expressions,
            max_k=3,
            max_input_tokens=5,
            max_input_length=FLAGS.max_input_length,
            num_examples=FLAGS.num_strings_per_task)
      else:
        if FLAGS.split in ['train', 'valid']:
          is_train = True
        elif FLAGS.split == 'test':
          is_train = False
        elif FLAGS.split == 'finetune':
          is_train = bool(i % 2)
        else:
          raise ValueError('Unhandled split: {}'.format(FLAGS.split))
        task = generate_task_for_experiment(FLAGS.experiment, is_train)

      entire_programs_writer.write(
          serialize_entire_program_example(task, token_id_table))
      for example in serialize_decomposition_examples(task, token_id_table):
        decomposition_data_writer.write(example)

if __name__ == '__main__':
  app.run(main)
