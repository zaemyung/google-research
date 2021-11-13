# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Compute realized predictions for a dataset."""

import json
import os

import tensorflow as tf
from absl import app, flags, logging
from official.common import distribute_utils
from official.nlp.bert import configs

import felix_flags  # pylint: disable=unused-import
import predict
import utils

FLAGS = flags.FLAGS
log_path='/tmp-network/user/zaemyung-kim/projects/IteraTE/IteraTE-repo/data/final_data/felix_log.txt'
if os.path.isfile(log_path):
  os.remove(log_path)


def log_data(text):
  print(text)
  with open(log_path, 'a') as f:
    f.write(str(text))
    f.write("\n")


def batch_generator():
  """Produces batches for felix to predict."""
  source_batch = []
  target_batch = []
  for sources, target in utils.yield_sources_and_targets(
      FLAGS.predict_input_file, FLAGS.input_format):

    source_batch.append(
        FLAGS.special_glue_string_for_joining_sources.join(sources))
    target_batch.append(target)
    if len(source_batch) == FLAGS.predict_batch_size:
      yield source_batch, target_batch
      source_batch = []
      target_batch = []

  if source_batch:
    yield source_batch, target_batch


def postprocess(sentences):
  gen_sents = []
  for sentence in sentences:
    sentence = sentence.replace('[CLS]', '').replace('[SEP]', '')
    gen_tokens = []
    for token in sentence.split():
      if token[:2] == '##':
        last_token = gen_tokens[-1]
        gen_tokens[-1] = last_token + token[2:]
      else:
        gen_tokens.append(token)
    gen_sent = ' '.join(gen_tokens)
    gen_sents.append(gen_sent)
  return gen_sents


def get_pred(predictor, sentence):
  sources = [sentence]
  source_batch = [FLAGS.special_glue_string_for_joining_sources.join(sources)]
  _, predicted_inserts = predictor.predict_end_to_end_batch(source_batch)
  return postprocess(predicted_inserts)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.use_open_vocab:
    raise ValueError('Currently only use_open_vocab=True is supported')

  label_map = utils.read_label_map(FLAGS.label_map_file)
  bert_config_tagging = configs.BertConfig.from_json_file(
      FLAGS.bert_config_tagging)
  bert_config_insertion = configs.BertConfig.from_json_file(
      FLAGS.bert_config_insertion)
  if FLAGS.tpu is not None:
    cluster_resolver = distribute_utils.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    with strategy.scope():
      predictor = predict.FelixPredictor(
          bert_config_tagging=bert_config_tagging,
          bert_config_insertion=bert_config_insertion,
          model_tagging_filepath=FLAGS.model_tagging_filepath,
          model_insertion_filepath=FLAGS.model_insertion_filepath,
          vocab_file=FLAGS.vocab_file,
          label_map=label_map,
          sequence_length=FLAGS.max_seq_length,
          max_predictions=FLAGS.max_predictions_per_seq,
          do_lowercase=FLAGS.do_lower_case,
          use_open_vocab=FLAGS.use_open_vocab,
          is_pointing=FLAGS.use_pointing,
          insert_after_token=FLAGS.insert_after_token,
          special_glue_string_for_joining_sources=FLAGS
          .special_glue_string_for_joining_sources)
  else:
    predictor = predict.FelixPredictor(
        bert_config_tagging=bert_config_tagging,
        bert_config_insertion=bert_config_insertion,
        model_tagging_filepath=FLAGS.model_tagging_filepath,
        model_insertion_filepath=FLAGS.model_insertion_filepath,
        vocab_file=FLAGS.vocab_file,
        label_map_file=FLAGS.label_map_file,
        sequence_length=FLAGS.max_seq_length,
        max_predictions=FLAGS.max_predictions_per_seq,
        do_lowercase=FLAGS.do_lower_case,
        use_open_vocab=FLAGS.use_open_vocab,
        is_pointing=FLAGS.use_pointing,
        insert_after_token=FLAGS.insert_after_token,
        special_glue_string_for_joining_sources=FLAGS
        .special_glue_string_for_joining_sources)

  # Load References
  count = 0
  original_revision_dict = {}
  generated_revisions_dict = {}
  with open(FLAGS.predict_input_file, 'r') as f:
    for line in f:
      json_line = json.loads(line)
      doc_id = json_line['doc_id']
      revision_depth = int(json_line['revision_depth'])
      before_revision = json_line['before_revision']
      # after_revision = json_line['after_revision']
      # edit_actions = json_line['edit_actions']
      sents_char_pos = json_line['sents_char_pos']
      if doc_id not in original_revision_dict:
        original_revision_dict[doc_id] = revision_depth
      else:
        original_revision_dict[doc_id] = max(original_revision_dict[doc_id], revision_depth)
      if int(revision_depth) != 1:
        continue
      log_data("\n")
      log_data(revision_depth)
      log_data(f"Before revision: {before_revision}")
      if len(before_revision) not in sents_char_pos:
        sents_char_pos.append(len(before_revision))
      if 0 not in sents_char_pos:
        sents_char_pos.insert(0,0)
      log_data(sents_char_pos)
      max_iteration = 0
      for ind in range(len(sents_char_pos)-1):

        sentence = before_revision[sents_char_pos[ind]:sents_char_pos[ind+1]]
        log_data(f"Original sentence: {sentence}")
        if count == 1:
          log_data(sentence)

        ct = 0
        # so that the process does not enter a recursive loop
        sentences_log = []
        # using replace, since we should not consider it a change when only space is added/removed
        sentences_log.append(sentence.replace(" ", ""))
        while ct < 10:
          gen_sentence = get_pred(predictor, sentence)[0]
          if gen_sentence.replace(" ", "") not in sentences_log:
            sentence = gen_sentence
            sentences_log.append(gen_sentence.replace(" ", ""))
            ct += 1
            log_data(f"Generated sentence: {ct} : {gen_sentence}")
            max_iteration = max(max_iteration, ct)
          else:
            log_data(f"Max iteration for this sentence: {ct}")
            break
      log_data(f"max iteration for this doc: {max_iteration}")
      if doc_id in generated_revisions_dict:
        log_data("PROBLEM \n\n\n\n\n\n")
      generated_revisions_dict[doc_id] = max_iteration

      log_data(original_revision_dict)
      log_data(generated_revisions_dict)
      log_data("\n")
  log_data(original_revision_dict)
  log_data(generated_revisions_dict)


if __name__ == '__main__':
  flags.mark_flag_as_required('predict_input_file')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('vocab_file')
  app.run(main)
