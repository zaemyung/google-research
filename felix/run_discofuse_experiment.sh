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

cd /tmp-network/user/zaemyung-kim/google-research/felix

. env/bin/activate

# Please update these paths.
export OUTPUT_DIR=/tmp-network/user/zaemyung-kim/projects/felix_exp
# BERT can be found at https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz
# (Access to Pretrained Checkpoints, not Pretrained hub modules)
# Note: only used uncased if do_lower_case=True. Furthermore, you will need to
# change the felix_config.json to take into account the different vocab sizes.
export BERT_BASE_DIR=/tmp-network/user/zaemyung-kim/models/BERT/uncased_L-12_H-768_A-12
# Requires untaring the downloaded file.
# DiscoFuse can be found at https://github.com/google-research-datasets/discofuse
export DISCOFUSE_DIR=/tmp-network/user/zaemyung-kim/data/discofuse/discofuse_v1/wikipedia
export FELIX_CONFIG_DIR=/tmp-network/user/zaemyung-kim/google-research/felix/discofuse
export PREDICTION_FILE=${OUTPUT_DIR}/pred.tsv
# If you wish to use another dataset please switch from input_format=discofuse
# to wikisplit. wikisplit expects tab seperated source target pairs.

# If False FelixInsert is used.
export USE_POINTING='True'

# Need to clone: git clone https://github.com/tensorflow/models.git
export PYTHONPATH=$PYTHONPATH:/tmp-network/user/zaemyung-kim/google-models


# # Label map construction
# echo "Constructing vocabulary"
# python phrase_vocabulary_constructor_main.py \
# --output="${OUTPUT_DIR}/label_map.json" \
# --use_pointing="${USE_POINTING}" \
# --do_lower_case="True"

# # Preprocess
# echo "Preprocessing data"
# python preprocess_main.py \
#   --input_file="${DISCOFUSE_DIR}/train.tsv" \
#   --input_format="discofuse" \
#   --output_file="${OUTPUT_DIR}/train.tfrecord" \
#   --label_map_file="${OUTPUT_DIR}/label_map.json" \
#   --vocab_file="${BERT_BASE_DIR}/vocab.txt" \
#   --do_lower_case="True" \
#   --use_open_vocab="True" \
#   --max_seq_length="128" \
#   --use_pointing="${USE_POINTING}" \
#   --split_on_punc="True"

# python preprocess_main.py \
#   --input_file="${DISCOFUSE_DIR}/dev.tsv" \
#   --input_format="discofuse" \
#   --output_file="${OUTPUT_DIR}/dev.tfrecord" \
#   --label_map_file="${OUTPUT_DIR}/label_map.json" \
#   --vocab_file="${BERT_BASE_DIR}/vocab.txt" \
#   --do_lower_case="True" \
#   --use_open_vocab="True" \
#   --max_seq_length="128" \
#   --use_pointing="${USE_POINTING}" \
#   --split_on_punc="True"

# # Train
# echo "Training tagging model"
# rm -rf "${OUTPUT_DIR}/model_tagging"
# mkdir -p "${OUTPUT_DIR}/model_tagging"
# python run_felix.py \
#     --train_file="${OUTPUT_DIR}/train.tfrecord" \
#     --eval_file="${OUTPUT_DIR}/dev.tfrecord" \
#     --model_dir_tagging="${OUTPUT_DIR}/model_tagging" \
#     --bert_config_tagging="${FELIX_CONFIG_DIR}/felix_config.json" \
#     --max_seq_length=128 \
#     --num_train_epochs=100 \
#     --num_train_examples=300000 \
#     --num_eval_examples=5000 \
#     --train_batch_size="64" \
#     --eval_batch_size="64" \
#     --log_steps="100" \
#     --steps_per_loop="100" \
#     --train_insertion="False" \
#     --use_pointing="${USE_POINTING}" \
#     --init_checkpoint="${BERT_BASE_DIR}/bert_model.ckpt" \
#     --learning_rate="0.00006" \
#     --pointing_weight="1" \
#     --use_weighted_labels="True" \

# echo "Training insertion model"
# rm -rf "${OUTPUT_DIR}/model_insertion"
# mkdir "${OUTPUT_DIR}/model_insertion"
# python run_felix.py \
#     --train_file="${OUTPUT_DIR}/train.tfrecord.ins" \
#     --eval_file="${OUTPUT_DIR}/dev.tfrecord.ins" \
#     --model_dir_insertion="${OUTPUT_DIR}/model_insertion" \
#     --bert_config_insertion="${FELIX_CONFIG_DIR}/felix_config.json" \
#     --max_seq_length=128 \
#     --num_train_epochs=100 \
#     --num_train_examples=300000 \
#     --num_eval_examples=5000 \
#     --train_batch_size="64" \
#     --eval_batch_size="64" \
#     --log_steps="100" \
#     --steps_per_loop="100" \
#     --train_insertion="False" \
#     --init_checkpoint="${BERT_BASE_DIR}/bert_model.ckpt" \
#     --use_pointing="${USE_POINTING}" \
#     --learning_rate="0.00006" \
#     --pointing_weight="1" \
#     --train_insertion="True"

# Predict
echo "Generating predictions"

python predict_main.py \
--input_format="discofuse" \
--predict_input_file="${DISCOFUSE_DIR}/test.tsv" \
--predict_output_file="${PREDICTION_FILE}" \
--label_map_file="${OUTPUT_DIR}/label_map.json" \
--vocab_file="${BERT_BASE_DIR}/vocab.txt" \
--max_seq_length=128 \
--predict_batch_size=64 \
--do_lower_case="True" \
--use_open_vocab="True" \
--bert_config_tagging="${FELIX_CONFIG_DIR}/felix_config.json" \
--bert_config_insertion="${FELIX_CONFIG_DIR}/felix_config.json" \
--model_tagging_filepath="${OUTPUT_DIR}/model_tagging" \
--model_insertion_filepath="${OUTPUT_DIR}/model_insertion" \
--use_pointing="${USE_POINTING}"
