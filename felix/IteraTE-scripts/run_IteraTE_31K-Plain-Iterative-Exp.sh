#!/bin/bash
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_v100"
#SBATCH --output=/tmp-network/user/zaemyung-kim/slurm_logs/%j-IteraTE.log
#SBATCH -p gpu

cd /tmp-network/user/zaemyung-kim/projects/IteraTE/google-research/felix

. /tmp-network/user/zaemyung-kim/projects/IteraTE/env/bin/activate

# Please update these paths.
export OUTPUT_DIR=/tmp-network/user/zaemyung-kim/projects/IteraTE/output_dir/31K_Plain
export BERT_BASE_DIR=/tmp-network/user/zaemyung-kim/projects/IteraTE/BERT-uncased_L-12_H-768_A-12
export FELIX_CONFIG_DIR=/tmp-network/user/zaemyung-kim/projects/IteraTE/google-research/felix/discofuse

# If False FelixInsert is used.
export USE_POINTING='True'

# Need to clone: git clone https://github.com/tensorflow/models.git
export PYTHONPATH=$PYTHONPATH:/tmp-network/user/zaemyung-kim/projects/IteraTE/google-models

export INPUT_FORMAT="IteraTE_Plain"
export PREDICTION_FILE="NotUsed"


# Predict
echo "Generating predictions"

python predict_main_iterative.py \
--input_format="${INPUT_FORMAT}" \
--predict_input_file="/tmp-network/user/zaemyung-kim/projects/IteraTE/IteraTE-repo/data/final_data/data_for_iterative_revision.json" \
--predict_output_file="${PREDICTION_FILE}" \
--label_map_file="${OUTPUT_DIR}/label_map.json" \
--vocab_file="${BERT_BASE_DIR}/vocab.txt" \
--max_seq_length=128 \
--predict_batch_size=1 \
--do_lower_case="True" \
--use_open_vocab="True" \
--bert_config_tagging="${FELIX_CONFIG_DIR}/felix_config.json" \
--bert_config_insertion="${FELIX_CONFIG_DIR}/felix_config.json" \
--model_tagging_filepath="${OUTPUT_DIR}/model_tagging" \
--model_insertion_filepath="${OUTPUT_DIR}/model_insertion" \
--use_pointing="${USE_POINTING}"
