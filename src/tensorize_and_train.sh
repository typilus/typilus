#!/bin/bash

MODEL_TYPE=$1
INPUT_FOLDER=$2
TARGET_FOLDER=$3

echo Training and tensorizing a $MODEL_TYPE model. Raw data at $INPUT_FOLDER. Tensorized data and trained model will be saved at $TARGET_FOLDER.

export PYTHONPATH=$PWD

TRAIN_TENSORIZED_FOLDER=$TARGET_FOLDER/train-tensorized
VALID_TENSORIZED_FOLDER=$TARGET_FOLDER/valid-tensorized
mkdir -p $TRAIN_TENSORIZED_FOLDER $VALID_TENSORIZED_FOLDER

python3 typilus/utils/tensorise.py --model $MODEL_TYPE $TRAIN_TENSORIZED_FOLDER $INPUT_FOLDER/train
python3 typilus/utils/tensorise.py --model $MODEL_TYPE $VALID_TENSORIZED_FOLDER $INPUT_FOLDER/valid --metadata-to-use $TRAIN_TENSORIZED_FOLDER/metadata.pkl.gz

python3 typilus/utils/train.py --model $MODEL_TYPE $TARGET_FOLDER/models $TRAIN_TENSORIZED_FOLDER $VALID_TENSORIZED_FOLDER

