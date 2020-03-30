# Typilus

A deep meta-learning method for predicting types in Python code.

#### Setup

We assume an environment with Python 3.6 or later with Docker installed.
First, optionally, create a Python virtual environment:

```bash
python3 -m venv ./typilus-env
source ./typilus-env/bin/activate
```

Install requirements:

```bash
pip3 install tensorflow==1.13.2 dpu-utils sentencepiece annoy
```

To install a GPU-based version of tensorflow, use `tensorflow-gpu==1.13.2`
instead.

Finally, set the `PYTHONPATH` environment variable to point to the project root
directory:

```bash
export PYTHONPATH=/path/to/root/dir/of/this/repo:$PYTHONPATH
```

## Overview

At a high level, Typilus consists of two parts:

1. Front-end, which parses Python programs, represents them in graphs, and
   splits the graphs into training set, validation set, and test set.
2. Back-end, which takes the graphs as inputs and trains a model or outputs
   predictions.

> The end-to-end process (corpus collection, training, and inference) takes a
> few days on our corpus. We therefore suggest that you start by replicating
> these steps for a small dataset.

### Prepare Data

This process prepares inputs to the machine learning back-end.

Go to the [README](data_preparation/README.md) in the `data_prepration` for
details on replicating the data preparation process.

> We cannot legally redistribute the data (=code), so we instead provide exact
> instructions for replicating our data collection and preparation to produce
> exactly the same corpus we used.

### Deep Learning Model

You can train a model by running

```bash
./tensorize_and_train.sh MODEL_TYPE INPUT_SPLIT_DATA TARGET_PATH_TO_TENSORIZE_AND_SAVE_MODEL 
```

where `INPUT_SPLIT_DATA` is the folder containing the `train/` and `valid/`
portions of the dataset, `MODEL_TYPE` is one of the model types in
`typilus/model/model_restore_helper.py` and
`TARGET_PATH_TO_TENSORIZE_AND_SAVE_MODEL` points to an arbitrary folder where
the tensorised data.

The script will tensorise and train the appropriate model and will indicate the
path of the saved model. If the model needs a type map, jump to
[Section: Create the Type Map](#typemap).

For finer-grained control of the tensorisation and training process, instead of
using the `tensorize_and_train.sh` script, follow the steps below.

#### Tensorise

First, convert the graphs into a tensor format by running:

```bash
python3 typilus/utils/tensorise.py --model MODEL_TYPE tensorized-data/train data_preparation/graph-dataset-split/train
python3 typilus/utils/tensorise.py --model MODEL_TYPE tensorized-data/valid data_preparation/graph-dataset-split/valid --metadata-to-use tensorized-data/train/metadata.pkl.gz
```

where `tensorized-data` is the folder where Typilus stores its tensorised
outputs.

A list of available model types can be found at
`typilus/model/model_restore_helper.py`. The Typilus model corresponds to
`graph2hybridmetric`. You must generate a _different_ tensorised version for
each model.

> Since many parts of TensorFlow 1.x have been deprecated during the migration
> to TensorFlow 2.0, the code emits multiple warnings that can be safely
> ignored.

#### Train a Model

To train a model on the data, run:

```bash
python3 typilus/utils/train.py --model MODEL_TYPE SAVE_FOLDER TENSORIZED_TRAIN_DATA_PATH TENSORIZED_VALID_DATA_PATH
```

`SAVE_FOLDER` is an arbitrary folder where to checkpoints and the final model
will be saved. `MODEL_TYPE` must match the one used to tensorise data.

> If you do *not* have a GPU, training will take substantial time. To select the
> id of the GPU device to use, set the `CUDA_VISIBLE_DEVICES` environment
> variable or follow the alternative methods provided by TensorFlow 1.x.

When running, `train.py` reports the exact location of the saved model.

### <a name="typemap"></a>Create the Type Map

If you have trained a model that employs a type map (_e.g._ Typilus), an
indexing step is necessary to build the type map:

```bash
python3 typilus/utils/index.py MODEL_PATH DATA_PATH
```

This script updates the model with a trained index which is ready for use. Here,
`DATA_PATH` should point to the directory containing the raw `.jsonl.gz` files
that should be indexed, commonly this is a folder containing the training and
validation data but _not_ the test data.

### Test the Model

Finally, to retrieve a model's predictions of a model, run:

```bash
python3 typilus/utils/predict.py MODEL_PATH TEST_DATA_PATH OUTPUT_JSON_PATH
```

Here `TEST_DATA_PATH` should point to the folder containing the pre-tensorised
`.jsonl.gz` data. This code will output a `jsonl.gz` with type predictions and
confidences with all values. `OUTPUT_JSON_PATH` must have a `.jsonl.gz`
suffix.

The output file can then be used for the type checking experiments described in
the PLDI 2020 paper. Look at the `README.md` file in `exp/type_check` for
further instructions.
