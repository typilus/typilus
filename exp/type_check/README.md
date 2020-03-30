# Type Checking Experiment for Typilus

An experiment that uses real-world, popular optional type checkers to assess the
types Typilus predicts.

## Setup

We require an environment with **Python 3.7 or later**.

First, in the **parent directory** of the directory of this README, create a
directory `raw_repos/`, which stores the Python repos. These repos should follow
the naming convention specified in the prediction file that has the extension
`jsonl.gz`.

Then, optionally, create a Python virtual environment:

```bash
python3 -m venv ./env
source ./env/bin/activate
```

Install required modules:

```bash
pip3 install -r requirements.txt
```

## Preprocessing

Some programs in `raw_repos/` fail to type check under mypy or pytype. This
preprocessing filters them. Assuming you are in the directory of this README,
run:

```bash
python3 run_exp.py SELECTED_TYPE_CHECKER -p PATH/TO/PREDICTION/FILE -c PATH/TO/PREDICTION/CORPUS --prep
```

The script `run_exp.py` has a few arguments, where

* the positional argument `SELECTED_TYPE_CHECKER` specifies the optional type
  checker this experiment uses (either **mypy** or **pytype**);
* the **required** option `-p` specifies the path to the prediction file that
  has the extension `.jsonl.gz`;
* the **required** option `-c` specifies the path to the corpus (i.e.
  `raw_repos/` created in data preparation), which contains the collected Python
  repos;
* the flag `--prep` instructs `run_exp.py` to preprocess the Python programs;
* the option `--timeout` specifies the maximum time in seconds of type checking
  a program, because pytype can spend hours or even days type checking a single
  file.

Run `python3 run_exp.py -h` for more information about other arguments.

With the flag `--prep`, the script saves the Python programs that type check
in a text file called `mypy[pytype]_valid_pypaths.txt` and stores the text file
in the directory containing the prediction file.

## Running the Experiment

To run the type checking experiment, issue:

```bash
python3 run_exp.py SELECTED_TYPE_CHECKER -p PATH/TO/PREDICTION/FILE -c PATH/TO/PREDICTION/CORPUS
```

This command saves the results of the type checking experiment in a log file,
such as `mypy_tc.log`, in the directory `./log/`. Use the provided
`logparser.py` to parse the log and produce statistics of interest:

```bash
python3 logparser.py PATH/TO/LOG/FILE PATH/TO/PREDICTION/FILE
```
