import argparse
from dpu_utils.utils import load_jsonl_gz
from dpu_utils.utils import ChunkWriter
import hashlib
import logging
import os
from tqdm import tqdm
from glob import iglob
from typing import Set


def get_fold( filename: str, train_ratio: float, valid_ratio: float) -> str:
    # Copied from: https://github.com/microsoft/graph-based-code-modelling/blob/master/Models/utils/dataset_split.py#L24

    hash_val = int(hashlib.md5(filename.encode(errors='ignore')).hexdigest(), 16) % (2 ** 16)
    train_bound = int(2 ** 16 * train_ratio)
    if hash_val <= train_bound:
        return "train"
    elif hash_val <= train_bound + int(2 ** 16 * valid_ratio):
        return "valid"
    else:
        return "test"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", help="path to data directory containing .gz files")
    parser.add_argument("-out-dir", help="path to output directory")
    parser.add_argument(
        "-train-pct",
        type=float,
        default=0.7,
        help="ratio of training set to whole corpus",
    )
    parser.add_argument(
        "-valid-pct",
        type=float,
        default=0.1,
        help="ratio of validation set to whole corpus",
    )
    args = parser.parse_args()

    num_train, num_valid, num_test = 0, 0, 0
    with ChunkWriter(os.path.join(args.out_dir, 'train'), file_prefix='graph-', max_chunk_size=1000, file_suffix='.jsonl.gz') as train_w,\
        ChunkWriter(os.path.join(args.out_dir, 'valid'), file_prefix='graph-', max_chunk_size=1000, file_suffix='.jsonl.gz') as valid_w,\
        ChunkWriter(os.path.join(args.out_dir, 'test'), file_prefix='graph-', max_chunk_size=1000, file_suffix='.jsonl.gz') as test_w:
        for f in tqdm(iglob(os.path.join(args.data_dir, "*.jsonl.gz"))):
            for ex in load_jsonl_gz(f):
                partition = get_fold(
                    ex["filename"], args.train_pct, args.valid_pct
                )
                if partition == "train":
                    train_w.add(ex)
                    num_train += 1
                elif partition == "valid":
                    valid_w.add(ex)
                    num_valid += 1
                else:
                    test_w.add(ex)
                    num_test += 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
    )
    logging.info("Train: %d", num_train)
    logging.info("Valid: %d", num_valid)
    logging.info("Test: %d", num_test)
    logging.info("Total: %d", num_train + num_valid + num_test)
