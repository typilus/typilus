#!/usr/bin/env python3

import argparse
import json
import logging
import os
import os.path as op
from itertools import chain, product

from dpu_utils.utils.dataloading import load_jsonl_gz

from annotater import Annotater
from tcmanager import MypyManager, PytypeManager, PyrightManager
from utils import ordinal


def parse_args():
    NUM_OF_PREDS = 5  # Can be calculated after loading the prediction file
    parser = argparse.ArgumentParser(description="Assess type suggestions.")
    parser.add_argument(
        "tc",
        choices={"mypy", "pytype", "pyright"},
        help="The chosen static type checker.",
    )
    parser.add_argument(
        "-i",
        "--ipaths",
        nargs="+",
        help="The paths to the files/directories to annotate and type check. If not set, use all paths specified in the prediction output.",
    )
    parser.add_argument(
        "-p", "--ppath", required=True, help="The path to the predicted types."
    )
    parser.add_argument(
        "-c", "--corpuspath", required=True, help="The path to the directory that contains the collected Python corpus."
    )
    parser.add_argument(
        "-g",
        "--granularity",
        default="var",
        choices={"var", "func", "file"},
        help="The granularity for adding type annotations.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        choices=set(range(1, NUM_OF_PREDS + 1)),
        help="Top K predictions to explore.",
    )
    parser.add_argument(
        "--prep", action="store_true", help="Prepare valid Python paths."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Type checking timeout threshold in seconds.",
    )
    return parser.parse_args()


def process_args(args):
    PRED_EXT = ".jsonl.gz"
    assert args.ppath.endswith(PRED_EXT)
    assert op.exists(args.ppath)
    if not op.isabs(args.ppath):
        args.ppath = op.abspath(args.ppath)
    args.valid_pypaths_file = op.join(
        op.dirname(args.ppath), f"{args.tc}_valid_pypaths.txt"
    )
    assert op.isdir(args.corpuspath)
    args.corpuspath = op.abspath(args.corpuspath)

    if args.ipaths is not None:
        assert all(op.exists(ipath) for ipath in args.ipaths)

    TC_MANAGERS = {
        "mypy": MypyManager,
        "pytype": PytypeManager,
        "pyright": PyrightManager,
    }
    args.TCManager = TC_MANAGERS[args.tc]

    return args


def config_logger(log_file_name):
    # create file handler
    LOG_DIR_PATH = op.join(op.dirname(op.abspath(__file__)), "log")
    if not op.exists(LOG_DIR_PATH):
        os.makedirs(LOG_DIR_PATH)
    log_path = op.join(LOG_DIR_PATH, log_file_name)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[fh, ch],
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


def prepare_valid_pypaths(args):
    logger = config_logger(f"{args.tc}_filter.log")
    logger.info("Preparing valid Python paths.")
    tcmanager = args.TCManager(args.tc, args.timeout)
    pypaths = {
        op.join(args.corpuspath, pred["provenance"])
        for pred in load_jsonl_gz(args.ppath)
    }
    valid_pypaths = {pp for pp in pypaths if tcmanager.light_assess(pp)}

    logger.info("=" * 10)
    logger.info(f"Saving valid paths in {args.valid_pypaths_file}.")
    with open(args.valid_pypaths_file, "w", encoding="utf8") as file:
        for vp in valid_pypaths:
            try:
                file.write(vp + "\n")
            except UnicodeEncodeError:
                logger.error(f"Path {vp} cannot be encoded with UTF-8!")
                continue

    logger.info("Prepared valid Python paths.")


def get_py_paths(ipath):
    PY_EXT = ".py"
    if op.isdir(ipath):
        for root, _, files in os.walk(ipath):
            for file in files:
                if file.endswith(PY_EXT):
                    yield op.abspath(op.join(root, file))
    else:
        assert ipath.endswith(PY_EXT)
        yield op.abspath(ipath)


def prepare_iters(args):
    assert op.isfile(
        args.valid_pypaths_file
    ), "prepare valid Python paths for the chosen type checker first."
    with open(args.valid_pypaths_file, encoding="utf8") as file:
        # ! If an argument in join is an absolute path, all previous components
        # ! are thrown away and joining continues from the absolute path.
        # ! See https://docs.python.org/3/library/os.path.html
        valid_pypaths = {op.join(args.corpuspath, l.strip()) for l in file}

    if args.ipaths is None:
        pypaths = valid_pypaths
    else:
        pypaths = valid_pypaths & set(
            chain.from_iterable(map(get_py_paths, args.ipaths))
        )

    pred_pypaths = [pred["provenance"] for pred in load_jsonl_gz(args.ppath)]
    if args.granularity == "var":
        pypath_predidx_gen = (
            (pp, predidx)
            for predidx, relpp in enumerate(pred_pypaths)
            for pp in pypaths
            if pp.endswith(relpp)
        )
        return list(product(pypath_predidx_gen, range(args.top)))
    elif args.granularity == "file":
        return [((pp, -1), 0) for pp in pypaths]
    else:  # when granularity is "func"
        raise NotImplementedError


def type_check(args):
    logger = config_logger(f"{args.tc}_tc.log")
    logger.info("Starting the type checking experiment.")
    annotater = Annotater(args.tc, args.ppath, args.granularity)
    tcmanager = args.TCManager(args.tc, args.timeout)

    iters = prepare_iters(args)
    ITER_TIMES = len(iters)
    assert ITER_TIMES > 0

    ANN_DELIMETER = "-" * 10
    for iter_idx, ((fpath, pred_idx), type_idx) in enumerate(iters):
        logger.info(ANN_DELIMETER)
        logger.debug(f"Iteration {iter_idx+1} of {ITER_TIMES}:")
        logger.info(f"On {fpath} with {ordinal(pred_idx+1)} prediction.")
        new_fpath = annotater.annotate(fpath, pred_idx, type_idx)
        if new_fpath != fpath:
            logger.info(f"Type checking {op.basename(new_fpath)}.")
            tcmanager.heavy_assess(new_fpath)
        else:
            logger.info("Skipped type checking: no annotation added.")

    logger.info(ANN_DELIMETER)
    logger.info("Finished the type checking experiment.")


def main():
    args = process_args(parse_args())
    if args.prep:
        prepare_valid_pypaths(args)
    else:
        type_check(args)


if __name__ == "__main__":
    main()
