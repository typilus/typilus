# /usr/bin/env python3

import argparse
import json
import os
import os.path as op
import re
from ast import literal_eval

from dpu_utils.utils.dataloading import load_jsonl_gz


def get_valid_paths(args):
    log_path, tc = args.logpath, args.tc
    tested_pypaths_file = op.join(op.dirname(log_path), f"{tc}_tested_pypaths.txt")
    valid_pypaths_file = op.join(op.dirname(log_path), f"{tc}_valid_pypaths.txt")

    with open(log_path) as file:
        lines = file.read().splitlines()

    tested_pypaths = []
    valid_pypaths = []
    for idx, line in enumerate(lines):
        if "tcmanager INFO: Light assessing" in line:
            if idx + 1 >= len(lines):
                break
            fpath = line.split()[-1]
            if op.isfile(fpath):
                tested_pypaths.append(fpath)
                if "tcmanager INFO: Passed the light assessment." in lines[idx + 1]:
                    valid_pypaths.append(fpath)

    with open(tested_pypaths_file, "w") as file:
        for fpath in tested_pypaths:
            file.write(fpath + "\n")

    with open(valid_pypaths_file, "w") as file:
        for fpath in valid_pypaths:
            file.write(fpath + "\n")


def parse_tc_log(args):
    predlines = list(load_jsonl_gz(args.predpath))
    with open(args.logpath) as file:
        loglines = file.read().splitlines()

    results = []
    odds = []
    DELIMITER = "-" * 10
    LEN = len(loglines)
    for log_idx, line in enumerate(loglines):
        if log_idx + 5 >= LEN:
            break

        if not (
            line.endswith(DELIMITER)
            and "tcmanager INFO: Produced" in loglines[log_idx + 4]
        ):
            continue
        file_line = loglines[log_idx + 1]
        assert file_line.endswith("prediction.")
        pred_idx_str = file_line.split()[-2][:-2]
        assert pred_idx_str.isdigit()
        pred_idx = int(pred_idx_str) - 1
        orig_type = predlines[pred_idx]["original_annotation"]

        anno_line = loglines[log_idx + 2]
        anno_pat = r"' with '([\w\[\]\., %]+)' of (\d\.\d+) at"
        m = re.search(anno_pat, anno_line)
        assert bool(m)
        pred_type = m.group(1)
        prob = float(m.group(2))

        tc_res_line = loglines[log_idx + 4]
        tc_res_pat = r"Produced (\d+) type error"
        m = re.search(tc_res_pat, tc_res_line)
        assert bool(m)
        no_type_errs = int(m.group(1))

        if "tcmanager INFO: Error breaking down:" in loglines[log_idx + 5]:
            err_bd_line = loglines[log_idx + 5]
            err_bd_pat = r"Error breaking down: (.+)\.$"
            m = re.search(err_bd_pat, err_bd_line)
            err_bd = literal_eval(m.group(1))
        else:
            err_bd = None

        # ! These are corner cases, where though our predictions are the same as
        # ! the original ones, they still cause type errors, due to various
        # ! reasons
        if orig_type is not None and pred_type == orig_type and no_type_errs != 0:
            odds.append(file_line)
            continue

        results.append(
            {
                "orig_type": orig_type,
                "pred_type": pred_type,
                "pred_idx": pred_idx,
                "prob": prob,
                "no_type_errs": no_type_errs,
                "err_bd": err_bd,
            }
        )

    return results, odds


def calc_stats(args):
    results, odds = parse_tc_log(args)

    result_dir_path = op.join(op.dirname(op.abspath(__file__)), "results")
    os.makedirs(result_dir_path, exist_ok=True)

    with open(op.join(result_dir_path, f"{args.tc}_results.json"), "w") as resfile:
        json.dump(results, resfile)

    with open(op.join(result_dir_path, f"{args.tc}_corner_cases.txt"), "w") as oddfile:
        for odd in odds:
            oddfile.write(odd + "\n")

    LEN = len(results)
    print(f"Number of results: {LEN}")
    print(f"Number of corner cases: {len(odds)}")

    ratio = cor_ratio(results)
    print(f"General ratio of correct predictions: {ratio}")

    add_results = [r for r in results if r["orig_type"] is None]
    ratio = round(len(add_results) / LEN, 2)
    print(f"Ratio of ε -> t: {ratio}")
    if len(add_results) > 0:
        ratio = cor_ratio(add_results)
        print(f"For ε -> t, ratio of correct predictions: {ratio}")

    diff_results = [
        r
        for r in results
        if r["orig_type"] is not None and r["orig_type"] != r["pred_type"]
    ]
    ratio = round(len(diff_results) / LEN, 2)
    print(f"Ratio of t -> t': {ratio}")
    if len(diff_results) > 0:
        ratio = cor_ratio(diff_results)
        print(f"For t -> t', ratio of correct predictions: {ratio}")

    same_results = [
        r
        for r in results
        if r["orig_type"] is not None and r["orig_type"] == r["pred_type"]
    ]
    ratio = round(len(same_results) / LEN, 2)
    print(f"Ratio of t -> t: {ratio}")
    if len(same_results) > 0:
        ratio = cor_ratio(same_results)
        print(f"For t -> t, ratio of correct predictions: {ratio}")

    PROB_THRESHOLD = 0.9
    conf_results = [r for r in results if r["prob"] >= PROB_THRESHOLD]
    ratio = cor_ratio(conf_results)
    print(f"For prob >= {PROB_THRESHOLD}, ratio of correct predictions: {ratio}")


def cor_ratio(results):
    cor_results = [r for r in results if r["no_type_errs"] == 0]
    return round(len(cor_results) / len(results), 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse logs of the type checking experiment."
    )
    parser.add_argument("logpath", help="the path to the log")
    parser.add_argument("predpath", help="the path to the predictions")

    args = parser.parse_args()

    assert op.isfile(args.logpath), "log file doesn't exist!"
    assert op.isfile(args.predpath), "prediction file doesn't exist!"
    logname = op.basename(args.logpath)
    logname_pat = r"^([a-z]+)_([a-z]+)\.log$"
    m = re.match(logname_pat, logname)
    assert bool(m), "log's file name is unexpected (should've been 'tc_type.log')!"
    tc = m.group(1)
    assert tc in {"mypy", "pytype"}
    args.tc = tc
    parsetype = m.group(2)
    assert parsetype in {"filter", "tc"}
    args.parsetype = parsetype
    return args


def main():
    HANDLER = {"filter": get_valid_paths, "tc": calc_stats}
    args = parse_args()
    HANDLER[args.parsetype](args)


if __name__ == "__main__":
    main()
