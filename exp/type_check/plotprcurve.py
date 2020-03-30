#!/usr/bin/env python
"""
Usage:
    plotprcurve.py [options] MYPY_RESULT_PATH PYTYPE_RESULT_PATH

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from sklearn.metrics import precision_recall_curve


class MetricForPrecRec:
    def __init__(self, name: str) -> None:
        self.__name = name
        self.__collected_is_correct = []  # type: List[bool]
        self.__collected_confidences = []  # type: List[float]

    def add(self, is_correct: bool, confidence: float) -> None:
        self.__collected_is_correct.append(is_correct)
        self.__collected_confidences.append(confidence)

    def get_pr_curve(self, num_buckets: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        assert len(self.__collected_confidences) == len(self.__collected_is_correct)
        pr_curve = precision_recall_curve(
            self.__collected_is_correct, self.__collected_confidences
        )
        return (self.__name,) + pr_curve


def get_metric(name, respath):
    metric = MetricForPrecRec(name)
    res = respath.read_as_json()
    for r in res:
        is_correct = r["no_type_errs"] == 0
        prob = r["prob"]
        metric.add(is_correct, prob)
    return metric


def run(arguments):
    mypy_res_path = RichPath.create(arguments["MYPY_RESULT_PATH"])
    pytype_res_path = RichPath.create(arguments["PYTYPE_RESULT_PATH"])

    mypy_metric = get_metric("Correct against mypy", mypy_res_path)
    pytype_metric = get_metric("Correct against pytype", pytype_res_path)

    pr_curves = [
        mypy_metric.get_pr_curve() + ("r--",),
        pytype_metric.get_pr_curve() + ("b:",),
    ]

    fig = plt.figure(figsize=(5.5, 2.5))
    ax = fig.add_subplot(111)
    for name, precision, recall, _, style in pr_curves:
        ax.plot(recall, precision, style, label=name, linewidth=2)
    plt.grid()
    plt.xlim([0, 1.005])
    plt.ylim([0, 1.005])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test.pdf", dpi=300)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
