#!/usr/bin/env python3
import argparse
from bs4 import BeautifulSoup
from collections import namedtuple
import os.path as op
import re
import requests
import toml


Setting = namedtuple("Setting", ["name", "url", "inc_errcodes", "errcode_pat"])

MYPY = Setting(
    name="mypy",
    url="https://github.com/python/mypy/blob/master/docs/source/error_code_list.rst",
    # https://docs.google.com/document/d/1szpNSARdq7RpuMHvpnPiTmpm05uvQFAA9kPAgIuWXYk/edit#
    inc_errcodes="""[attr-defined]
[union-attr]
[call-args]
[arg-type]
[call-overload]
[valid-type]
[override]
[return]
[return-value]
[assignment]
[type-var]
[operator]
[index]
[list-item]
[name-defined]
[dict-item]
[typeddict-item]
[func-returns-value]
[valid-newtype]
[exit-return]
[misc]""",
    errcode_pat=r"\[[\w-]+\]",
)

PYTYPE = Setting(
    name="pytype",
    url="https://github.com/google/pytype/blob/master/docs/errors.md",
    # https://docs.google.com/document/d/1szpNSARdq7RpuMHvpnPiTmpm05uvQFAA9kPAgIuWXYk/edit#
    inc_errcodes="""annotation-type-mismatch
attribute-error
bad-concrete-type
bad-function-defaults
bad-return-type
bad-slots
bad-unpacking
base-class-error
duplicate-keyword-argument
ignored-type-comment
invalid-annotation
invalid-function-type-comment
invalid-namedtuple-arg
invalid-type-comment
invalid-typevar
key-var
missing-parameter
name-error
not-callable
not-indexable
redundant-function-type-comment
unsupported-operands
wrong-arg-count
wrong-arg-types
wrong-keyword-args""",
    errcode_pat=r"(\w+-)+\w+",
)


def main():
    errcodes = {"title": "Error codes"}
    for setting in {MYPY, PYTYPE}:
        response = requests.get(setting.url)
        soup = BeautifulSoup(response.text, "html.parser")

        err_codes = set()
        for h2 in soup.find_all("h2"):
            match = re.search(setting.errcode_pat, h2.text)
            if hasattr(match, "group"):
                err_codes.add(match.group(0))

        errcodes[setting.name] = {
            "all": sorted(list(err_codes)),
            "included": sorted(setting.inc_errcodes.splitlines()),
        }

    opath = op.join(op.dirname(op.abspath(__file__)), "errcodes.toml")
    with open(opath, "w") as ofile:
        toml.dump(errcodes, ofile)


if __name__ == "__main__":
    main()
