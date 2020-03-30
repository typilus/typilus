#!/usr/bin/env python3

import inspect
import logging
import os.path as op
import typing
from enum import Enum
from itertools import chain
from math import exp
from re import findall

import astunparse
import typed_astunparse
from dpu_utils.utils.dataloading import load_jsonl_gz
from typed_ast.ast3 import (
    AnnAssign,
    Assign,
    Attribute,
    FunctionDef,
    Import,
    ImportFrom,
    Load,
    Name,
    NodeTransformer,
    Str,
    Subscript,
    Tuple,
    alias,
    arg,
    copy_location,
    dump,
    fix_missing_locations,
    parse,
)

from utils import rreplace


class AnnotationKind(Enum):
    PARA = "parameter"
    FUNC = "class-or-function"
    VAR = "variable"


class Annotater(NodeTransformer):
    # __PP = pprint.PrettyPrinter()  # debugging only
    __ANYS = {"Any", "typing.Any"}
    __NORETURNS = {"NoReturn", "typing.NoReturn"}
    __NONES = {"None", "type(None)"}
    __IGNORED_PRED_PROB = -1
    __MYPY_SPECIALS = {
        ("TypedDict", "mypy_extensions"),
        ("Final", "typing_extensions"),
        ("Literal", "typing_extensions"),
    }

    def __init__(self, tc, ppath, granularity):
        self.__tc = tc
        self.__ppath = ppath
        self.__predlines = list(load_jsonl_gz(self.__ppath))
        self.__granularity = granularity
        self.__type_idx = 0
        self.__rel_lines = []
        self.__used_types = set()
        self.__logger = logging.getLogger(__name__)
        self.__typing_types = self.__read_file("metadata/typing_types.txt")

    def __read_file(self, fpath):
        with open(fpath, encoding="utf8") as f:
            return {line.strip() for line in f}

    def __sift(self, fpath=None, pred_idx=None):
        if fpath is not None:
            rel_lines = filter(
                lambda p: fpath.endswith(p["provenance"]), self.__predlines
            )
        elif pred_idx is not None:
            rel_lines = [self.__predlines[pred_idx]]

        # ! If the term is a property access, don't annotate it,
        # ! because both mypy and pytype don't support this case.
        for rel_line in rel_lines:
            if "." in rel_line["name"]:
                self.__logger.warning("Ignoring property accesses.")
            else:
                self.__rel_lines.append(rel_line)

    def __get_types_2_import(self):
        used_atomic_types = set(
            chain.from_iterable(map(lambda t: findall(r"\w+", t), self.__used_types))
        )
        assert len(used_atomic_types) > 0
        return used_atomic_types & self.__typing_types

    # ! Python's importing mechanism is permissive: it allows duplicated imports;
    # ! it allows an alias import and a regular import for the same object to
    # ! coexist. So I don't bother to remove duplicated imports.
    # ! For example, the following two imports
    # ! "from typing import List
    # ! from typing import Ls"
    # ! allow List and Ls to coexist.
    def __add_type_imports(self, types_2_import):
        if self.__tc == "mypy":
            for t, m in self.__MYPY_SPECIALS:
                if t in types_2_import:
                    types_2_import.remove(t)
                    names = [alias(name=t, asname=None)]
                    import_mypy_special = ImportFrom(module=m, names=names, level=0)
                    self.__tree.body.insert(self.__insertion_idx, import_mypy_special)

        if len(types_2_import) > 0:
            names = [alias(name=t, asname=None) for t in types_2_import]
            import_types = ImportFrom(module="typing", names=names, level=0)
            self.__tree.body.insert(self.__insertion_idx, import_types)

        import_typing = Import(names=[alias(name="typing", asname=None)])
        self.__tree.body.insert(self.__insertion_idx, import_typing)

    def __reset(self):
        self.__rel_lines.clear()
        self.__used_types.clear()
        self.__unmodified = True
        self.__insertion_idx = 0

    # ! All "# type: ignore"s are saved in the top-level "type_ignores" attribute.
    def annotate(self, fpath, pred_idx, type_idx):
        self.__reset()

        if pred_idx == -1:
            self.__sift(fpath=fpath)
        else:
            self.__sift(pred_idx=pred_idx)

        # if no proper (i.e. non-property-access) predictions are for this file,
        # or the predictions are fewer than args.top
        if len(self.__rel_lines) == 0 or type_idx >= len(
            self.__rel_lines[0]["predicted_annotation_logprob_dist"]
        ):
            return fpath

        self.__type_idx = type_idx

        with open(fpath) as src:
            self.__tree = parse(src.read())
        # self.__PP.pprint(dump(tree))

        new_tree = self.visit(self.__tree)
        if self.__unmodified:
            return fpath
        self.__add_type_imports(self.__get_types_2_import())
        new_tree = fix_missing_locations(new_tree)

        OLD_EXT = ".py"
        NEW_EXT = f"_tpl_{type_idx}.py"
        new_fpath = rreplace(fpath, OLD_EXT, NEW_EXT, 1)
        with open(new_fpath, "w", encoding="utf8") as dst:
            dst.write(typed_astunparse.unparse(new_tree))

        return new_fpath

    def __get_index(self, name, lineno, kind):
        for idx, predline in enumerate(self.__rel_lines):
            if (
                predline["name"] == name
                and predline["location"][0] == lineno
                and predline["annotation_type"] == kind.value
            ):
                return idx
        return -1

    def __process_old_anns(self, node):
        # ! For One-at-a-Time, don't remove original type annotations
        if self.__granularity == "var":
            return node

        if isinstance(node, arg):
            return copy_location(
                arg(arg=node.arg, annotation=None, type_comment=None), node
            )
        elif isinstance(node, FunctionDef):
            return copy_location(
                FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=node.body,
                    decorator_list=node.decorator_list,
                    returns=None,
                    type_comment=None,
                ),
                node,
            )
        elif isinstance(node, AnnAssign):
            return copy_location(
                AnnAssign(
                    target=node.target,
                    annotation=None,
                    value=node.value,
                    simple=node.simple,
                ),
                node,
            )
        elif isinstance(node, Assign):
            return copy_location(
                Assign(
                    targets=node.targets,
                    annotation=None,
                    value=node.value,
                    type_comment=None,
                ),
                node,
            )
        else:
            return node

    def __skip_original_type(self, orig_type):
        # Ignore original "NoReturn"
        if orig_type in self.__NORETURNS:
            self.__logger.warning("Ignoring original 'NoReturn'.")
            return True
        # Ignore original "None"
        elif orig_type in self.__NONES:
            self.__logger.warning("Ignoring original 'None'.")
            return True
        else:
            return False

    def __skip_predicted_type(self, pred_type):
        # Ignore predicted "NoReturn"
        if pred_type in self.__NORETURNS:
            self.__logger.warning("Ignoring predicted 'NoReturn'.")
            return True
        # Ignore predicted "Any"
        elif pred_type in self.__ANYS:
            self.__logger.warning("Ignoring predicted 'Any'.")
            return True
        # Ignore corner case predictions, e.g. "..." 
        elif not findall(r"\w+", pred_type):
            self.__logger.warning(f"Ignoring prediction '{pred_type}'.")
            return True
        else:
            return False

    def __extract_type_and_prob(self, identifier, lineno, kind):
        pred_idx = self.__get_index(identifier, lineno, kind)

        if pred_idx == -1:
            return None, None

        pred_info = self.__rel_lines[pred_idx]
        orig_type = pred_info["original_annotation"]
        pred = pred_info["predicted_annotation_logprob_dist"][self.__type_idx]
        pred_type = pred[0]

        if self.__skip_original_type(orig_type) or self.__skip_predicted_type(
            pred_type
        ):
            return pred_type, self.__IGNORED_PRED_PROB

        pred_prob = round(exp(pred[1]), 2)
        assert 0 <= pred_prob <= 1

        self.__used_types.add(pred_type)
        self.__unmodified = False
        del self.__rel_lines[pred_idx]
        self.__logger.info(
            f"Annotating '{identifier}' with '{pred_type}' of {pred_prob:.2f} at line {lineno}."
        )

        return pred_type, pred_prob

    def visit_ImportFrom(self, node):
        if node.module == "__future__":
            # ! Potential insertion point for typing imports must be after.
            # ! a __future__ import.
            candidate_idx = self.__tree.body.index(node) + 1
            self.__insertion_idx = max(self.__insertion_idx, candidate_idx)

        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "__future__":
                candidate_idx = self.__tree.body.index(node) + 1
                self.__insertion_idx = max(self.__insertion_idx, candidate_idx)

        return node

    # ! arg exists in only Python 3; Python 2 uses Name.
    # ! See https://greentreesnakes.readthedocs.io/en/latest/nodes.html#arg
    def visit_arg(self, node):
        pred_type, pred_prob = self.__extract_type_and_prob(
            node.arg, node.lineno, AnnotationKind.PARA
        )

        if pred_prob is None:
            return self.__process_old_anns(node)
        elif pred_prob == self.__IGNORED_PRED_PROB:
            return node
        else:
            return copy_location(
                arg(
                    arg=node.arg,
                    annotation=Name(id=pred_type, ctx=Load()),
                    type_comment=None,
                ),
                node,
            )

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        pred_type, pred_prob = self.__extract_type_and_prob(
            node.name, node.lineno, AnnotationKind.FUNC
        )

        if pred_prob is None:
            return self.__process_old_anns(node)
        elif pred_prob == self.__IGNORED_PRED_PROB:
            return node
        else:
            return copy_location(
                FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=node.body,
                    decorator_list=node.decorator_list,
                    returns=Name(id=pred_type, ctx=Load()),
                    type_comment=None,
                ),
                node,
            )

    def visit_AnnAssign(self, node):
        self.generic_visit(node)

        target = node.target
        varname = (
            target.id
            if isinstance(target, Name)
            else astunparse.unparse(target).strip()
        )

        pred_type, pred_prob = self.__extract_type_and_prob(
            varname, node.lineno, AnnotationKind.VAR
        )

        if pred_prob is None:
            return self.__process_old_anns(node)
        elif pred_prob == self.__IGNORED_PRED_PROB:
            return node
        else:
            return copy_location(
                AnnAssign(
                    target=node.target,
                    annotation=Name(id=pred_type, ctx=Load()),
                    value=node.value,
                    simple=node.simple,
                ),
                node,
            )

    def visit_Assign(self, node):
        self.generic_visit(node)

        targets = node.targets
        # ! Consider only the case when "targets" has only one non-Tuple element
        if len(targets) > 1 or isinstance(targets[0], Tuple):
            return node
        varname = astunparse.unparse(targets[0]).strip()

        pred_type, pred_prob = self.__extract_type_and_prob(
            varname, node.lineno, AnnotationKind.VAR
        )

        if pred_prob is None:
            return self.__process_old_anns(node)
        elif pred_prob == self.__IGNORED_PRED_PROB:
            return node
        else:
            # ! Assume no type comments on AnnAssign nodes
            return copy_location(
                AnnAssign(
                    target=node.targets[0],
                    annotation=Name(id=pred_type, ctx=Load()),
                    value=node.value,  # TODO: validate
                    simple=0,
                ),
                node,
            )

    # def visit_Name(self, node):
    #     # TODO: parameter annotation for Python 2
    #     return node

    # def visit_For(self, node):
    #     # TODO: type comments are valid for "for" statement
    #     self.generic_visit(node)
    #     return node

    # def visit_With(self, node):
    #     # TODO: type comments are valid for "with" statement
    #     self.generic_visit(node)
    #     return node
