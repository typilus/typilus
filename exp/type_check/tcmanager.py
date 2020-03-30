#!/usr/bin/env python3

from abc import ABC, abstractmethod
import logging
import os
import os.path as op
import re
import subprocess
from collections import Counter, namedtuple
from getpass import getuser

import toml

from custom_exceptions import (
    CustomError,
    FileNonExisting,
    Py3Incompatible,
    TypeCheckingTooLong,
    CheckerCrash,
    CheckerConfigError,
    OutputParseError,
    CustomWarning,
    FailToTypeCheck,
)

fields = ("no_type_errs", "no_files", "no_ignored_errs", "no_warnings", "err_breakdown")
ParsedResult = namedtuple("ParsedResult", fields, defaults=(None,) * len(fields))


class TCManager(ABC):
    def __init__(self, tc, timeout):
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)
        errcodes = toml.load("metadata/errcodes.toml")[tc]
        self._all_errcodes = errcodes["all"]
        self._inc_errcodes = errcodes["included"]

    def _check_file_existence(self, fpath):
        if not op.isfile(fpath):
            raise FileNonExisting

    def _check_py3_compatibility(self, fpath):
        if subprocess.run(["python3", "-m", "py_compile", fpath]).returncode != 0:
            raise Py3Incompatible

    def _check_basics(self, fpath):
        self._check_file_existence(fpath)
        self._check_py3_compatibility(fpath)

    @abstractmethod
    def _build_tc_cmd(self, fpath):
        pass

    def _type_check(self, fpath):
        try:
            cwd = os.getcwd()
            os.chdir(op.dirname(fpath))
            result = subprocess.run(
                self._build_tc_cmd(op.basename(fpath)),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            retcode = result.returncode
            outlines = result.stdout.splitlines()
            return retcode, outlines
        except subprocess.TimeoutExpired:
            raise TypeCheckingTooLong
        finally:
            os.chdir(cwd)

    @abstractmethod
    def _check_tc_outcome(self, returncode, outlines):
        pass

    def light_assess(self, fpath):
        self._logger.info(f"Light assessing {fpath}.")
        try:
            self._check_basics(fpath)
            retcode, outlines = self._type_check(fpath)
            self._check_tc_outcome(retcode, outlines)
            self._logger.info("Passed the light assessment.")
            return True
        except CustomError as e:
            self._logger.error(str(e))
            return False
        except CustomWarning as e:
            self._logger.warning(str(e))
            return False

    @abstractmethod
    def _parse_tc_output(self, returncode, outlines):
        pass

    @abstractmethod
    def _report_errors(self, parsed_result):
        pass

    def heavy_assess(self, fpath):
        try:
            retcode, outlines = self._type_check(fpath)
            parsed_result = self._parse_tc_output(retcode, outlines)
            self._report_errors(parsed_result)
        except CustomError as e:
            self._logger.error(str(e))


class MypyManager(TCManager):
    def _build_tc_cmd(self, fpath):
        # Mypy needs a flag to display the error codes
        return ["mypy", "--show-error-codes", fpath]

    def _check_tc_outcome(self, _, outlines):
        if any(l.endswith(err) for l in outlines for err in self._inc_errcodes):
            raise FailToTypeCheck

    def _parse_tc_output(self, retcode, outlines):
        last_line = outlines[-1]
        err_breakdown = None
        if retcode == 0:
            if not last_line.startswith("Success: "):
                raise OutputParseError
            no_type_errs = 0
            no_files = next(int(w) for w in last_line.split() if w.isdigit())
            no_ignored_errs = 0
        else:
            c = Counter(
                err for l in outlines for err in self._inc_errcodes if l.endswith(err)
            )
            err_breakdown = dict(c)
            no_type_errs = sum(c.values())
            if last_line.startswith("Found ") and last_line.endswith(" source file)"):
                numbers = [int(s) for s in last_line.split() if s.isdigit()]
                no_errs = numbers[0]
                no_files = numbers[1]
                no_ignored_errs = no_errs - no_type_errs
            else:
                raise OutputParseError

        return ParsedResult(
            no_type_errs, no_files, no_ignored_errs, err_breakdown=err_breakdown
        )

    def _report_errors(self, parsed_result):
        self._logger.info(
            f"Produced {parsed_result.no_type_errs} type error(s) in {parsed_result.no_files} file(s)."
        )
        if parsed_result.err_breakdown:
            self._logger.info(f"Error breaking down: {parsed_result.err_breakdown}.")


class PytypeManager(TCManager):
    def _build_tc_cmd(self, fpath):
        ignored_errcodes = set(self._all_errcodes) - set(self._inc_errcodes)
        # Disable generating unrelated kinds of errors
        return ["pytype", fpath, "-d", ",".join(ignored_errcodes)]

    def _check_tc_outcome(self, returncode, outlines):
        if returncode != 0:
            raise FailToTypeCheck

    def _kill_pytype_procs(self):
        user_procs = subprocess.run(
            # ! Doesn't work on Windows.
            ["ps", "-U", getuser(), "-o", "pid,command"],
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
        pytype_pids = (p.split()[0] for p in user_procs if "pytype-single" in p)
        for pid in pytype_pids:
            if not pid.isdigit():
                self._logger.debug(f"Failed to parse pid; got a non-numeric: {pid}.")
                continue
            retcode = subprocess.run(["kill", pid]).returncode
            if retcode == 0:
                self._logger.debug(f"killed process {pid}.")
            else:
                self._logger.debug(f"Failed to kill process {pid}.")

    def _type_check(self, fpath):
        try:
            result = subprocess.run(
                self._build_tc_cmd(fpath),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            retcode = result.returncode
            outlines = result.stdout.splitlines()
            return retcode, outlines
        except subprocess.TimeoutExpired:
            self._kill_pytype_procs()
            raise TypeCheckingTooLong

    def _parse_tc_output(self, retcode, outlines):
        last_line = outlines[-1]
        err_breakdown = None
        if retcode == 0:
            if not last_line.startswith("Success: "):
                raise OutputParseError
            no_type_errs = 0
            penultimate_line = outlines[-2]
            # ! Currently doesn't consider dependencies as source files
            # TODO: handle the case when parsing the output fails
            no_files = next(int(s) for s in penultimate_line.split() if s.isdigit())
        else:
            c = Counter(
                err
                for l in outlines
                for err in self._inc_errcodes
                if l.endswith("[" + err + "]")
            )
            err_breakdown = dict(c)
            no_type_errs = sum(c.values())
            # ! Currently doesn't consider dependencies as source files
            # TODO: handle the case when parsing the output fails
            no_files = next(int(s) for s in last_line.split() if s.isdigit())

        return ParsedResult(no_type_errs, no_files, err_breakdown=err_breakdown)

    def _report_errors(self, parsed_result):
        self._logger.info(
            f"Produced {parsed_result.no_type_errs} type error(s) in {parsed_result.no_files} file(s); ignored {parsed_result.no_ignored_errs} other errors."
        )
        if parsed_result.err_breakdown is not None:
            self._logger.info(f"Error breaking down: {parsed_result.err_breakdown}.")


class PyrightManager(TCManager):
    def _build_tc_cmd(self, fpath):
        return ["pyright", fpath]

    def _check_tc_outcome(self, returncode, outlines):
        if returncode == 1:
            raise FailToTypeCheck
        elif returncode == 2:
            raise CheckerCrash
        elif returncode == 3:
            raise CheckerConfigError

    def _parse_tc_output(self, returncode, outlines):
        if returncode == 0:
            if len(outlines) != 4:
                raise OutputParseError
            second_line = outlines[1]
            no_files = next(int(s) for s in second_line.split() if s.isdigit())
            no_warnings = 0
        elif returncode == 1:
            second_line = outlines[1]
            no_files = next(int(s) for s in second_line.split() if s.isdigit())
            penultimate_line = outlines[-2]
            gen = (int(s) for s in penultimate_line.split() if s.isdigit())
            no_type_errs = next(gen)
            no_warnings = next(gen)
        # returncode shouldn't be 2 or 3 after the light assessment
        elif returncode == 2:
            raise CheckerCrash
        elif returncode == 3:
            raise CheckerConfigError

        return ParsedResult(no_type_errs, no_files, no_warnings=no_warnings)

    def _report_errors(self, parsed_result):
        self._logger.info(
            f"Produced {parsed_result.no_type_errs} type error(s) in {parsed_result.no_files} file(s); also produced {parsed_result.no_warnings} warnings."
        )
