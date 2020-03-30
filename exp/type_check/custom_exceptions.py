class CustomError(Exception):
    pass


class FileNonExisting(CustomError):
    def __init__(self):
        super().__init__("File not existing!")


class Py3Incompatible(CustomError):
    def __init__(self):
        super().__init__("File incompatible with Python 3!")


class TypeCheckingTooLong(CustomError):
    def __init__(self):
        super().__init__("Type checking file taking too long!")


class CheckerCrash(CustomError):
    def __init__(self):
        super().__init__("Type checker crashed!")


class CheckerConfigError(CustomError):
    def __init__(self):
        super().__init__("Failed to read type checker's config file!")


class OutputParseError(CustomError):
    def __init__(self):
        super().__init__("Failed to parse type checking output!")


class CustomWarning(Exception):
    pass


class FailToTypeCheck(CustomWarning):
    def __init__(self):
        super().__init__("File containing type errors!")
