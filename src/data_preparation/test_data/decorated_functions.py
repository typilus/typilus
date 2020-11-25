from abc import ABC, abstractmethod


def custom_decorator(func):
    def wrapper():
        func()

    return wrapper


def another_custom_decorator(func):
    def wrapper():
        func()

    return wrapper


class ClassWithDecorator(ABC):

    @abstractmethod
    def abstract_method(self):
        pass

    @staticmethod
    def static_method(self):
        pass

    @classmethod
    def class_method(cls):
        pass

    @custom_decorator
    def custom_method(self):
        pass


@custom_decorator
def decorated_function():
    pass


@custom_decorator
@another_custom_decorator
def double_decorated_function():
    pass
