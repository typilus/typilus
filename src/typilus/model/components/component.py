from abc import ABC, abstractmethod
from typing import Any, Dict, List

import tensorflow as tf


class Component(ABC):
    @staticmethod
    @abstractmethod
    def make_placeholders(name: str, placeholder_dict: Dict[str, Any], hyperparameters: Dict[str, Any]):
        pass

    @staticmethod
    @abstractmethod
    def make_parameters(name: str, parameters: Dict[str, tf.Tensor], metadata: Dict[str, Any], hyperparameters: Dict[str, Any]):
        pass

    @staticmethod
    @abstractmethod
    def make_model(name: str, placeholder_dict: Dict[str, tf.Tensor], parameters: Dict[str, tf.Tensor], hyperparameters: Dict[str, Any],
                   is_train: bool = True):
        pass

    @staticmethod
    @abstractmethod
    def init_metadata(name: str, raw_metadata: Dict[str, Any], hyperparameters: Dict[str, Any]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load_metadata_from_sample(name: str, raw_sample: Any, raw_metadata: Dict[str, Any],
                                  hyperparameters: Dict[str, Any]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def finalise_metadata(name: str, raw_metadata_list: List[Dict[str, Any]], final_metadata: Dict[str, Any],
                          hyperparameters: Dict[str, Any]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load_data_from_sample(name: str, metadata: Dict[str, Any], data: Any,
                              result_holder: Dict[str, Any], hyperparameters: Dict[str, Any], is_train: bool = True) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def init_minibatch(name: str, batch_data: Dict[str, Any], hyperparameters: Dict[str, Any]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def extend_minibatch_by_sample(name: str, batch_data: Dict[str, Any], sample: Dict[str, Any],
                                   hyperparameters: Dict[str, Any]) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def finalise_minibatch(name: str, batch_data: Dict[str, Any], placeholders: Dict[str, tf.Tensor], minibatch: Dict[tf.Tensor, Any],
                           hyperparameters: Dict[str, Any], is_train: bool) -> None:
        pass