from typing import List, Dict, Any, Tuple, Optional

import tensorflow as tf
from dpu_utils.utils import RichPath

from typilus.model.typeclassificationmodel import TypeClassificationModel
from typilus.model.utils import ignore_type_annotation
from .pathbasedmodel import PathBasedModel
from .typemetriclearningmodel import TypeMetricLearningModel


class Path2Metric(PathBasedModel):
    @staticmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Path2Metric, Path2Metric)._init_metadata(hyperparameters, raw_metadata)

    @staticmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any],
                                   raw_metadata: Dict[str, Any]) -> None:
        super(Path2Metric, Path2Metric)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)

    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return super()._finalise_metadata(raw_metadata_list)

    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = PathBasedModel.get_default_hyperparameters()
        defaults.update({
            'max_type_annotation_vocab_size': 100,
            'margin': 2
        })
        return defaults

    def __init__(self, hyperparameters, run_name: Optional[str] = None, model_save_dir: Optional[str] = None, log_save_dir: Optional[str] = None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)
        self.__type_classification = TypeClassificationModel(self)

        self.__type_metric = TypeMetricLearningModel(self,
                                                     type_representation_size=self.hyperparameters['path_encoder_lstm_hidden_size'],
                                                     margin=self.hyperparameters['margin'])

    def _make_parameters(self):
        super()._make_parameters()

    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)
        self.__type_metric._make_placeholders(is_train)


    def _make_model(self, is_train: bool = True):
        super()._make_model(is_train)
        self.__type_metric._make_model(self.ops['target_variable_representations'], is_train)

    @staticmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool = True) -> bool:
        keep_sample = super(Path2Metric, Path2Metric)._load_data_from_sample(
            hyperparameters, metadata, raw_sample, result_holder, is_train)
        if not keep_sample:
            return False

        target_class = []
        for node_idx, annotation_data in raw_sample['supernodes'].items():
            annotation = annotation_data['annotation']
            if is_train and ignore_type_annotation(annotation):
                continue
            target_class.append(annotation)

        result_holder['target_type'] = target_class
        return len(target_class) > 0


    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)
        self.__type_metric._init_minibatch(batch_data)

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        batch_finished = super()._extend_minibatch_by_sample(batch_data, sample)
        self.__type_metric._extend_minibatch_by_sample(batch_data, sample)
        return batch_finished

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)
        self.__type_metric._finalise_minibatch(batch_data, is_train, minibatch)
        return minibatch

    def create_index(self, data_paths: List[RichPath]):
        self.__type_metric.create_index(data_paths, self.metadata)

    # ------- These are the bits that we only need for test-time:
    def _encode_one_test_sample(self, sample_data_dict: Dict[tf.Tensor, Any]) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return (self.sess.run(self.ops['target_representations'],
                              feed_dict=sample_data_dict),
                None)

    def annotate_single(self, raw_sample: Dict[str, Any], loaded_test_sample: Dict[str, Any], provenance: str):
        return self.__type_metric.annotate_single(raw_sample, loaded_test_sample, provenance, self.metadata)
