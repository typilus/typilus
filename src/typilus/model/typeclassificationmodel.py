from abc import ABC
from collections import Counter
from typing import Dict, Any, List, Optional

import tensorflow as tf
from dpu_utils.mlutils import Vocabulary

from typilus.model.model import write_to_minibatch, Model
from typilus.model.utils import ignore_type_annotation


class TypeClassificationModel(ABC):
    """
    Abstract class that will be inherited by classification-style models.

    Inheritors will need to also inherit Model for this class to work
    """
    def __init__(self, model):
        self.__model = model  # type: Model

    def class_id_to_class(self, class_id: int) -> str:
        name = self.__model.metadata['annotation_vocab'].get_name_for_id(class_id)
        if self.__model.metadata['annotation_vocab'].is_unk(name):
            return 'typing.Any'
        return name

    def _make_parameters(self, representation_size: int):
        type_vocabulary_size = len(self.__model.metadata['annotation_vocab'])
        self.__model.parameters['cg_representation_to_annotations_out'] = \
            tf.get_variable(name='cg_representation_to_annotations_out',
                            initializer=tf.random_uniform_initializer(),
                            shape=[
                                representation_size, type_vocabulary_size],
                            )

        self.__model.parameters['annotation_bias'] = tf.get_variable(
            'annotation_class_bias',
            initializer=tf.zeros_initializer(),
            shape=(type_vocabulary_size,)
        )

    def _make_placeholders(self, is_train: bool) -> None:
        self.__model.placeholders['typed_annotation_target_class'] = \
            tf.placeholder(tf.int32, shape=(None,),
                           name="typed_annotation_target_class")

    def _make_model(self, target_representations, is_train: bool = True) -> None:
        target_logits = tf.matmul(
            target_representations, self.__model.parameters['cg_representation_to_annotations_out']) + \
                        self.__model.parameters['annotation_bias']

        # For full classification
        self.__model.ops['target_log_probs'] = tf.nn.log_softmax(target_logits)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.__model.placeholders['typed_annotation_target_class'],
            logits=target_logits
        )
        if 'loss' in self.__model.ops:
            self.__model.ops['loss'] += tf.reduce_mean(losses)
        else:
            self.__model.ops['loss'] = tf.reduce_mean(losses)

    @staticmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        raw_metadata['type_occurences_counter'] = Counter()

    @staticmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any],
                                   raw_metadata: Dict[str, Any]) -> None:
        annotations = (annotation_data['annotation'] for annotation_data in raw_sample['supernodes'].values()
                       if not ignore_type_annotation(annotation_data['annotation']))
        if 'strip_type_parameters' in hyperparameters and hyperparameters['strip_type_parameters']:
            annotations = (t.split("[")[0] for t in annotations)
        raw_metadata['type_occurences_counter'].update(annotations)

    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]], final_metadata: Dict[str, Any]):
        # Merge counters
        merged_type_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_type_counter.update(raw_metadata["type_occurences_counter"])

        final_metadata['annotation_vocab'] = Vocabulary.create_vocabulary(
            merged_type_counter,
            max_size=self.__model.hyperparameters['max_type_annotation_vocab_size'])
        return final_metadata

    @staticmethod
    def _get_idx_for_type(type_name: str, metadata: Dict[str, Any], hyperparameters: Dict[str, Any]):
        if 'strip_type_parameters' in hyperparameters and hyperparameters['strip_type_parameters'] and type_name is not None:
            type_name = type_name.split("[")[0]
        return metadata['annotation_vocab'].get_id_or_unk(type_name)


    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        batch_data['batch_target_variable_class'] = []

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]):
        batch_data['batch_target_variable_class'].extend(sample['variable_target_class'])

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool, minibatch) -> None:
        write_to_minibatch(
            minibatch, self.__model.placeholders['typed_annotation_target_class'], batch_data['batch_target_variable_class'])

    def annotate_single(self, raw_sample: Dict[str, Any], loaded_test_sample: Dict[str, Any], provenance: str):
        _, fetches = self.__model._run_epoch_in_batches(
            loaded_test_sample, '(test)', is_train=False, quiet=True,
            additional_fetch_dict={'target_log_probs': self.__model.ops['target_log_probs']}
        )
        target_log_probs = fetches['target_log_probs']

        # assumes that loading happens in the same order.
        original_annotations = []
        for node_idx, annotation_data in raw_sample['supernodes'].items():
            node_idx = int(node_idx)
            if 'ignored_supernodes' in loaded_test_sample and node_idx in loaded_test_sample['ignored_supernodes']:
                continue

            annotation = annotation_data['annotation']
            original_annotations.append((node_idx, annotation, annotation_data['name'], annotation_data['location'], annotation_data['type']))

        assert len(original_annotations) == target_log_probs.shape[0]

        # This is also classification-specific due to class_id_to_class
        for i, (node_idx, node_type, var_name, annotation_location, annotation_type) in enumerate(original_annotations):
            annotation = self.__model.Annotation(
                provenance=provenance,
                node_id=node_idx,
                name=var_name,
                original_annotation=node_type,
                annotation_type=annotation_type,
                predicted_annotation_logprob_dist={self.class_id_to_class(j): target_log_probs[i, j] for j in
                                                   range(target_log_probs.shape[1])},
                location=annotation_location
            )
            yield annotation
