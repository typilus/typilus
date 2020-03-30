import logging
import tempfile
from collections import defaultdict
from typing import Dict, Any, List

import annoy
import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from typilus.model.model import write_to_minibatch, Model
from typilus.model.utils import ignore_type_annotation


class TypeMetricLearningModel:
    """
    Abstract class that will be inherited by metric learning-style models.

    Inheritors will need to also inherit Model for this class to work
    """
    def __init__(self, model, type_representation_size: int, margin: float):
        self.__model = model  # type: Model
        self.__type_representation_size = type_representation_size
        assert margin >= 0
        self.__margin = margin

    def _make_placeholders(self, is_train: bool) -> None:
        self.__model.placeholders['typed_annotation_pairs_are_equal'] = \
            tf.placeholder(tf.int32, shape=(None, None), name="typed_annotation_pairs_are_equal")

    def _make_model(self, target_representations, is_train: bool = True) -> None:
        # target_representations is N x D
        self.__model.ops['target_representations'] = target_representations

        # Compute all pairs L2 distances
        target_representations_1 = tf.expand_dims(target_representations, axis=0)  # 1 x N x D
        target_representations_2 = tf.expand_dims(target_representations, axis=1)  # N x 1 x D
        distances = tf.norm(target_representations_1 - target_representations_2, axis=-1, ord=1)  # N x N

        typed_annotation_pairs_are_equal = tf.cast(self.__model.placeholders['typed_annotation_pairs_are_equal'], tf.float32)

        max_positive_distance = tf.reduce_max(distances * typed_annotation_pairs_are_equal, axis=-1)  # N
        neg_dist_filter = distances <= tf.expand_dims(max_positive_distance + self.__margin, axis=-1)
        pos_mask = typed_annotation_pairs_are_equal + tf.eye(tf.shape(distances)[0])
        neg_dist_filter = tf.cast(neg_dist_filter, dtype=tf.float32) * (1 - pos_mask)
        mean_negative_distances = tf.reduce_sum(
            distances * neg_dist_filter, axis=-1) / (tf.reduce_sum(neg_dist_filter, axis=-1)+1e-10)  # N

        min_negative_distance = tf.reduce_min(distances + pos_mask*3000, axis=-1)
        pos_dist_filter = tf.cast(distances >= tf.expand_dims(min_negative_distance - self.__margin, axis=-1), dtype=tf.float32)
        pos_dist_filter *= typed_annotation_pairs_are_equal
        mean_positive_distances = tf.reduce_sum(
            distances * pos_dist_filter, axis=-1
        ) / (tf.reduce_sum(pos_dist_filter, axis=-1) + 1e-10)

        triplet_loss = 0.5 * tf.nn.relu(mean_positive_distances - min_negative_distance + self.__margin)
        triplet_loss += 0.5 * tf.nn.relu(max_positive_distance - mean_negative_distances + self.__margin)

        if 'loss' in self.__model.ops:
            self.__model.ops['loss'] += tf.reduce_mean(triplet_loss)
        else:
            self.__model.ops['loss'] = tf.reduce_mean(triplet_loss)

    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        batch_data['batch_target_variable_type'] = []  # type: List[str]

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]):
        ## Assumes that the containing object has a list of 'target_type'
        batch_data['batch_target_variable_type'].extend(sample['target_type'])

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool, minibatch) -> None:
        targets = batch_data['batch_target_variable_type']
        num_annotations = len(targets)
        types_are_equal = np.zeros((num_annotations, num_annotations), dtype=np.bool)
        for i in range(num_annotations):
            for j in range(i+1, num_annotations):
                if targets[i] == targets[j]:
                    types_are_equal[i, j] = True
                    types_are_equal[j, i] = True

        write_to_minibatch(minibatch, self.__model.placeholders['typed_annotation_pairs_are_equal'], types_are_equal)

    def create_index(self, data_paths: List[RichPath], metadata: Dict[str, Any]) -> None:
        def representation_iter():
            data_chunk_iterator = (r.read_by_file_suffix() for r in data_paths)
            with self.__model.sess.as_default():
                for raw_data_chunk in data_chunk_iterator:
                    for raw_sample in raw_data_chunk:
                        loaded_sample = {}
                        use_example = self.__model._load_data_from_sample(self.__model.hyperparameters,
                                                                          self.__model.metadata,
                                                                          raw_sample=raw_sample,
                                                                          result_holder=loaded_sample,
                                                                          is_train=False)
                        if not use_example:
                            continue

                        _, fetches = self.__model._run_epoch_in_batches(
                            loaded_sample, '(indexing)', is_train=False, quiet=True,
                            additional_fetch_dict={'target_representations': self.__model.ops['target_representations']}
                        )
                        target_representations = fetches['target_representations']

                        idx = 0
                        for node_idx, annotation_data in raw_sample['supernodes'].items():
                            node_idx = int(node_idx)
                            if 'ignored_supernodes' in loaded_sample and node_idx in loaded_sample['ignored_supernodes']:
                                continue

                            annotation = annotation_data['annotation']
                            if ignore_type_annotation(annotation):
                                idx += 1
                                continue

                            yield target_representations[idx], annotation
                            idx += 1

        index = annoy.AnnoyIndex(self.__type_representation_size, 'manhattan')
        indexed_element_types = []
        logging.info('Creating index...')
        for i, (representation, type) in enumerate(representation_iter()):
            index.add_item(i, representation)
            indexed_element_types.append(type)
        logging.info('Indexing...')
        index.build(20)
        logging.info('Index Created.')

        with tempfile.NamedTemporaryFile() as f:
            index.save(f.name)
            with open(f.name, 'rb') as fout:
                metadata['index'] = fout.read()
        metadata['indexed_element_types'] = indexed_element_types

    def annotate_single(self, raw_sample: Dict[str, Any], loaded_test_sample: Dict[str, Any], provenance: str,
                        metadata: Dict[str, Any]):
        _, fetches = self.__model._run_epoch_in_batches(
            loaded_test_sample, '(test)', is_train=False, quiet=True,
            additional_fetch_dict={'target_representations': self.__model.ops['target_representations']}
        )
        target_representations = fetches['target_representations']
        if target_representations.shape[0] > 10000:
            return

        if not isinstance(metadata['index'], annoy.AnnoyIndex):
            with tempfile.NamedTemporaryFile() as f:
                with open(f.name, 'wb') as fout:
                    fout.write(metadata['index'])
                metadata['index'] = annoy.AnnoyIndex(self.__type_representation_size, 'manhattan')
                metadata['index'].load(f.name)

        # assumes that loading happens in the same order.
        original_annotations = []
        for node_idx, annotation_data in raw_sample['supernodes'].items():
            node_idx = int(node_idx)
            if 'ignored_supernodes' in loaded_test_sample and node_idx in loaded_test_sample['ignored_supernodes']:
                continue

            annotation = annotation_data['annotation']

            original_annotations.append((node_idx, annotation, annotation_data['name'], annotation_data['location'], annotation_data['type']))

        assert len(original_annotations) == target_representations.shape[0]

        # This is also classification-specific due to class_id_to_class
        for i, (node_idx, node_type, var_name, annotation_location, annotation_type) in enumerate(original_annotations):
            representation = target_representations[i]
            nn_idx, distance =  metadata['index'].get_nns_by_vector(representation, n=10, include_distances=True)
            distances = 1 / (np.array(distance) + 1e-10) ** 2
            distances /= np.sum(distances)
            rel_types = defaultdict(int)
            for n, p in zip(nn_idx, distances):
                rel_types[metadata['indexed_element_types'][n]] += p
            annotation = self.__model.Annotation(
                provenance=provenance,
                node_id=node_idx,
                name=var_name,
                original_annotation=node_type,
                annotation_type=annotation_type,
                predicted_annotation_logprob_dist={t: np.log(v) for t, v in rel_types.items()},
                location=annotation_location
            )
            yield annotation
