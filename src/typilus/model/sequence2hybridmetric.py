from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from typilus.model.components.multiheadattention import transformer_model
from typilus.model.components.tokenembedder import TokenEmbedder
from typilus.model.typeclassificationmodel import TypeClassificationModel
from typilus.model.typemetriclearningmodel import TypeMetricLearningModel
from typilus.model.utils import ignore_type_annotation
from .model import Model
from .model import write_to_minibatch


class Sequence2HybridMetric(Model):
    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = Model.get_default_hyperparameters()
        defaults.update({

            'token_embedding_style': 'Subtoken',  # One of ['Token', 'CharCNN', 'Subtoken']
            'token_vocab_size': 10000,   # Used only when cg_node_label_embedding_style is 'Token' or 'Subtoken'
            'token_char_length': 16,  # Used only when cg_node_label_embedding_style is 'CharCNN'
            'token_max_subtokens': 5,  # Used only when cg_node_label_embedding_style is 'Subtoken'
            'token_embedding_size': 64,

            'seq_layer_type': 'birnn', # One of ['BiRNN', 'SelfAtt']
            'num_seq_layers': 2,
            'use_consistency_layer': True,
            'max_seq_len': 50000,

            # BiRNN-specific
            'birnn_hidden_size': 32,

            #Transformer-specific
            'tranformer_hidden_size': 64,
            'tranformer_num_attention_heads': 8,
            'tranformer_intermediate_size': 128,

            'batch_size': 32,
            'margin': 2,
            'max_type_annotation_vocab_size': 100,
            'strip_type_parameters': True

        })
        return defaults

    def __init__(self, hyperparameters, run_name: Optional[str] = None, model_save_dir: Optional[str] = None, log_save_dir: Optional[str] = None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)
        seq_layer_type = self.hyperparameters['seq_layer_type'].lower()
        if seq_layer_type == 'birnn':
            out_size = 2 * self.hyperparameters['birnn_hidden_size']
        elif seq_layer_type == 'selfatt':
            out_size = self.hyperparameters['tranformer_hidden_size']
        else:
            raise ValueError('Unrecognized type of Sequential Layer %s' % seq_layer_type)

        self.__type_metric = TypeMetricLearningModel(self,
                                                     type_representation_size=out_size,
                                                     margin=self.hyperparameters['margin'])
        self.__type_classification = TypeClassificationModel(self)

    def _make_parameters(self):
        super()._make_parameters()
        TokenEmbedder.make_parameters('token', self.parameters, self.metadata, self.hyperparameters)

        seq_layer_type = self.hyperparameters['seq_layer_type'].lower()
        if  seq_layer_type == 'birnn':
            self.parameters['seq_layers'] = [
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hyperparameters['birnn_hidden_size'], return_sequences=True))
                for _ in range(self.hyperparameters['num_seq_layers'])
            ]
            out_size = 2 * self.hyperparameters['birnn_hidden_size']
        elif seq_layer_type == 'selfatt':
            self.parameters['seq_layers'] =[
                lambda input: transformer_model(input,
                                                hidden_size=self.hyperparameters['tranformer_hidden_size'],
                                                num_hidden_layers=1,
                                                num_attention_heads=self.hyperparameters['tranformer_num_attention_heads'],
                                                intermediate_size=self.hyperparameters['tranformer_intermediate_size'])
                for _ in range(self.hyperparameters['num_seq_layers'])
            ]
            out_size = self.hyperparameters['tranformer_hidden_size']
        else:
            raise ValueError('Unrecognized type of Sequential Layer %s' % seq_layer_type)

        self.parameters['metric_to_classification_linear_layer'] = \
            tf.get_variable(name='metric_to_classification_linear_layer',
                            initializer=tf.random_normal_initializer(),
                            shape=[out_size, out_size],
                            )
        self.__type_classification._make_parameters(representation_size=out_size)

    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)
        TokenEmbedder.make_placeholders('token', self.placeholders, hyperparameters=self.hyperparameters)

        self.placeholders['embedding_gather_matrix'] = tf.placeholder(tf.int32, shape=(None, None),
                                                                      name='embedding_gather_matrix')

        self.placeholders['sequence_lengths'] = tf.placeholder(tf.int32,
                                                                       shape=(None,),
                                                                       name="sequence_lengths")

        self.placeholders['variable_bound_token_ids'] = tf.placeholder(tf.int32,
                                                                     shape=(None,),
                                                                     name="variable_bound_token_ids")
        self.placeholders['token_variable_ids'] = tf.placeholder(tf.int32,
                                                                       shape=(None,),
                                                                       name="token_variable_ids")
        self.placeholders['num_variables'] = tf.placeholder(tf.int32, shape=[], name="num_variables")

        self.__type_metric._make_placeholders(is_train)
        self.__type_classification._make_placeholders(is_train)

    def _make_model(self, is_train: bool = True):
        super()._make_model(is_train)

        initial_token_embeddings = TokenEmbedder.make_model('token', self.placeholders, self.parameters,
                                                            self.hyperparameters, is_train)  # T x D

        sequence_token_embeddings = tf.gather_nd(params=initial_token_embeddings,
                                                 indices=tf.expand_dims(self.placeholders['embedding_gather_matrix'], axis=-1)
                                                 )  # B x max-len x D

        def get_variable_embeddings(all_sequence_embeddings):
            flat_sequence_embeddings = tf.reshape(all_sequence_embeddings, (-1, all_sequence_embeddings.get_shape()[-1]))  # B*max-len x D
            target_token_embeddings = tf.gather(params=flat_sequence_embeddings,
                                                indices=self.placeholders['variable_bound_token_ids'])

            return tf.unsorted_segment_mean(
                data=target_token_embeddings,
                segment_ids=self.placeholders['token_variable_ids'],
                num_segments=self.placeholders['num_variables']  # TODO: Do not depend in any way on the classes.
            ) # num-variables x H


        # Multiple layers of BiRNN/Transformer and "consistency" layer.
        current_out = sequence_token_embeddings  # B x max-len x H
        for i, seq_layer in enumerate(self.parameters['seq_layers']):
            # Mask out-of-sequence-tokens
            mask = tf.cast(
                tf.reshape(tf.range(tf.shape(sequence_token_embeddings)[1]), [1, -1, 1]) < tf.reshape(self.placeholders['sequence_lengths'], [-1, 1, 1]),
                current_out.dtype
            )
            current_out *= mask

            with tf.variable_scope('seqlayer_%s' % i):
                current_out = seq_layer(current_out)

            if i < len(self.parameters['seq_layers']) - 1 and self.hyperparameters['use_consistency_layer']:
                # Apply "consistency" layer to all layers but the last
                variable_embeddings = get_variable_embeddings(current_out) # num-variables x H

                variable_embedding_per_token = tf.gather(params=variable_embeddings, indices=self.placeholders['token_variable_ids'])  # num-variable-tokens x H

                current_out_shape = tf.shape(current_out)
                updates = tf.scatter_nd(
                    indices=tf.expand_dims(self.placeholders['variable_bound_token_ids'], axis=-1),
                    updates=variable_embedding_per_token,
                    shape=[current_out_shape[0] * current_out_shape[1], tf.shape(current_out)[2]]
                )
                current_out += tf.reshape(updates, tf.shape(current_out))

        variable_embeddings = get_variable_embeddings(current_out)
        self.ops['variable_embeddings'] = variable_embeddings

        self.__type_metric._make_model(variable_embeddings, is_train)

        classification_representation = tf.matmul(variable_embeddings,
                                            self.parameters['metric_to_classification_linear_layer'])
        classification_representation = tf.nn.dropout(classification_representation,
                                                rate=1-self.placeholders['dropout_keep_rate'])
        self.__type_classification._make_model(classification_representation, is_train)


    @staticmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Sequence2HybridMetric, Sequence2HybridMetric)._init_metadata(
            hyperparameters, raw_metadata)
        TokenEmbedder.init_metadata('token', raw_metadata, hyperparameters)
        TypeClassificationModel._init_metadata(hyperparameters, raw_metadata)

    @staticmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Sequence2HybridMetric, Sequence2HybridMetric)._load_metadata_from_sample(
            hyperparameters, raw_sample, raw_metadata)
        TokenEmbedder.load_metadata_from_sample('token', [raw_sample['nodes'][i] for i in raw_sample['token-sequence'] if i < len(raw_sample['nodes'])], raw_metadata,
                                                hyperparameters)
        TypeClassificationModel._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)

    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super()._finalise_metadata(raw_metadata_list)
        TokenEmbedder.finalise_metadata('token', raw_metadata_list, final_metadata, self.hyperparameters)
        self.__type_classification._finalise_metadata(raw_metadata_list, final_metadata)
        return final_metadata

    @staticmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool = True) -> bool:
        keep_sample = super(Sequence2HybridMetric, Sequence2HybridMetric)._load_data_from_sample(
            hyperparameters, metadata, raw_sample, result_holder, is_train)
        if not keep_sample:
            return False

        token_node_idxs = set(raw_sample['token-sequence'])
        node_idx_to_supernode_idx = {}  #  type: Dict[int, int]
        for from_idx, to_idxs in raw_sample['edges']['OCCURRENCE_OF'].items():
            from_idx = int(from_idx)
            if from_idx not in token_node_idxs:
                # Some supernodes do not have an associated token. Such nodes are attributes
                if str(from_idx) in raw_sample['edges']['CHILD']:
                    right_token_idx = max(raw_sample['edges']['CHILD'][str(from_idx)])
                    assert right_token_idx in token_node_idxs
                    from_idx = right_token_idx
                else:
                    continue
            for to_idx in to_idxs:
                node_idx_to_supernode_idx[from_idx] = to_idx

        supernodes_with_related_nodes = set(node_idx_to_supernode_idx.values())

        variable_types = []  # type: List[str]
        variable_type_idxs = []  # type: List[int]
        ignored_supernodes = set()
        supernode_idxs_to_annotated_variable_idx = {}  # type: Dict[int, int]
        for node_idx, supernode_data in raw_sample['supernodes'].items():
            node_idx = int(node_idx)
            annotation = supernode_data['annotation']
            if ignore_type_annotation(annotation) and is_train:
                ignored_supernodes.add(node_idx)
                continue
            if node_idx not in supernodes_with_related_nodes:
                ignored_supernodes.add(node_idx)
                continue

            variable_idx = len(supernode_idxs_to_annotated_variable_idx)
            variable_types.append(annotation)
            variable_type_idxs.append(TypeClassificationModel._get_idx_for_type(annotation, metadata, hyperparameters))
            supernode_idxs_to_annotated_variable_idx[node_idx] = variable_idx

        if len(variable_types) == 0:
            return False

        token_idx, variable_idx = [], []
        def create_token_sequence():
            for i, node_idx in enumerate(raw_sample['token-sequence']):
                supernode_idx = node_idx_to_supernode_idx.get(node_idx)
                if supernode_idx is not None:
                    annotated_variable_idxs = supernode_idxs_to_annotated_variable_idx.get(supernode_idx)
                    if annotated_variable_idxs is not None:
                        token_idx.append(i)
                        variable_idx.append(annotated_variable_idxs)
                yield raw_sample['nodes'][node_idx]

        token_sequence = list(create_token_sequence())
        if len(token_sequence) > hyperparameters['max_seq_len']:
            return False

        # Did we see at least one token per variable?
        assert len(np.unique(variable_idx)) == len(variable_types)

        TokenEmbedder.load_data_from_sample('token', metadata,
                                            token_sequence,
                                            result_holder,
                                            hyperparameters, is_train)

        result_holder['sequence_length'] = len(token_sequence)
        result_holder['variable_token_idxs'] = np.array(token_idx, dtype=np.uint32)
        result_holder['variable_idxs'] = np.array(variable_idx, dtype=np.uint32)
        result_holder['target_type'] = variable_types
        result_holder['variable_target_class'] = np.array(variable_type_idxs, dtype=np.uint32)
        result_holder['ignored_supernodes'] = ignored_supernodes
        return keep_sample

    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)
        TokenEmbedder.init_minibatch('token', batch_data, self.hyperparameters)
        batch_data['batch_sequence_lengths'] = []
        batch_data['batch_variable_token_idxs'] = []
        batch_data['batch_variable_idxs'] = []
        self.__type_metric._init_minibatch(batch_data)
        self.__type_classification._init_minibatch(batch_data)

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        super()._extend_minibatch_by_sample(batch_data, sample)

        TokenEmbedder.extend_minibatch_by_sample('token', batch_data, sample, self.hyperparameters)
        batch_data['batch_sequence_lengths'].append(sample['sequence_length'])

        batch_data['batch_variable_token_idxs'].append(sample['variable_token_idxs'])
        batch_data['batch_variable_idxs'].extend(sample['variable_idxs'] + len(batch_data['batch_target_variable_type']))

        self.__type_metric._extend_minibatch_by_sample(batch_data, sample)
        self.__type_classification._extend_minibatch_by_sample(batch_data, sample)
        return len(batch_data['batch_sequence_lengths']) >= self.hyperparameters['batch_size']

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)
        TokenEmbedder.finalise_minibatch('token', batch_data, self.placeholders, minibatch,
                                         self.hyperparameters, is_train)

        max_sequence_len = max(batch_data['batch_sequence_lengths'])
        batch_size = len(batch_data['batch_sequence_lengths'])
        # TODO: Later, force split very long sequences

        # embedding_gather_matrix contains the idxs such that embeddings[embedding_gather_matrix] creates a B x max_sequence_len x D matrix
        embedding_gather_matrix = np.zeros((batch_size, max_sequence_len), dtype=np.int32)
        current_idx = 0
        for i, length in enumerate(batch_data['batch_sequence_lengths']):
            embedding_gather_matrix[i, :length] = np.arange(start=current_idx, stop=current_idx+length)
            current_idx += length
        write_to_minibatch(minibatch, self.placeholders['embedding_gather_matrix'], embedding_gather_matrix)


        total_variable_tokens = len(batch_data['batch_variable_idxs'])
        variable_token_gather_idxs = np.empty(total_variable_tokens, dtype=np.int32)
        gathered_tokens_so_far = 0
        for i, tidx in enumerate(batch_data['batch_variable_token_idxs']):
            num_variable_tokens = len(tidx)
            variable_token_gather_idxs[gathered_tokens_so_far:gathered_tokens_so_far+num_variable_tokens] = tidx + i * max_sequence_len
            gathered_tokens_so_far += num_variable_tokens

        write_to_minibatch(
            minibatch, self.placeholders['variable_bound_token_ids'], variable_token_gather_idxs)

        write_to_minibatch(
            minibatch, self.placeholders['sequence_lengths'], batch_data['batch_sequence_lengths'])
        write_to_minibatch(
            minibatch, self.placeholders['token_variable_ids'], batch_data['batch_variable_idxs'])
        self.__type_metric._finalise_minibatch(batch_data, is_train, minibatch)
        self.__type_classification._finalise_minibatch(batch_data, is_train, minibatch)
        minibatch[self.placeholders['num_variables']] = len(batch_data['batch_target_variable_type'])

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
