import logging
import re
from abc import abstractmethod
from collections import defaultdict
from itertools import product
from typing import Dict, Any, Optional, List

import numpy as np
import tensorflow as tf
from dpu_utils.mlutils import Vocabulary
from dpu_utils.tfutils import unsorted_segment_softmax

from .components.tokenembedder import TokenEmbedder
from .model import Model, write_to_minibatch
from .samplingiter import sampling_iter
from .utils import ignore_type_annotation

IDENTIFIER_REGEX = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")


class PathBasedModel(Model):

    LOGGER = logging.getLogger('PathBasedModel')

    @staticmethod
    @abstractmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = Model.get_default_hyperparameters()
        my_defaults = {
                        'max_num_paths_in_batch': 5000,
                        'max_num_paths_per_variable': 300,
                        'max_num_paths_per_file': 30000,

                        'leaf_label_embedding_style': 'Subtoken',  # One of ['Token', 'CharCNN', 'Subtoken']
                        'leaf_label_vocab_size': 10000,  # Used only when cg_node_label_embedding_style is 'Token' or 'Subtoken'
                        'leaf_label_char_length': 16,  # Used only when cg_node_label_embedding_style is 'CharCNN'
                        'leaf_label_max_subtokens': 5,  # Used only when cg_node_label_embedding_style is 'Subtoken'
                        'leaf_label_embedding_size': 64,

                        'max_path_size': 30,
                        'path_encoder_lstm_hidden_size': 128,
                        'path_encoding_size': 128,

                        'dropout_rate': 0.1

                     }
        defaults.update(my_defaults)
        return defaults

    def __init__(self, hyperparameters: Dict[str, Any], run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)

    @abstractmethod
    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)

        TokenEmbedder.make_placeholders('leaf_label', self.placeholders, hyperparameters=self.hyperparameters)

        self.placeholders['path_elements'] = tf.placeholder(dtype=tf.int64,
                                                    shape=[None, self.hyperparameters['max_path_size']],
                                                    name='path_elements')
        self.placeholders['path_lengths'] = tf.placeholder(dtype=tf.int32,
                                                    shape=[None],
                                                    name='path_lengths')

        self.placeholders['leaf_idxs'] = tf.placeholder(dtype=tf.int64,
                                                           shape=[None, 2],
                                                           name='leaf_idxs')

        self.placeholders['path_to_sample_idx'] = tf.placeholder(dtype=tf.int32,
                                                           shape=[None],
                                                           name='path_to_sample')

        self.placeholders['num_samples'] = tf.placeholder(dtype=tf.int32,
                                                          shape=[],
                                                          name='num_samples')

    def _make_parameters(self):
        super()._make_parameters()
        TokenEmbedder.make_parameters('leaf_label', self.parameters, self.metadata, self.hyperparameters)

        self.parameters['path_element_embeddings'] = tf.get_variable(name=f'path_element_embeddings',
                                shape=[len(self.metadata['non_terminal_dict']), self.hyperparameters['path_encoding_size']],
                                initializer=tf.random_uniform_initializer())

        self.parameters['birnn'] = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hyperparameters['path_encoder_lstm_hidden_size'], return_sequences=False),
            merge_mode='concat'
        )

        self.parameters['path_to_encoding_mx'] =  tf.get_variable(name='path_to_encoding_mx',
                                shape=[2*self.hyperparameters['path_encoder_lstm_hidden_size'] + 2*self.hyperparameters['leaf_label_embedding_size'],
                                       self.hyperparameters['path_encoding_size']],
                                initializer=tf.random_uniform_initializer())

        self.parameters['path_encoding_self_att'] = tf.get_variable(name='path_encoding_self_att',
                                                                 shape=[self.hyperparameters['path_encoding_size']],
                                                                 initializer=tf.random_uniform_initializer())

    @abstractmethod
    def _make_model(self, is_train: bool=True) -> None:
        super()._make_model(is_train)

        with tf.variable_scope("PathsModel"):
            leaf_embeddings = TokenEmbedder.make_model('leaf_label', self.placeholders, self.parameters,
                                                                  self.hyperparameters, is_train)
            leaf_embeddings = tf.nn.dropout(leaf_embeddings, rate=self.hyperparameters['dropout_rate'])  # [num-leafs, D]

            path_leaf_embeddings = tf.gather(leaf_embeddings, self.placeholders['leaf_idxs'])  # [num-paths, 2, D]
            path_leaf_embeddings = tf.reshape(path_leaf_embeddings, (tf.shape(path_leaf_embeddings)[0], -1))  # [num-paths, 2*D]

            path_element_embeddings = tf.nn.embedding_lookup(
                self.parameters['path_element_embeddings'],
                self.placeholders['path_elements']
            )   # [num-paths, max-path-size, D']


            length_mask = tf.reshape(tf.range(self.hyperparameters['max_path_size']), (1, -1)) \
                          < tf.expand_dims(self.placeholders['path_lengths'], axis=-1)  # [num-paths, max-path-size]
            path_embeddings =  self.parameters['birnn'](
                inputs=path_element_embeddings,
                mask=length_mask
            )  # [num-paths, H']



            full_path_embeddings = tf.einsum('bh,hd->bd',
                                             tf.concat([path_leaf_embeddings, path_embeddings], axis=-1),
                                             self.parameters['path_to_encoding_mx'])   # [num-paths, H]

            # Do a weighted sum of the paths
            path_scores = tf.einsum('bd,d->b', full_path_embeddings, self.parameters['path_encoding_self_att'])
            path_probs = unsorted_segment_softmax(path_scores,
                                                  segment_ids=self.placeholders['path_to_sample_idx'],
                                                  num_segments=self.placeholders['num_samples'])  # [num-paths]

            weighted_paths = tf.expand_dims(path_probs, axis=-1) * full_path_embeddings  # [num-paths, H]
            self.ops['target_variable_representations'] = tf.unsorted_segment_sum(
                weighted_paths,
                segment_ids=self.placeholders['path_to_sample_idx'],
                num_segments=self.placeholders['num_samples']
            )  # [num-samples, H]


    @staticmethod
    @abstractmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(PathBasedModel, PathBasedModel)._init_metadata(hyperparameters, raw_metadata)
        TokenEmbedder.init_metadata('leaf_label', raw_metadata, hyperparameters)
        raw_metadata['path_elements'] = set()

    @staticmethod
    @abstractmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(PathBasedModel, PathBasedModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)

        all_leaf_node_ids = set(raw_sample['token-sequence'])
        all_leaf_nodes = [raw_sample['nodes'][fid] for fid in all_leaf_node_ids
                          if IDENTIFIER_REGEX.match(raw_sample['nodes'][fid])]   # type: List[str]
        all_non_terminals = (raw_sample['nodes'][int(fid)] for fid in raw_sample['edges']['CHILD'].keys() if int(fid) not in all_leaf_node_ids)
        TokenEmbedder.load_metadata_from_sample('leaf_label', all_leaf_nodes, raw_metadata, hyperparameters)
        raw_metadata['path_elements'].update(all_non_terminals)

    @abstractmethod
    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super()._finalise_metadata(raw_metadata_list)

        TokenEmbedder.finalise_metadata('leaf_label', raw_metadata_list, final_metadata, self.hyperparameters)

        # First, merge all needed information:
        merged_non_terminals = set()
        for raw_metadata in raw_metadata_list:
            merged_non_terminals.update(raw_metadata["path_elements"])

        final_metadata['non_terminal_dict'] = Vocabulary.create_vocabulary(merged_non_terminals, max_size=10000,
                                                                           count_threshold=0)
        return final_metadata

    @staticmethod
    def __load_path_data_from_sample(hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                                     raw_sample: Dict[str, Any], result_holder: Dict[str, Any],
                                     is_train: bool=True) -> bool:
        leaf_node_ids = set(t for t in raw_sample['token-sequence'] if IDENTIFIER_REGEX.match(raw_sample['nodes'][t]))

        supernode_to_ground_nodes = defaultdict(set)   # int->set(int)
        for from_idx, to_idxs in raw_sample['edges']['OCCURRENCE_OF'].items():
            from_idx = int(from_idx)
            for to_idx in to_idxs:
                supernode_to_ground_nodes[to_idx].add(from_idx)

        # Pick a random subset of leaf nodes to supernodes for each variable
        paths_per_super_node = defaultdict(set)
        for supernode_idx, supernode_data in raw_sample['supernodes'].items():
            supernode_idx = int(supernode_idx)
            annotation = supernode_data['annotation']
            if is_train and ignore_type_annotation(annotation):
                continue
            all_path_leaves = ((f,t) for f, t in product(supernode_to_ground_nodes[supernode_idx], leaf_node_ids) if f<t)
            paths_per_super_node[supernode_idx] = set(sampling_iter(all_path_leaves, hyperparameters['max_num_paths_per_variable']))

        if len(paths_per_super_node) == 0:
            return False

        # Some files have too many paths/variables. Reduce the number of paths to something that a GPU can handle.
        # In practice this kicks in very rarely.
        total_num_paths = sum(len(p) for p in paths_per_super_node.values())
        if total_num_paths > hyperparameters['max_num_paths_per_file']:
            target_num_paths_per_sample = max(5, int(hyperparameters['max_num_paths_per_file'] / len(paths_per_super_node)))
            for s_id in paths_per_super_node:
                paths_per_super_node[s_id] = set(sampling_iter(paths_per_super_node[s_id], target_num_paths_per_sample))
            PathBasedModel.LOGGER.warning(f'Pruning file with {total_num_paths} paths. Now file has {sum(len(p) for p in paths_per_super_node.values())} paths.')

        if sum(len(t) for t in paths_per_super_node.values()) > hyperparameters['max_num_paths_per_file']:
            PathBasedModel.LOGGER.warning(f'Discarding example. Too large.')
            return False

        # now just get the concrete paths
        child_edges = {int(k): set(v) for k,v in raw_sample['edges']['CHILD'].items()}

        node_to_parent = {}  # type: Dict[int, int]
        for from_idx, to_idxs in child_edges.items():
            for to_idx in to_idxs:
                assert to_idx not in node_to_parent, 'A node cannot have multiple parents.'
                node_to_parent[to_idx] = from_idx

        def get_all_ancestors(node_idx):
            parents = set()
            while node_idx is not None:
                parents.add(node_idx)
                node_idx = node_to_parent.get(node_idx)
            return parents

        def get_path(node1_idx, node2_idx):
            node1_parents = get_all_ancestors(node1_idx)
            node2_parents = get_all_ancestors(node2_idx)
            common_ancestors = node1_parents & node2_parents
            if len(common_ancestors) == 0:
                return None  # Rare cases where a dummy node was constructed but is not part of the AST.

            path_node1_to_join_node = [node1_idx]
            join_node = node1_idx
            while join_node not in common_ancestors:
                join_node = node_to_parent[join_node]
                path_node1_to_join_node.append(join_node)

            path_node2_to_join_node = [node2_idx]
            join_node = node2_idx
            while join_node not in common_ancestors:
                join_node = node_to_parent[join_node]
                path_node2_to_join_node.append(join_node)

            assert path_node1_to_join_node[-1] == path_node2_to_join_node[-1]
            return path_node1_to_join_node + path_node2_to_join_node[:-1][::-1]

        token_to_id = {}
        tokens = []
        path_elements = []
        path_leaf_idxs = []
        path_to_sample_idx = []
        for sample_idx, (supernode_idx, paths_for_supernode) in enumerate(paths_per_super_node.items()):
            paths = (get_path(*p) for p in paths_for_supernode)
            for path in paths:
                if path is None or len(path) > hyperparameters['max_path_size']:
                    continue
                if path[0] in leaf_node_ids:
                    left_leaf = raw_sample['nodes'][path[0]]
                    path = path[1:]
                else:
                    left_leaf = 'Attribute'  # this happens for attributes

                if path[-1] in leaf_node_ids:
                    right_leaf = raw_sample['nodes'][path[-1]]
                    path = path[:-1]
                else:
                    right_leaf = 'Attribute'

                left_leaf_idx = token_to_id.get(left_leaf)
                if left_leaf_idx is None:
                    left_leaf_idx = len(token_to_id)
                    token_to_id[left_leaf] = left_leaf_idx
                    tokens.append(left_leaf)

                right_leaf_idx = token_to_id.get(right_leaf)
                if right_leaf_idx is None:
                    right_leaf_idx = len(token_to_id)
                    token_to_id[right_leaf] = right_leaf_idx
                    tokens.append(right_leaf)

                path_leaf_idxs.append([left_leaf_idx, right_leaf_idx])
                path_idxs = metadata['non_terminal_dict'].get_id_or_unk_multiple([raw_sample['nodes'][n] for n in path])
                path_elements.append(path_idxs)
                path_to_sample_idx.append(sample_idx)

        if len(path_elements) == 0:
            return False

        result_holder['path_elements'] = path_elements
        result_holder['path_leaf_idxs'] = np.array(path_leaf_idxs, dtype=np.uint16)
        result_holder['num_leaf_tokens'] = len(tokens)
        result_holder['path_sample_idx'] = np.array(path_to_sample_idx, dtype=np.uint16)
        result_holder['num_targets_in_sample'] = len(paths_per_super_node)
        TokenEmbedder.load_data_from_sample('leaf_label', metadata, tokens, result_holder, hyperparameters,
                                            is_train)
        return True

    @staticmethod
    @abstractmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool=True) -> bool:
        keep_sample = super(PathBasedModel, PathBasedModel)._load_data_from_sample(hyperparameters,
                                                                                   metadata,
                                                                                   raw_sample,
                                                                                   result_holder,
                                                                                   is_train)
        return keep_sample and PathBasedModel.__load_path_data_from_sample(hyperparameters,
                                                                           metadata,
                                                                           raw_sample=raw_sample,
                                                                           result_holder=result_holder,
                                                                           is_train=is_train)

    @abstractmethod
    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)

        batch_data['num_leaf_tokens'] = 0
        batch_data['num_samples'] = 0
        TokenEmbedder.init_minibatch('leaf_label', batch_data, self.hyperparameters)

        batch_data['path_leaf_idxs'] = []

        batch_data['path_elements'] = []
        batch_data['path_sample_idxs'] = []

    @abstractmethod
    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        super()._extend_minibatch_by_sample(batch_data, sample)
        TokenEmbedder.extend_minibatch_by_sample('leaf_label', batch_data, sample, self.hyperparameters)

        num_leaf_tokens_in_sample = sample['num_leaf_tokens']
        num_targets_in_sample = sample['num_targets_in_sample']

        batch_data['path_leaf_idxs'].append(sample['path_leaf_idxs'] + batch_data['num_leaf_tokens'])
        batch_data['path_elements'].extend(sample['path_elements'])
        batch_data['path_sample_idxs'].extend(sample['path_sample_idx'] + batch_data['num_samples'])

        batch_data['num_leaf_tokens'] += num_leaf_tokens_in_sample
        batch_data['num_samples'] += num_targets_in_sample
        return len(batch_data['path_elements']) >= self.hyperparameters['max_num_paths_in_batch']

    @abstractmethod
    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)
        TokenEmbedder.finalise_minibatch('leaf_label', batch_data, self.placeholders, minibatch, self.hyperparameters, is_train)

        path_elements = np.zeros((len(batch_data['path_elements']), self.hyperparameters['max_path_size']), dtype=np.int32)
        path_lengths = np.zeros(len(batch_data['path_elements']))

        for i, path in enumerate(batch_data['path_elements']):
            path_length = min(len(path), self.hyperparameters['max_path_size'])
            path_elements[i, :path_length] = path[:path_length]
            path_lengths[i] = path_length

        write_to_minibatch(minibatch, self.placeholders['path_elements'], path_elements)
        write_to_minibatch(minibatch, self.placeholders['path_lengths'], path_lengths)

        write_to_minibatch(minibatch, self.placeholders['leaf_idxs'], np.concatenate(batch_data['path_leaf_idxs'], axis=0))
        write_to_minibatch(minibatch, self.placeholders['path_to_sample_idx'], batch_data['path_sample_idxs'])
        write_to_minibatch(minibatch, self.placeholders['num_samples'], batch_data['num_samples'])
        if len(batch_data['path_elements']) > 30000: print(f"Num paths in mb {len(batch_data['path_elements'])}")
        return minibatch
