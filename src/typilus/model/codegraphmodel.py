import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List

import numpy as np
import tensorflow as tf

from .components.tokenembedder import TokenEmbedder
from .components.sparsegnn import SparseGGNN
from .model import Model, write_to_minibatch


class CodeGraphModel(Model):

    LOGGER = logging.getLogger('CodeGraphModel')

    @staticmethod
    @abstractmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = Model.get_default_hyperparameters()
        my_defaults = {
                        'max_num_cg_nodes_in_batch': 80000,

                        # Context Program Graph things:
                        'excluded_cg_edge_types': [],

                        'cg_node_label_embedding_style': 'Subtoken',  # One of ['Token', 'CharCNN', 'Subtoken']
                        'cg_node_label_vocab_size': 10000,   # Used only when cg_node_label_embedding_style is 'Token' or 'Subtoken'
                        'cg_node_label_char_length': 16,  # Used only when cg_node_label_embedding_style is 'CharCNN'
                        'cg_node_label_max_subtokens': 5,  # Used only when cg_node_label_embedding_style is 'Subtoken'
                        "cg_node_label_embedding_size": 64,

                        "cg_ggnn_layer_timesteps": [7, 1],
                        # "cg_ggnn_layer_timesteps": [],  # name_based baseline
                        "cg_ggnn_residual_connections": {"1": [0]},
                        # "cg_ggnn_residual_connections": {},  # name_based baseline

                        "cg_ggnn_hidden_size": 64,
                        "cg_ggnn_use_edge_bias": False,
                        "cg_ggnn_use_edge_msg_avg_aggregation": False,
                        "cg_ggnn_use_propagation_attention": False,
                        "cg_ggnn_graph_rnn_activation": "tanh",
                        "cg_ggnn_graph_rnn_cell": "GRU",
                        "cg_ggnn_message_aggregation": "max",

                     }
        defaults.update(my_defaults)
        return defaults

    def __init__(self, hyperparameters: Dict[str, Any], run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)

    @abstractmethod
    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)

        TokenEmbedder.make_placeholders('cg_node_label', self.placeholders, hyperparameters=self.hyperparameters)

        # Placeholders for context graph:
        cg_edge_type_num = len(self.metadata['cg_edge_type_dict'])

        self.placeholders['cg_adjacency_lists'] = \
            [tf.placeholder(dtype=tf.int64, shape=[None, 2], name='cg_adjacency_lists_e%s' % e)
             for e in range(cg_edge_type_num)]

        if self.hyperparameters['cg_ggnn_use_edge_bias'] or self.hyperparameters['cg_ggnn_use_edge_msg_avg_aggregation']:
            self.placeholders['cg_num_incoming_edges_per_type'] = \
                tf.placeholder(dtype=tf.float32, shape=[None, cg_edge_type_num], name='cg_num_incoming_edges_per_type')
            self.placeholders['cg_num_outgoing_edges_per_type'] = \
                tf.placeholder(dtype=tf.float32, shape=[None, cg_edge_type_num], name='cg_num_outgoing_edges_per_type')

    def _make_parameters(self):
        super()._make_parameters()
        TokenEmbedder.make_parameters('cg_node_label', self.parameters, self.metadata, self.hyperparameters)

    @abstractmethod
    def _make_model(self, is_train: bool=True) -> None:
        super()._make_model(is_train)

        # ----- Compute representation of all nodes in context graph using a GGNN:
        with tf.variable_scope("CodeGraph"):
            # (1) Compute initial embeddings for the nodes in the graph:

            initial_cg_node_embeddings = TokenEmbedder.make_model('cg_node_label', self.placeholders, self.parameters,
                                                                  self.hyperparameters, is_train)
            initial_cg_node_embeddings = tf.nn.dropout(initial_cg_node_embeddings, rate=1-self.hyperparameters['dropout_keep_rate'])
            l1 = tf.keras.layers.Dense(
                units=self.hyperparameters['cg_ggnn_hidden_size'],
                use_bias=False
            )
            initial_node_states = l1(initial_cg_node_embeddings)
            initial_node_states = tf.nn.dropout(initial_node_states,
                                                rate=1-self.placeholders['dropout_keep_rate'])

            # (2) Create GGNN and pass things through it:
            ggnn_hypers = {name.replace("cg_ggnn_", "", 1): value
                           for (name, value) in self.hyperparameters.items()
                           if name.startswith("cg_ggnn_")}
            ggnn_hypers['n_edge_types'] = len(self.metadata['cg_edge_type_dict'])

            ggnn_hypers['add_backwards_edges'] = True
            ggnn = SparseGGNN(ggnn_hypers)

            # All node representations are now in this op
            self.ops['cg_node_representations'] = \
                ggnn.sparse_gnn_layer(self.placeholders['dropout_keep_rate'],
                                      initial_node_states,
                                      self.placeholders['cg_adjacency_lists'],
                                      self.placeholders.get('cg_num_incoming_edges_per_type'),
                                      self.placeholders.get('cg_num_outgoing_edges_per_type'),
                                      {})

    @staticmethod
    @abstractmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(CodeGraphModel, CodeGraphModel)._init_metadata(hyperparameters, raw_metadata)
        TokenEmbedder.init_metadata('cg_node_label', raw_metadata, hyperparameters)
        raw_metadata['cg_edge_types'] = set()

    @staticmethod
    @abstractmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(CodeGraphModel, CodeGraphModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)

        TokenEmbedder.load_metadata_from_sample('cg_node_label', raw_sample['nodes'], raw_metadata, hyperparameters)
        raw_metadata['cg_edge_types'].update((e for e in raw_sample['edges']))

    @abstractmethod
    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super()._finalise_metadata(raw_metadata_list)

        # First, merge all needed information:
        merged_edge_types = set()
        for raw_metadata in raw_metadata_list:
            merged_edge_types.update(raw_metadata["cg_edge_types"])

        # Store edges allowed in the context graph, and assign numerical IDs to them:
        all_used_cg_edges = list(merged_edge_types - set(self.hyperparameters['excluded_cg_edge_types']))

        final_metadata['cg_edge_type_dict'] = {e: i for i, e in enumerate(all_used_cg_edges)}
        TokenEmbedder.finalise_metadata('cg_node_label', raw_metadata_list, final_metadata, self.hyperparameters)

        return final_metadata

    @staticmethod
    def __load_codegraph_data_from_sample(hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                                          raw_sample: Dict[str, Any], result_holder: Dict[str, Any],
                                          is_train: bool=True) -> bool:
        graph_node_labels = raw_sample['nodes']
        num_nodes = len(graph_node_labels)

        if num_nodes >= hyperparameters['max_num_cg_nodes_in_batch']:
            CodeGraphModel.LOGGER.warning("Dropping example using %i nodes in context graph" % (num_nodes,))
            return False

        # Translate node label, either using the token vocab or into a character representation:
        TokenEmbedder.load_data_from_sample('cg_node_label', metadata, raw_sample['nodes'], result_holder, hyperparameters, is_train)
        result_holder['num_nodes'] = num_nodes

        assert 'edges' in raw_sample

        # Split edges according to edge_type and count their numbers:
        edges = [[] for _ in metadata['cg_edge_type_dict']]

        num_edge_types = len(metadata['cg_edge_type_dict'])
        num_incoming_edges_per_type = np.zeros((num_nodes, num_edge_types), dtype=np.uint16)
        num_outgoing_edges_per_type = np.zeros((num_nodes, num_edge_types), dtype=np.uint16)

        edges_per_type = {}
        for edge_type, edge_dict in raw_sample['edges'].items():
            edge_list = []
            for from_idx, to_idxs in edge_dict.items():
                from_idx = int(from_idx)
                for to_idx in to_idxs:
                    edge_list.append((from_idx, to_idx))
            edges_per_type[edge_type] = edge_list

        for (e_type, e_type_idx) in metadata['cg_edge_type_dict'].items():
            if e_type in edges_per_type and len(edges_per_type[e_type]) > 0:
                edges[e_type_idx] = np.array(edges_per_type[e_type], dtype=np.int32)
            else:
                edges[e_type_idx] = np.zeros((0, 2), dtype=np.int32)

            # TODO: This is needed only in some configurations of the GNN!
            num_incoming_edges_per_type[:, e_type_idx] = np.bincount(edges[e_type_idx][:, 1],
                                                                    minlength=num_nodes)
            num_outgoing_edges_per_type[:, e_type_idx] = np.bincount(edges[e_type_idx][:, 0],
                                                                    minlength=num_nodes)
        assert not all(len(e) == 0 for e in edges)
        result_holder['cg_edges'] = edges
        result_holder['cg_num_incoming_edges_per_type'] = num_incoming_edges_per_type
        result_holder['cg_num_outgoing_edges_per_type'] = num_outgoing_edges_per_type
        return True

    @staticmethod
    @abstractmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool=True) -> bool:
        keep_sample = super(CodeGraphModel, CodeGraphModel)._load_data_from_sample(hyperparameters,
                                                                                   metadata,
                                                                                   raw_sample,
                                                                                   result_holder,
                                                                                   is_train)
        return keep_sample and CodeGraphModel.__load_codegraph_data_from_sample(hyperparameters,
                                                                                metadata,
                                                                                raw_sample=raw_sample,
                                                                                result_holder=result_holder,
                                                                                is_train=is_train)

    @abstractmethod
    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)

        batch_data['cg_node_offset'] = 0

        TokenEmbedder.init_minibatch('cg_node_label', batch_data, self.hyperparameters)
        batch_data['cg_adjacency_lists'] = [[] for _ in self.metadata['cg_edge_type_dict']]

        batch_data['cg_num_incoming_edges_per_type'] = []
        batch_data['cg_num_outgoing_edges_per_type'] = []

    @abstractmethod
    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        super()._extend_minibatch_by_sample(batch_data, sample)

        TokenEmbedder.extend_minibatch_by_sample('cg_node_label', batch_data, sample, self.hyperparameters)

        batch_data['cg_num_incoming_edges_per_type'].extend(sample['cg_num_incoming_edges_per_type'])
        batch_data['cg_num_outgoing_edges_per_type'].extend(sample['cg_num_outgoing_edges_per_type'])
        for edge_type in self.metadata['cg_edge_type_dict'].values():
            batch_data['cg_adjacency_lists'][edge_type].extend(sample['cg_edges'][edge_type] + batch_data['cg_node_offset'])

        batch_data['cg_node_offset'] += sample['num_nodes']
        return batch_data['cg_node_offset'] >= self.hyperparameters['max_num_cg_nodes_in_batch']

    @abstractmethod
    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)

        TokenEmbedder.finalise_minibatch('cg_node_label', batch_data, self.placeholders, minibatch, self.hyperparameters, is_train)

        if self.hyperparameters['cg_ggnn_use_edge_bias'] or self.hyperparameters['cg_ggnn_use_edge_msg_avg_aggregation']:
            write_to_minibatch(minibatch, self.placeholders['cg_num_incoming_edges_per_type'], batch_data['cg_num_incoming_edges_per_type'])
            write_to_minibatch(minibatch, self.placeholders['cg_num_outgoing_edges_per_type'], batch_data['cg_num_outgoing_edges_per_type'])

        for edge_type_idx, adjacency_list in enumerate(batch_data['cg_adjacency_lists']):
            write_to_minibatch(minibatch, self.placeholders['cg_adjacency_lists'][edge_type_idx], adjacency_list)

        return minibatch
