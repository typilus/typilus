import re
from collections import Counter
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary

from typilus.model.model import write_to_minibatch
from .component import Component

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1

class TokenEmbedder(Component):

    STRING_LITERAL_REGEX = re.compile('^[fub]?["\'](.*)["\']$')
    STRING_LITERAL =  '$StrLiteral$'
    INT_LITERAL = '$IntLiteral$'
    FLOAT_LITERAL = '$FloatLiteral$'

    @staticmethod
    def filter_literals(token: str) -> str:
        try:
            v = int(token)
            return TokenEmbedder.INT_LITERAL
        except ValueError:
            pass
        try:
            v = float(token)
            return TokenEmbedder.FLOAT_LITERAL
        except ValueError:
            pass
        string_lit = TokenEmbedder.STRING_LITERAL_REGEX.match(token)
        if string_lit:
            return TokenEmbedder.STRING_LITERAL
        return token

    @staticmethod
    def make_placeholders(name: str, placeholder_dict: Dict[str, tf.Tensor], hyperparameters: Dict[str, Any]):
        node_label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()

        if node_label_embedding_style == 'token':
            placeholder_dict[f'{name}_token_ids'] = \
                tf.placeholder(dtype=tf.int32, shape=[None], name=f'{name}_token_ids')

        elif node_label_embedding_style == 'subtoken':
            placeholder_dict[f'{name}_subtoken_ids'] = \
                tf.placeholder(dtype=tf.int32, shape=[None, hyperparameters[f'{name}_max_subtokens']],
                               name=f'{name}_subtoken_ids')
            placeholder_dict[f'{name}_num_subtokens'] = \
                tf.placeholder(dtype=tf.int32, shape=[None], name=f'{name}_num_subtokens')

        elif node_label_embedding_style == 'charcnn':
            placeholder_dict[f'{name}_unique_chars'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None, hyperparameters[f'{name}_char_length']],
                               name=f'{name}_unique_chars')
            placeholder_dict[f'{name}_unique_indices'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name=f'{name}_unique_indices')
        else:
            raise Exception(
                "Unknown node label embedding style '%s'!" % hyperparameters[f'{name}_embedding_style'])

    @staticmethod
    def make_parameters(name: str, parameters: Dict[str, tf.Tensor], metadata: Dict[str, Any], hyperparameters: Dict[str, Any]):
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()
        label_embedding_size = hyperparameters[f'{name}_embedding_size']

        if label_embedding_style == 'token':
            vocab_size = len(metadata[f'{name}_vocab'])
            parameters[f'{name}_embeddings'] = \
                tf.get_variable(name=f'{name}_embeddings',
                                shape=[vocab_size, label_embedding_size],
                                initializer=tf.random_uniform_initializer())
        elif label_embedding_style == 'subtoken':
            vocab_size = len(metadata[f'{name}_subtoken_vocab'])
            parameters[f'{name}_subtoken_embeddings'] = \
                tf.get_variable(name=f'{name}_subtoken_embeddings',
                                shape=[vocab_size, label_embedding_size],
                                initializer=tf.random_uniform_initializer())

    @staticmethod
    def __get_node_label_subtoken_embeddings(name: str, parameters: Dict[str, tf.Tensor], node_label_subtoken_ids: tf.Tensor,
                                             node_label_subtoken_length: tf.Tensor, hyperparameters: Dict[str, Any]) -> tf.Tensor:
        """
        :param node_label_subtoken_ids: Tensor of shape [V x max-num-subtokens] representing subtokens of each node (identified by ID into label vocab).
        :param node_label_subtoken_length: Tensor of shape [V] representing the number of subtokens.
        :return: Tensor of shape [V, D] representing embedded node label information about each node.
        """
        subtoken_embeddings = tf.nn.embedding_lookup(parameters[f'{name}_subtoken_embeddings'], node_label_subtoken_ids)  # V x max_num_subtokens x D
        mask = tf.cast(
            tf.reshape(tf.range(hyperparameters[f'{name}_max_subtokens']), (1, -1)) < tf.reshape(node_label_subtoken_length, (-1, 1)),
            subtoken_embeddings.dtype)
        subtoken_embeddings *= tf.expand_dims(mask, axis=-1)

        return tf.math.reduce_sum(subtoken_embeddings, axis=1) / tf.cast(tf.reshape(node_label_subtoken_length, (-1, 1)), subtoken_embeddings.dtype)

    @staticmethod
    def __get_node_label_charcnn_embeddings(name: str,
                                            unique_chars: tf.Tensor,
                                            node_label_unique_indices: tf.Tensor,
                                            hyperparameters: Dict[str, Any]) -> tf.Tensor:
        """
        :param unique_chars: Unique labels occurring in batch
                           Shape: [num unique labels, hyperparameters['node_label_char_length']], dtype=int32
        :param node_label_unique_indices: For each node in batch, index of corresponding (unique) node label in node_label_chars_unique.
                                          Shape: [V], dtype=int32
        :return: Tensor of shape [V, D] representing embedded node label information about each node.
        """
        label_embedding_size = hyperparameters[f'{name}_embedding_size']  # D
        # U ~ num unique labels
        # C ~ num characters (self.hyperparameters['node_label_char_length'])
        # A ~ num characters in alphabet
        unique_chars_one_hot = tf.one_hot(indices=unique_chars,
                                          depth=len(ALPHABET),
                                          axis=-1)  # Shape: [U, C, A]

        char_conv_l1_kernel_size = 4  # TODO: As Hyper. Last layer is essentially an MLP


        char_conv_l1 = tf.layers.conv1d(inputs=unique_chars_one_hot,
                                        filters=64,
                                        kernel_size=char_conv_l1_kernel_size,
                                        activation=tf.nn.leaky_relu)     # Shape: [U, C - (char_conv_l1_kernel_size - 1), 16]

        char_conv_l2_kernel_size = 5
        char_conv_l2 = tf.layers.conv1d(inputs=char_conv_l1,
                                        filters=label_embedding_size,
                                        kernel_size=char_conv_l2_kernel_size,
                                        activation=tf.nn.leaky_relu)     # Shape: [U, 1, D]

        unique_representations = tf.reduce_max(char_conv_l2, axis=1)  # Shape: [U, D]
        node_label_representations = tf.gather(params=unique_representations,
                                               indices=node_label_unique_indices)
        return node_label_representations

    @staticmethod
    def make_model(name: str, placeholder_dict: Dict[str, tf.Tensor], parameters: Dict[str, tf.Tensor],
                   hyperparameters: Dict[str, Any], is_train: bool = True):
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()

        if label_embedding_style == 'token':
            return tf.nn.embedding_lookup(parameters[f'{name}_embeddings'], placeholder_dict[f'{name}_token_ids'])

        elif label_embedding_style == 'subtoken':
            return TokenEmbedder.__get_node_label_subtoken_embeddings(
                name, parameters,
                placeholder_dict[f'{name}_subtoken_ids'],
                placeholder_dict[f'{name}_num_subtokens'],
                hyperparameters)
        elif label_embedding_style == 'charcnn':
            return TokenEmbedder.__get_node_label_charcnn_embeddings(
                name,
                placeholder_dict[f'{name}_unique_chars'],
                placeholder_dict[f'{name}_unique_indices'],
                hyperparameters)
        else:
            raise Exception("Unknown node label embedding style '%s'!" % label_embedding_style)


    @staticmethod
    def init_metadata(name: str, raw_metadata: Dict[str, Any], hyperparameters: Dict[str, Any]) -> None:
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()
        if label_embedding_style == 'token':
            raw_metadata[f'{name}_counter'] = Counter()
        elif label_embedding_style == 'subtoken':
            raw_metadata[f'{name}_subtoken_counter'] = Counter()

    @staticmethod
    def load_metadata_from_sample(name: str, raw_sample: List[str], raw_metadata: Dict[str, Any],
                                  hyperparameters: Dict[str, Any]) -> None:
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()

        if label_embedding_style == 'token':
            for label_token in raw_sample:
                raw_metadata[f'{name}_counter'][label_token] += 1
        elif label_embedding_style == 'subtoken':
            for label_token in raw_sample:
                filtered_token = TokenEmbedder.filter_literals(label_token)
                if filtered_token != label_token:
                    raw_metadata[f'{name}_subtoken_counter'][label_token] += 1  # Do not subtokenize
                else:
                    for subtoken in split_identifier_into_parts(label_token):
                        raw_metadata[f'{name}_subtoken_counter'][subtoken] += 1

    @staticmethod
    def finalise_metadata(name: str, raw_metadata_list: List[Dict[str, Any]],
                          final_metadata: Dict[str, Any], hyperparameters: Dict[str, Any]) -> None:
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()

        merged_node_label_counter = Counter()
        for raw_metadata in raw_metadata_list:
            if label_embedding_style == 'token':
                merged_node_label_counter += raw_metadata[f'{name}_counter']
            elif label_embedding_style == 'subtoken':
                merged_node_label_counter += raw_metadata[f'{name}_subtoken_counter']

        def add_special_literals(vocab: Vocabulary) -> None:
            vocab.add_or_get_id(TokenEmbedder.STRING_LITERAL)
            vocab.add_or_get_id(TokenEmbedder.FLOAT_LITERAL)
            vocab.add_or_get_id(TokenEmbedder.INT_LITERAL)

        if label_embedding_style == 'token':
            # Store token, type, and production vocabs:
            final_metadata[f'{name}_vocab'] = \
                Vocabulary.create_vocabulary(
                    merged_node_label_counter,
                    max_size=hyperparameters[f'{name}_vocab_size'])
            add_special_literals(final_metadata[f'{name}_vocab'])
        elif label_embedding_style == 'subtoken':
            final_metadata[f'{name}_subtoken_vocab'] = \
                Vocabulary.create_vocabulary(
                    merged_node_label_counter,
                    max_size=hyperparameters[f'{name}_vocab_size'])
            add_special_literals(final_metadata[f'{name}_subtoken_vocab'])

    @staticmethod
    def load_data_from_sample(name: str, metadata: Dict[str, Any], data: List[str],
                              result_holder: Dict[str, Any], hyperparameters: Dict[str, Any], is_train: bool = True) -> bool:
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()
        num_nodes = len(data)

        if label_embedding_style == 'token':
            # Translate node labels using the token vocabulary:
            node_labels = np.zeros((num_nodes,), dtype=np.uint16)
            for (node, label) in enumerate(data):
                if metadata[f'{name}_vocab'].is_unk(label):
                    label = TokenEmbedder.filter_literals(label)  # UNKs that are literals will be converted to special symbols.
                node_labels[node] = metadata[f'{name}_vocab'].get_id_or_unk(label)
            result_holder[f'{name}_token_ids'] = node_labels

        elif label_embedding_style == 'subtoken':
            max_num_subtokens = hyperparameters[f'{name}_max_subtokens']
            node_subtokens = np.zeros((num_nodes, max_num_subtokens), dtype=np.uint16)
            node_subtoken_length = np.zeros(num_nodes, dtype=np.uint8)
            for (node, label) in enumerate(data):
                filtered_label = TokenEmbedder.filter_literals(label)
                if filtered_label == label:
                    subtoken_ids = metadata[f'{name}_subtoken_vocab'].get_id_or_unk_multiple(split_identifier_into_parts(label))[:max_num_subtokens]
                elif metadata[f'{name}_subtoken_vocab'].is_unk(label):
                    subtoken_ids = metadata[f'{name}_subtoken_vocab'].get_id_or_unk_multiple([filtered_label])
                else:
                    subtoken_ids = metadata[f'{name}_subtoken_vocab'].get_id_or_unk_multiple([label])
                node_subtokens[node, :len(subtoken_ids)] = subtoken_ids
                node_subtoken_length[node] = len(subtoken_ids)
            result_holder[f'{name}_subtoken_ids'] = node_subtokens
            result_holder[f'{name}_subtoken_lengths'] = node_subtoken_length

        elif label_embedding_style == 'charcnn':
            # Translate node labels into character-based representation, and make unique per context graph:
            node_label_chars = np.zeros(shape=(num_nodes,
                                               hyperparameters[f'{name}_char_length']),
                                        dtype=np.uint8)
            for (node, label) in enumerate(data):
                for (char_idx, label_char) in enumerate(label[:hyperparameters[f'{name}_char_length']]):
                    node_label_chars[int(node), char_idx] = ALPHABET_DICT.get(label_char, 1)
            unique_chars, node_label_unique_indices = np.unique(node_label_chars,
                                                                      axis=0,
                                                                      return_inverse=True)
            result_holder[f'{name}_unique_chars'] = unique_chars
            result_holder[f'{name}_unique_indices'] = node_label_unique_indices
        else:
            raise Exception("Unknown node label embedding style '%s'!" % label_embedding_style)
        return True

    @staticmethod
    def init_minibatch(name: str, batch_data: Dict[str, Any], hyperparameters: Dict[str, Any]) -> None:
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()
        if label_embedding_style == 'token':
            batch_data[f'{name}_token_ids'] = []
        elif label_embedding_style == 'subtoken':
            batch_data[f'{name}_subtoken_ids'] = []
            batch_data[f'{name}_subtoken_lengths'] = []
        elif label_embedding_style == 'charcnn':
            batch_data[f'{name}_index_offset'] = 0
            batch_data[f'{name}_unique_chars'] = []
            batch_data[f'{name}_unique_indices'] = []
        else:
            raise Exception("Unknown node label embedding style '%s'!" % label_embedding_style)

    @staticmethod
    def extend_minibatch_by_sample(name: str, batch_data: Dict[str, Any], sample: Dict[str, Any],
                                   hyperparameters: Dict[str, Any]) -> bool:
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()
        if label_embedding_style == 'token':
            batch_data[f'{name}_token_ids'].extend(sample[f'{name}_token_ids'])

        elif label_embedding_style == 'subtoken':
            batch_data[f'{name}_subtoken_ids'].extend(sample[f'{name}_subtoken_ids'])
            batch_data[f'{name}_subtoken_lengths'].extend(sample[f'{name}_subtoken_lengths'])

        elif label_embedding_style == 'charcnn':
            # As we keep adding new "unique" labels, we need to shift the indices we are referring accordingly:
            batch_data[f'{name}_unique_chars'].extend(sample[f'{name}_unique_chars'])
            batch_data[f'{name}_unique_indices'].extend(
                sample[f'{name}_unique_indices'] + batch_data[f'{name}_index_offset'])
            batch_data[f'{name}_index_offset'] += len(sample[f'{name}_unique_chars'])
        else:
            raise Exception("Unknown node label embedding style '%s'!" % label_embedding_style)
        return True

    @staticmethod
    def finalise_minibatch(name: str, batch_data: Dict[str, Any], placeholders: Dict[str, tf.Tensor], minibatch: Dict[tf.Tensor, Any],
                           hyperparameters: Dict[str, Any], is_train: bool) -> None:
        label_embedding_style = hyperparameters[f'{name}_embedding_style'].lower()

        if label_embedding_style == 'token':
            write_to_minibatch(minibatch, placeholders[f'{name}_token_ids'], batch_data[f'{name}_token_ids'])

        elif label_embedding_style == 'subtoken':
            write_to_minibatch(minibatch, placeholders[f'{name}_subtoken_ids'],
                               batch_data[f'{name}_subtoken_ids'])
            write_to_minibatch(minibatch, placeholders[f'{name}_num_subtokens'],
                               batch_data[f'{name}_subtoken_lengths'])

        elif label_embedding_style == 'charcnn':
            write_to_minibatch(minibatch, placeholders[f'{name}_unique_chars'], batch_data[f'{name}_unique_chars'])
            write_to_minibatch(minibatch, placeholders[f'{name}_unique_indices'], batch_data[f'{name}_unique_indices'])
        else:
            raise Exception("Unknown node label embedding style '%s'!" % label_embedding_style)
