from typing import Dict, Any, Optional, Type

import tensorflow as tf
from dpu_utils.utils import RichPath

from .graph2annotation import Graph2Annotation
from .graph2hybridmetric import Graph2HybridMetric
from .graph2metric import Graph2Metric
from .model import Model
from .path2annotation import Path2Annotation
from .path2hybridmetric import Path2HybridMetric
from .path2metric import Path2Metric
from .sequence2annotation import Sequence2Annotation
from .sequence2hybridmetric import Sequence2HybridMetric
from .sequence2metric import Sequence2Metric


def get_model_class_from_name(model_name: str) -> Type[Model]:
    model_name = model_name.lower()
    if model_name in ['graph2annotation']:
        return Graph2Annotation
    elif model_name in ['graph2metric']:
        return Graph2Metric
    elif model_name in ['seq2annotation', 'sequence2annotation']:
        return Sequence2Annotation
    elif model_name in ['seq2metric', 'sequence2metric']:
        return Sequence2Metric
    elif model_name in ['graph2hybridmetric']:
        return Graph2HybridMetric
    elif model_name in ['seq2hybridmetric', 'sequence2hybridmetric']:
        return Sequence2HybridMetric
    elif model_name in {'path2annotation'}:
        return Path2Annotation
    elif model_name in {'path2metric'}:
        return Path2Metric
    elif model_name in {'path2hybridmetric'}:
        return Path2HybridMetric
    else:
        raise Exception("Unknown model '%s'!" % model_name)


def restore(path: RichPath, is_train: bool, hyper_overrides: Optional[Dict[str, Any]]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None) -> Model:
    saved_data = path.read_by_file_suffix()

    if hyper_overrides is not None:
        saved_data['hyperparameters'].update(hyper_overrides)

    model_class = get_model_class_from_name(saved_data['model_type'])
    model = model_class(hyperparameters=saved_data['hyperparameters'],
                        run_name=saved_data.get('run_name'),
                        model_save_dir=model_save_dir,
                        log_save_dir=log_save_dir)   # pytype: disable=not-instantiable
    model.metadata.update(saved_data['metadata'])
    model.make_model(is_train=is_train)

    variables_to_initialize = []
    with model.sess.graph.as_default():
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in sorted(model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), key=lambda v: v.name):
                used_vars.add(variable.name)
                if variable.name in saved_data['weights']:
                    # print('Initializing %s from saved value.' % variable.name)
                    restore_ops.append(variable.assign(saved_data['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in sorted(saved_data['weights']):
                if var_name not in used_vars:
                    if var_name.endswith('Adam:0') or var_name.endswith('Adam_1:0') or var_name in ['beta1_power:0', 'beta2_power:0']:
                        continue
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            model.sess.run(restore_ops)
    return model
