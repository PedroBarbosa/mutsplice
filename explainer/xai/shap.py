import shap
import numpy as np
import tensorflow as tf

def deepshap():
    """
    Runs deepshap
    """
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x), compile=False) for x in paths]
    print(paths)
    #e = shap.DeepExplainer(model, background)