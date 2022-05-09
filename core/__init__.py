import importlib
import torch.nn as nn


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called [model_name]Model() will
    be instantiated. It has to be a subclass of nn.Module,
    and it is case-insensitive.
    """
    model_filename = model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of nn.Module with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from core import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance