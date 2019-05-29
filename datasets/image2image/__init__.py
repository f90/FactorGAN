import importlib
from datasets.image2image.base_dataset import BaseDataset
import os

def get_aligned_dataset(opt, subset):
    opt.dataset_mode = "aligned"
    opt.dataroot = os.path.join("datasets", "image2image", opt.dataset)
    opt.phase = subset
    opt.direction = "AtoB"
    opt.no_flip = True
    dataset = create_dataset(opt)
    return dataset

def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "datasets.image2image." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)

    # Add fixed no. of channel information for A and B
    if opt.dataset == "edges2shoes":
        dataset.A_nc = 1
        dataset.B_nc = 3
    else:
        dataset.A_nc = 3
        dataset.B_nc = 3

    print("dataset [%s] was created" % (instance.name()))
    return instance