#!/usr/bin/env python
# coding: utf-8

import os, sys, yaml, argparse, re
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix

np.random.seed(22)

if "__sysstdout__" not in locals():
    __sysstdout__ = sys.stdout


def load_yaml(fname):
    yaml_dict = yaml.safe_load(open(fname))
    yaml_dict_list = []
    if "__dependency__" in yaml_dict:
        yaml_dict["__dependency__"] = ", ".join(
            [
                (
                    dep_fname.strip()
                    if os.path.isabs(dep_fname.strip())
                    else f"{os.getcwd()}/{dep_fname.strip()}"
                )
                for dep_fname in yaml_dict["__dependency__"].split(",")
            ]
        )
        for dep_fname in yaml_dict["__dependency__"].split(","):
            yaml_dict_list = yaml_dict_list + load_yaml(dep_fname.strip())
    yaml_dict_list.append(yaml_dict)
    return yaml_dict_list


def load_config_and_runtime_args(argv):
    try:
        config_sep_index = [x.startswith("-") for x in argv[1:]].index(True)
    except:
        config_sep_index = len(argv[1:])
    config_args = argv[1 : 1 + config_sep_index]
    runtime_args = argv[1 + config_sep_index :]

    parser = argparse.ArgumentParser()
    yaml_dict_lol = [load_yaml(fname) for fname in config_args]
    yaml_dict_list = [
        yaml_dict for yaml_dict_list in yaml_dict_lol for yaml_dict in yaml_dict_list
    ]
    config = {k: v for d in yaml_dict_list for k, v in d.items()}
    config = pd.json_normalize(config, sep="_").to_dict(orient="records")[0]
    for k, v in config.items():
        parser.add_argument(
            f"--{k}", default=v, type=str_to_bool if isinstance(v, bool) else type(v)
        )
    args = parser.parse_args(runtime_args)
    args.__dict__ = {
        k: (
            re.sub(r"\[(\w+)\]", lambda x: args.__dict__[x.group(0)[1:-1]], v)
            if isinstance(v, str)
            else v
        )
        for k, v in args.__dict__.items()
    }
    return args


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
