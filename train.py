def track_lines_slurm(text):
    with open("./out.txt", 'a') as f:
        f.write(text + "\n")

from utils import *

import os
import re
import random
import click
import json
import time
from tqdm import tqdm
from pathlib import Path
from typing import Any, Iterable
import argparse
from copy import deepcopy

import psutil

import numpy as np
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity

from torch.utils.data import DataLoader, TensorDataset, Dataset
#from datasets import load_dataset, Dataset

from dictionary_learning import AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder
from dictionary_learning.trainers import StandardTrainer, JumpReluTrainer, GatedSAETrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE

import webdataset as wds

def webdataset_main(
    configs=[],
    data_path="/scratch/tltse/data/idiolect_embeddings/full/vectors_data/data-{00000..00297}.tar",
    save_dir="/scratch/tltse/extra2_testing_3mill_unpool_webdataset/",
    save_model_dir_names=[],
    batch_size=None,
    multi_config_parallel_training=False,
    trainer_config={
        "save_steps": [2000, 4000, 6000],
        "log_steps": 600,
        "steps": 8000,
        "verbose": True
    }
):
    assert(len(save_model_dir_names) == len(configs))

    device = "cuda"
    if not Path(save_dir).exists():
        Path(save_dir).mkdir()

    with open(Path(save_dir) / "training_cfg.json", 'w') as f:
        json.dump(trainer_config, f, indent=4)

    print("load dataset")
    dataset = wds.DataPipeline(
        wds.ResampledShards(data_path),
        wds.shuffle(10000),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode("pill"),
        wds.to_tuple("pth"),
        wds.batched(batch_size))
    train_dataloader = wds.WebLoader(dataset, num_workers=6, batch_size=None)
    
    lm_name = "mistral" 
    layer = -1

    # train the sparse autoencoder (SAE)
    print("start training")
    print("multi_config_parallel_training:", multi_config_parallel_training)
    if multi_config_parallel_training:
        ae = trainSAE(
            data=train_dataloader,
            trainer_configs=configs,
            save_dir=save_dir + "webdataset/",
            save_model_dir_names=save_model_dir_names,
            **trainer_config
        )
        return None
    print("start seq training")
    for cfg, model_name in zip(configs, save_model_dir_names):
        ae = trainSAE(
            data=train_dataloader,
            trainer_configs=[cfg],
            save_dir=save_dir + "webdataset/",
            save_model_dir_names=[model_name],
            **trainer_config
        )
    return None

def create_save_dir_name_from_config(config):
    filter_config_infos = lambda x: "_".join([k + "_" + str(v) for k, v in x.items() if not isinstance(v, list)])
    if "matryoshka" in config["lm_name"]:
        return f"matryoshka_{filter_config_infos(config)}"
    elif "jump" in config["lm_name"]:
        return f"jumprelu_{filter_config_infos(config)}"
    elif "gate" in config["lm_name"]:
        return f"gate_{filter_config_infos(config)}"
    elif "vanilla" in config["lm_name"]:
        return f"vanilla_{filter_config_infos(config)}"

def load_config(
    config_fname,
    save_dir,
    dict_size_magifier=[],
    topk=[],
    alpha=[],
    steps=None,
    skip_base=False
):
    with open(config_fname, 'r') as f:
        configs = json.load(f)
    print("loaded config from file")

    if not Path(save_dir).exists():
        Path(save_dir).mkdir()

    with open(Path(save_dir) / "model_cfgs.json", 'w') as f:
        full_model_cfg = {}
        full_model_cfg["model_cfg"] = configs
        full_model_cfg["hyper_parameter"] = {
            "save_dir": save_dir,
            "dict_size_magifier": dict_size_magifier,
            "topk": topk,
            "alpha": alpha,
            "steps": steps
        }
        json.dump(full_model_cfg, f, indent=4)

    processed_configs = []
    save_model_dir_names = []

    for config in configs:
        config['steps'] = steps

        for n in [None] + dict_size_magifier:
            if n != None:
                config["dict_size"] = config["activation_dim"] * int(n)

            if "matryoshka" in config["lm_name"]:
                config["trainer"] = MatryoshkaBatchTopKTrainer
                config["dict_class"] = MatryoshkaBatchTopKSAE
            elif "jump" in config["lm_name"]:
                config["trainer"] = JumpReluTrainer
                config["dict_class"] = JumpReluAutoEncoder
            elif "gate" in config["lm_name"]:
                config["trainer"] = GatedSAETrainer
                config["dict_class"] = GatedAutoEncoder
            elif "vanilla" in config["lm_name"]:
                config["trainer"] = StandardTrainer
                config["dict_class"] = AutoEncoder
            else:
                raise

            save_model_dir_names.append(f"base_{config['lm_name']}")
            processed_configs.append(config)

    print("complete basic")
    print(str(len(processed_configs)))
    hyperparameter_search_configs = deepcopy(processed_configs)

    for config in processed_configs:
        if "matryoshka" in config["lm_name"]:
            for k in topk:
                _config = deepcopy(config)
                _config["k"] = int(k)

                hyperparameter_search_configs.append(_config)
                save_model_dir_names.append(f"topk_{k}_{config['lm_name']}")
                #save_model_dir_names.append(create_save_dir_name_from_config(_config))

        else:
            for a in alpha:
                _config = deepcopy(config)
                if "sparsity_penalty" in _config.keys():
                    _config["sparsity_penalty"] = float(a)
                elif "l1_penalty" in _config.keys():
                    _config["l1_penalty"] = float(a)
                else:
                    raise

                hyperparameter_search_configs.append(_config)
                save_model_dir_names.append(f"alpha_{a}_{config['lm_name']}")
                #save_model_dir_names.append(create_save_dir_name_from_config(_config))
    
    if skip_base:
        print("number of configs before filtering: ", len(hyperparameter_search_configs))
        hyperparameter_search_configs, save_model_dir_names = zip(*[(c, n) for c, n in zip(hyperparameter_search_configs, save_model_dir_names) if ("base" not in n)])
        print("number of configs after filtering: ", len(hyperparameter_search_configs))
        hyperparameter_search_configs, save_model_dir_names = list(hyperparameter_search_configs),list(save_model_dir_names)
    assert(len(hyperparameter_search_configs) == len(save_model_dir_names))
    return hyperparameter_search_configs, save_model_dir_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train sae / generation."
    )
    parser.add_argument(
        "--data_path",
        help="path to data",
        default=None
    )
    parser.add_argument(
        "--save_dir",
        help="path to save",
        default=None
    )
    parser.add_argument(
        "--model_config_fname",
        help="path to config",
        default=None
    )
    parser.add_argument(
        "--training_config_fname",
        help="path to config",
        default=None
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048
    )
    parser.add_argument('--multi_config_parallel_training', action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_base', action=argparse.BooleanOptionalAction)

    parser.add_argument('--topk', nargs='+', default=[])
    parser.add_argument('--alpha',nargs='+', default=[])
    parser.add_argument('--dict_size_magifier', nargs='+', default=[])

    args = parser.parse_args()
    for k, value in args._get_kwargs():
        if value is not None:
            print(f"{k}: {value}")

    if args.training_config_fname:
        trainer_config = json.load(open(args.training_config_fname, 'r'))
    else:
        trainer_config = {
            "save_steps": [2000, 4000, 6000],
            "log_steps": 600,
            "steps": 8000,
            "verbose": True
        }
    print("load configs") 
    #assert(args.skip_base)
    hyperparameter_search_configs, save_model_dir_names = load_config(
        args.model_config_fname,
        args.save_dir,
        dict_size_magifier=args.dict_size_magifier,
        topk=args.topk,
        alpha=args.alpha,
        steps=trainer_config["steps"],
        skip_base=args.skip_base)

    print("config:") 
    print(hyperparameter_search_configs)
    print("enter webdataset_main")
    assert(not args.multi_config_parallel_training)
    webdataset_main(
        configs=hyperparameter_search_configs,
        data_path=args.data_path,
        save_dir=args.save_dir,
        save_model_dir_names=save_model_dir_names,
        batch_size=int(args.batch_size),
        multi_config_parallel_training=args.multi_config_parallel_training,
        trainer_config=trainer_config
    )
