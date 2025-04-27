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

@torch.no_grad()
def main_generation(
    model_path = "/scratch/st-jzhu71-1/ttse05/model_checkpoints/matryoshka_test/alpha_0.0001_topk_25/trainer_1/ae.pt",
    output_path = [
        "/scratch/st-jzhu71-1/ttse05/model_checkpoints/matryoshka_test/alpha_0.0001_topk_25/trainer_1/train_embed.recon.jsonl",
        "/scratch/st-jzhu71-1/ttse05/model_checkpoints/matryoshka_test/alpha_0.0001_topk_25/trainer_1/train_embed.sparse.jsonl"
    ],
    model_type=MatryoshkaBatchTopKSAE,
    dataset=None,
    output_dir = [
        "/scratch/st-jzhu71-1/ttse05/model_checkpoints/matryoshka_test/alpha_0.0001_topk_25/trainer_1/train_embed_recon/",
        "/scratch/st-jzhu71-1/ttse05/model_checkpoints/matryoshka_test/alpha_0.0001_topk_25/trainer_1/train_embed_sparse/",
    ],
    data_path = "/project/def-lingjzhu/tltse/official_data/english.preds.jsonl"
):
    device = "cuda"

    print("loading model")

    batch_size = 128
    ae = model_type.from_pretrained(model_path)
    _ = ae.eval()
    _ = ae.to(device)

    print("dataset")
    if dataset is None:
        mistral_dataset = MistralDataset(data_path)
    else:
        mistral_dataset = dataset
    
    mistral_dataloader = DataLoader(
        mistral_dataset, batch_size=batch_size, shuffle=False
    )
    if output_dir:
        f1_recon, f2_sparse = (
            open(output_path[0], 'w', encoding='utf8'), 
            open(output_path[1], 'w', encoding='utf8'), 
        )

    all_cosine_distances = []
    all_mse = []
    all_explained_variance = []

    # to prevent OOM, we calculate variance_explain per batch
    for batch in tqdm(mistral_dataloader):
        doc_ids = batch["documentID"]
        embeds = batch["features"].to(device)

        f_x =  ae.encode(embeds)
        x_hat =  ae.decode(f_x)

        cosine_distances = 1 - CosineSimilarity(dim=1)(embeds, x_hat)
        square_error = (embeds - x_hat) ** 2

        total_variance = torch.var(embeds, dim=0).sum()
        residual_variance = torch.var(embeds - x_hat, dim=0).sum()

        x_hat = x_hat.detach().cpu().numpy().tolist()
        f_x = f_x.detach().cpu().numpy().tolist()

        cosine_distances = cosine_distances.detach().cpu().numpy().tolist()
        square_error = float(square_error.mean().detach().cpu())

        all_cosine_distances += cosine_distances
        all_mse.append(square_error)

        frac_variance_explained = 1 - residual_variance / total_variance
        frac_variance_explained = float(frac_variance_explained.float())
        
        all_explained_variance.append(frac_variance_explained)
        if output_dir:
            for batch_index, doc_id in enumerate(doc_ids):
                sparse_embed = f_x[batch_index]
                recon_embed = x_hat[batch_index]
                
                print(
                    json.dumps({"documentID": doc_id, "features": recon_embed}, ensure_ascii=False),
                    file=f1_recon
                )
                print(
                    json.dumps({"documentID": doc_id, "features": sparse_embed}, ensure_ascii=False),
                    file=f2_sparse
                )

    if output_dir:
        f1_recon.close()
        f2_sparse.close()

    return all_cosine_distances, all_mse, all_explained_variance

def main_multi_generation(
    save_dir=Path("/scratch/tltse/extra2_testing_3mill_unpool_webdataset/"),
    data_path="/project/def-lingjzhu/tltse/official_data/english.preds.jsonl"
):
    
    all_checkpoint_paths = list(save_dir.glob("**/ae.pt"))
    mistral_dataset = MistralDataset(data_path)

    for p in all_checkpoint_paths:
        parent_path = p.parent
        assert (parent_path / "config.json").exists()

        with open(parent_path / "config.json", 'r') as f:
            model_config = json.load(f)["trainer"]
        if "jump" in model_config["lm_name"]:
            model_type = JumpReluAutoEncoder
        elif "gate" in model_config["lm_name"]:
            model_type = GatedAutoEncoder
        elif "vanilla" in model_config["lm_name"]:
            model_type = AutoEncoder
        elif "matryoshka" in model_config["lm_name"]:
            model_type = MatryoshkaBatchTopKSAE
        else:
            raise
        
        all_cosine_distances, all_mse, all_explained_variance = main_generation(
            model_path = p,
            output_path = [
                str((parent_path / "eval_embed.recon.jsonl")),
                str((parent_path / "eval_embed.sparse.jsonl")),
            ],
            model_type=model_type,
            dataset=mistral_dataset
        )
        print("completed generation in:")
        print(str((parent_path / "eval_embed.recon.jsonl")))
        print(str((parent_path / "eval_embed.sparse.jsonl")))

        print("running statistics in: ")
        stat_output_file = str((parent_path / "sparse_stat.jsonl"))
        print(stat_output_file)
        stat_obj = SparseEmbeddingStatistics(str((parent_path / "eval_embed.sparse.jsonl")))
        stat_result = stat_obj.run()
        stat_result["cosine_distance"] = sum(all_cosine_distances) / len(all_cosine_distances)
        stat_result["mse"] = sum(all_mse) / len(all_mse)
        stat_result["explained_variance_eval_embed"] = sum(all_explained_variance) / len(all_cosine_distances)

        with open(stat_output_file, 'w') as f:
            json.dump(stat_result, f, indent=4)

        if not (parent_path / "checkpoints").exists(): continue
        for c in (parent_path / "checkpoints").glob("ae_*.pt"):
            checkpoint_step = re.split(r"\.|_", c.name)[1]
            all_cosine_distances, all_mse, all_explained_variance = main_generation(
                model_path = c,
                output_path = [
                    str((parent_path / f"eval_embed_{checkpoint_step}.recon.jsonl")),
                    str((parent_path / f"eval_embed_{checkpoint_step}.sparse.jsonl")),
                ],
                model_type=model_type,
                dataset=mistral_dataset
            )
            stat_output_file = str((parent_path / f"sparse_stat_{checkpoint_step}.jsonl"))
            stat_obj = SparseEmbeddingStatistics(str((parent_path / f"eval_embed_{checkpoint_step}.sparse.jsonl")))
            stat_result = stat_obj.run()
            stat_result["cosine_distance"] = sum(all_cosine_distances) / len(all_cosine_distances)
            stat_result["mse"] = sum(all_mse) / len(all_mse)
            stat_result["explained_variance_eval_embed"] = sum(all_explained_variance) / len(all_cosine_distances)

            with open(stat_output_file, 'w') as f:
                json.dump(stat_result, f, indent=4)

def run_single_checkpoint_generation(model_path, data_path, output_path):
    assert(len(output_path) == 2)

    mistral_dataset = MistralDataset(data_path)
    parent_path = Path(model_path).parent
    if not (parent_path / "config.json").exists():
        parent_path = Path(model_path).parent.parent

    assert (parent_path / "config.json").exists()

    with open(parent_path / "config.json", 'r') as f:
        model_config = json.load(f)["trainer"]
    if "jump" in model_config["lm_name"]:
        model_type = JumpReluAutoEncoder
    elif "gate" in model_config["lm_name"]:
        model_type = GatedAutoEncoder
    elif "vanilla" in model_config["lm_name"]:
        model_type = AutoEncoder
    elif "matryoshka" in model_config["lm_name"]:
        model_type = MatryoshkaBatchTopKSAE
    else:
        raise
    
    checkpoint_step = re.split(r"\.|_", Path(model_path).name)[1]
    print("generate files in: ", output_path)
    all_cosine_distances, all_mse, all_explained_variance = main_generation(
        model_path = model_path,
        output_path = output_path,
        model_type=model_type,
        dataset=mistral_dataset
    )
    assert(Path(output_path[0]).exists())
    assert(Path(output_path[1]).exists())

    stat_output_file = str((parent_path / f"sparse_stat_{checkpoint_step}.jsonl"))

    stat_obj = SparseEmbeddingStatistics(output_path[1])
    stat_result = stat_obj.run()
    stat_result["cosine_distance"] = sum(all_cosine_distances) / len(all_cosine_distances)
    stat_result["mse"] = sum(all_mse) / len(all_mse)
    stat_result["explained_variance_eval_embed"] = sum(all_explained_variance) / len(all_cosine_distances)

    with open(stat_output_file, 'w') as f:
        json.dump(stat_result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train sae / generation."
    )
    parser.add_argument(
        "--model_path",
        help="path to model checkpoint",
        default=None
    )
    parser.add_argument(
        "--data_path",
        help="path to eval data",
        default="/project/def-lingjzhu/tltse/official_data/english.preds.jsonl"
    )
    parser.add_argument(
        "--output_path",
        help="path to generated output files",
        nargs='+',
        default=[]
    )

    args = parser.parse_args()
    run_single_checkpoint_generation(args.model_path, args.data_path, args.output_path)