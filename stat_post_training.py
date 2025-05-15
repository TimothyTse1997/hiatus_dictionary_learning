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

import numpy as np
import torch

from dictionary_learning import AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder
from dictionary_learning.trainers import StandardTrainer, JumpReluTrainer, GatedSAETrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE

import webdataset as wds

class SparseEmbeddingStatistics:

    def __init__(self, data_path, threshold=0):
        self.all_embedding = self.load_data(data_path)
        self.threshold = threshold
        pass

    def load_data(self, data_path):
        all_embedding = []

        for line in tqdm(open(data_path, "r")):
            item = json.loads(line)
            embedding_key = "embedding" if "embedding" in item else "features"
            embed = item[embedding_key]
            all_embedding.append(embed)
        return np.array(all_embedding)
    
    def get_average_number_of_activation(self):
        activation_per_embed = np.sum(np.abs(self.all_embedding) > self.threshold) / self.all_embedding.shape[0]
        return {"activation_per_embed": str(activation_per_embed)}

    def get_num_activation_for_most_embed(self, target_fraction=0.5):

        total_activation = np.sum(np.abs(self.all_embedding) > self.threshold)
        target_size = int(total_activation * target_fraction)

        count_array = np.sum(np.abs(self.all_embedding) > self.threshold, axis=0)
        count_array = np.sort(count_array)[::-1]
        cumsum_count_array = np.cumsum(count_array)
        try:
            freq_activation_count = np.where(cumsum_count_array > target_size)[0][0]
        except Exception as e:
            print(e)
            freq_activation_count = 0

        return {f"freq_activation_count_{target_fraction}": str(freq_activation_count)}
    
    def get_num_dim_zero_activate(self):
        count_array = np.sum(np.abs(self.all_embedding) > self.threshold, axis=0)
        return {"num_dim_zero_activation": str(np.sum(count_array==0))}

    def get_non_zero_dim_activations_counts(self):
        count_array = np.sum(np.abs(self.all_embedding) > self.threshold, axis=0)
        non_zero_index = np.where(count_array > 0)[0]
        non_zero_count = count_array[non_zero_index]
        non_zero_count = non_zero_count.tolist()
        non_zero_count.sort(key=lambda x: -x)
        return {"count_activation": non_zero_count}
    
    def run(self):
        result = {}
        stat_fn = [
            self.get_average_number_of_activation,
            #self.get_num_activation_for_most_embed,
            self.get_num_dim_zero_activate
        ]
        for fn in stat_fn:
            result.update(fn())
        for i in [0.2, 0.5, 0.8, 0.9]:
            result.update(
                self.get_num_activation_for_most_embed(target_fraction=i)
            )
        result.update(self.get_non_zero_dim_activations_counts())

        return result

class ExplainedVariance:

    def __init__(
        self,
        recon_file,
        gold_file="/project/def-lingjzhu/tltse/official_data/english.preds.jsonl",
    ):
        self.recon_file = recon_file
        self.gold_file = gold_file
        self.recon_embs = self.load_recon_data(self.recon_file)
        self.gold_ids, self.gold_embs = self.load_gold_data(self.gold_file)
        assert(self.gold_embs.shape == self.recon_embs.shape)

    def load_recon_data(self, data_path):
        all_embedding = []

        for line in tqdm(open(data_path, "r")):
            item = json.loads(line)
            embedding_key = "embedding" if "embedding" in item else "features"
            embed = item[embedding_key]
            all_embedding.append(embed)
        return np.array(all_embedding)

    def load_gold_data(self, data_path):
        all_embedding = []
        all_id = []

        for line in tqdm(open(data_path, "r")):
            item = json.loads(line)
            embedding_key = "embedding" if "embedding" in item else "features"
            all_embedding.append(item[embedding_key])
            all_id.append(item["documentID"])
        assert(len(all_embedding) == len(all_id))
        return all_id, np.array(all_embedding)

    def matrix_cosine(self, x, y):
        return np.einsum('ij,ij->i', x, y) / (
              np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        )
    
    @torch.no_grad()
    def dict_learning_ev(self):
        act = torch.from_numpy(self.gold_embs)
        act_hat = torch.from_numpy(self.recon_embs)

        total_variance = torch.var(act, dim=0).sum()
        residual_variance = torch.var(act - act_hat, dim=0).sum()

        frac_variance_explained = 1 - residual_variance / total_variance
        return float(frac_variance_explained)

    def run_ev(self):
        distance = (self.gold_embs - self.recon_embs)
        #evs = 1 - (np.var(distance, axis=0).sum() / np.var(self.gold_embs, axis=0).sum())
        #ev_score = evs.mean()
        ev_score = self.dict_learning_ev()
        mse_score = (distance ** 2).mean()
        cosine_similarity = self.matrix_cosine(self.gold_embs, self.recon_embs).mean()

        return float(ev_score), float(mse_score), float(cosine_similarity)

def load_model_from_dir(model_path):
    parent_path = model_path.parent
    if not (parent_path / "config.json").exists():
        parent_path = model_path.parent.parent

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

    ae = model_type.from_pretrained(model_path)
    return ae


@torch.no_grad()
def eval_on_webdataset(
    data_path="/scratch/tltse/data/english_preds_eval/data-{00000..00607}.tar",
    ae=None,
):
    device = 'cuda'
    batch_size = 16384

    dataset = wds.DataPipeline(
        wds.SimpleShardList(data_path),
        wds.tarfile_to_samples(),
        wds.decode("pill"),
        wds.to_tuple("pth", "__key__", "txt"),
        wds.batched(batch_size))

    train_dataloader = wds.WebLoader(dataset, num_workers=1, batch_size=None)

    stat_result = {}

    all_explained_variance = []
    all_mse = []
    all_non_zero_count = 0
    total = 0

    # to prevent OOM, we calculate variance_explain per batch
    for step, (act, doc_ids, txts) in enumerate(tqdm(train_dataloader)):
        act = act.to(device)
        f_x = ae.encode(act)

        x_hat =  ae.decode(f_x).bfloat16().cpu()
        act = act.bfloat16().cpu()
        f_x = f_x.bfloat16().cpu()

        L0 = int(torch.where(f_x > 0)[0].size()[0])
        all_non_zero_count += L0
        total += act.shape[0]

        diff = (act - x_hat)
        mse = float((diff ** 2).mean().float())

        all_mse.append(mse)

        total_variance = torch.var(act, dim=0).sum()
        residual_variance = torch.var(diff, dim=0).sum()

        frac_variance_explained = 1 - residual_variance / total_variance
        frac_variance_explained = float(frac_variance_explained.float())
        all_explained_variance.append(frac_variance_explained)

    stat_result['ev_score_train_set'] = sum(all_explained_variance) / len(all_explained_variance)
    stat_result['mse_trainset'] = sum(all_mse) / len(all_mse)
    stat_result['L0_trainset'] = all_non_zero_count / total
    return stat_result


@torch.no_grad()
def main(
    save_dir = Path("/scratch/tltse/gated_vanilla_3mill_unpool_webdataset/"),
    data_path = "/scratch/tltse/data/english_preds_eval/data-{00000..00607}.tar",
    skip_intermediate=False,
    train_eval_datapath="/scratch/tltse/data/idiolect_embeddings/full/vectors_data/data-{00293..00297}.tar"
):
    device = 'cuda'
    batch_size = 16384
    if not skip_intermediate:
        all_file_paths = list(save_dir.glob("**/*.pt"))
    else:
        all_file_paths = list(save_dir.glob("**/ae.pt"))

    dataset = wds.DataPipeline(
        wds.SimpleShardList(data_path),
        wds.tarfile_to_samples(),
        wds.decode("pill"),
        wds.to_tuple("pth", "__key__", "txt"),
        wds.batched(batch_size))

    train_dataloader = wds.WebLoader(dataset, num_workers=1, batch_size=None)

    for p in all_file_paths:
        parent_path = p.parent
        print("start processing file: ", str(parent_path))

        if not "ae_" in p.name:
            assert(p.name == "ae.pt")
            recon_path = str((parent_path / "eval_embed.recon.jsonl"))
            sparse_path = str((parent_path / "eval_embed.sparse.jsonl"))
            checkpoint_step = None
        else:
            checkpoint_step = int(re.split(r"\.|_", p.name)[1])
            parent_path = parent_path.parent
            recon_path = str((parent_path / f"eval_embed_{checkpoint_step}.recon.jsonl"))
            sparse_path = str((parent_path / f"eval_embed_{checkpoint_step}.sparse.jsonl"))
            #assert(Path(recon_path).exists())
            #assert(Path(sparse_path).exists())

        ae = load_model_from_dir(p)
        _ = ae.eval().bfloat16()
        _ = ae.to(device)       

        print("start eval on train-domain result")
        train_eval_result = eval_on_webdataset(
            data_path=train_eval_datapath,
            ae=ae)

        if checkpoint_step is None:
            stat_output_file = str((parent_path / "sparse_stat.jsonl"))
            stat_output_file_bk = str((parent_path / "sparse_stat_bk.jsonl"))
        else:
            stat_output_file = str((parent_path / f"sparse_stat_{checkpoint_step}.jsonl"))
            stat_output_file_bk = str((parent_path / f"sparse_stat_{checkpoint_step}_bk.jsonl"))

        #assert(Path(stat_output_file).exists())

        if not Path(stat_output_file).exists():
            stat_obj = SparseEmbeddingStatistics(sparse_path)
            stat_result = stat_obj.run()
        else:
            stat_result = json.load(
                open(Path(stat_output_file), 'r')
            )
            #print(f"skip {str(parent_path)} base")
            #continue

        ev_obj = ExplainedVariance(recon_path)
        _, mse_score, cosine_similarity = ev_obj.run_ev()
        #stat_result['ev_score'] = ev_score
        stat_result['mse_score'] = mse_score
        stat_result['cosine_distance_numpy'] = 1-cosine_similarity

        del ev_obj

        #all_emb, all_recon_emb = [], []
        all_explained_variance = []
        # to prevent OOM, we calculate variance_explain per batch
        for step, (act, doc_ids, txts) in enumerate(tqdm(train_dataloader)):
            act = act.to(device)
            f_x = ae.encode(act)

            x_hat =  ae.decode(f_x).bfloat16().cpu()
            act = act.bfloat16().cpu()

            diff = (act - x_hat)
            mse = (diff ** 2).mean()

            total_variance = torch.var(act, dim=0).sum()
            residual_variance = torch.var(diff, dim=0).sum()

            frac_variance_explained = 1 - residual_variance / total_variance
            frac_variance_explained = float(frac_variance_explained.float())
            all_explained_variance.append(frac_variance_explained)

        stat_result['ev_score_eval_set_token'] = sum(all_explained_variance) / len(all_explained_variance)

        stat_result.update(train_eval_result)

        #with open(stat_output_file, 'w') as f:
        with open(stat_output_file, 'w') as f:
            print("STATE POST EVAL")
            print("SAVING STAT AT:", str(stat_output_file))
            print("model checkpoint:", str(p))
            assert('ev_score_eval_set_token' in stat_result.keys())
            assert('ev_score_train_set' in stat_result.keys())
            json.dump(stat_result, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train sae / generation."
    )
    parser.add_argument(
        "--checkpoint_dir",
        help="path to checkpoint",
        default=None
    )
    parser.add_argument(
        "--data_path",
        help="path to eval embed",
        default="/scratch/tltse/data/english_preds_eval/data-{00000..00607}.tar"
    )

    args = parser.parse_args()
    main(save_dir=Path(args.checkpoint_dir), data_path=args.data_path)
