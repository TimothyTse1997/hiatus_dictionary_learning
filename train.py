import re
import random
import click
import json
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, Dataset

from dictionary_learning import AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder
from dictionary_learning.trainers import StandardTrainer, TrainerJumpRelu, GatedSAETrainer
from dictionary_learning.training import trainSAE


def prepare_embedding_dataset(data_path):
    all_embedding = []

    for line in tqdm(open(data_path, "r")):
        item = json.loads(line)
        embedding_key = "embedding" if "embedding" in item else "features"
        embed = item[embedding_key]
        all_embedding.append(embed)
        if len(all_embedding) % 5000 == 0: print(f"processed {len(all_embedding)} data")

    text_embedding = F.normalize(torch.tensor(all_embedding))
    embedding_dataset = TensorDataset(text_embedding)
    return embedding_dataset

class MistralDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_id, self.all_embedding = self.load_data(self.data_path)
        return
    
    def __len__(self):
        return len(self.all_id)

    def load_data(self, data_path):
        all_embedding = []
        all_id = []

        for line in tqdm(open(data_path, "r")):
            item = json.loads(line)
            all_embedding.append(item["features"])
            all_id.append(item["documentID"])
        assert(len(all_embedding) == len(all_id))
        return all_id, all_embedding

    def __getitem__(self, idx):
        embed = torch.tensor(self.all_embedding[idx])
        current_data = {
            "documentID": self.all_id[idx],
            "features": F.normalize(embed, dim=0),
        }
        return current_data

def main():
    device = "cuda"
    save_dir = "/scratch/tltse/checkpoints/dictionary_learning/SAE/multi_model_sparsity_test2/"
    batch_size = 128
    print("load dataset")
    #embedding_dataset = prepare_embedding_dataset("/project/def-lingjzhu/tltse/official_data/english.preds.jsonl")
    embedding_dataset = prepare_embedding_dataset("/project/def-lingjzhu/tltse/mistral_embedding.jsonl")
    train_dataloader = DataLoader(
        embedding_dataset, batch_size=batch_size, shuffle=True
    )

    lm_name = "mistral" 
    layer = -1
    sparsity_penalties = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    # train the sparse autoencoder (SAE)
    print("start training")
    for alpha in sparsity_penalties:
        print(f"start training with alpha: {alpha}")
        trainer_cfgs = [
            {
                "trainer": StandardTrainer,
                "dict_class": AutoEncoder,
                "activation_dim": 2560,
                "dict_size": 2560*10,
                "lr": 1e-3,
                "device": device,
                "l1_penalty": alpha,
                "lm_name": lm_name + "_vanilla",
                "layer": layer
                #"save_steps": 1000,
            },
            {
                "trainer": TrainerJumpRelu,
                "dict_class": JumpReluAutoEncoder,
                "activation_dim": 2560,
                "dict_size": 2560*10,
                "lr": 7e-5,
                "device": device,
                "sparsity_penalty": alpha,
                "lm_name": lm_name + "_jump",
                "layer": layer
                #"save_steps": 1000,
            },
            {
                "trainer": GatedSAETrainer,
                "dict_class": GatedAutoEncoder,
                "activation_dim": 2560,
                "dict_size": 2560*10,
                "lr": 5e-5,
                "device": device,
                "l1_penalty": alpha,
                "lm_name": lm_name + "_gated",
                "layer": layer
                #"save_steps": 1000,
            } 
        ]
        ae = trainSAE(
            data=train_dataloader,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
            trainer_configs=trainer_cfgs, #[trainer_cfg],
            save_steps=1000,
            save_dir=save_dir + f"alpha_{str(alpha)}/"
        )

@torch.no_grad()
def main_generation(
    model_path = "/scratch/tltse/checkpoints/dictionary_learning/SAE/debug2/trainer_0/ae.pt",
    output_path = [
        "/scratch/tltse/checkpoints/dictionary_learning/SAE/debug2/eval_embed.recon.jsonl",
        "/scratch/tltse/checkpoints/dictionary_learning/SAE/debug2/eval_embed.sparse.jsonl"
    ],
    model_type=AutoEncoder
):
    device = "cuda"

    print("loading model")

    batch_size = 128
    #print(model_path)
    #print(model_type)
    ae = model_type.from_pretrained(model_path)
    _ = ae.eval()
    _ = ae.to(device)

    print("dataset")
    data_path = "/project/def-lingjzhu/tltse/official_data/english.preds.jsonl"
    mistral_dataset = MistralDataset(data_path)
    mistral_dataloader = DataLoader(
        mistral_dataset, batch_size=batch_size, shuffle=False
    )
    #with open(output_path, 'w', encoding='utf8') as f:
    f1_recon, f2_sparse = (
        open(output_path[0], 'w', encoding='utf8'), 
        open(output_path[1], 'w', encoding='utf8'), 
    )

    for batch in tqdm(mistral_dataloader):
        doc_ids = batch["documentID"]
        embeds = batch["features"].to(device)
        f_x =  ae.encode(embeds)
        x_hat =  ae.decode(f_x)

        x_hat = x_hat.detach().cpu().numpy().tolist()
        f_x = f_x.detach().cpu().numpy().tolist()
        for doc_id, rep in zip(doc_ids, x_hat):
            print(
                json.dumps({"documentID": doc_id, "features": rep}, ensure_ascii=False),
                file=f1_recon
            )
        for doc_id, rep in zip(doc_ids, f_x):
            print(
                json.dumps({"documentID": doc_id, "features": rep}, ensure_ascii=False),
                file=f2_sparse
            )
    f1_recon.close()
    f2_sparse.close()

def main_multi_generation():
    save_dir = Path("/scratch/tltse/checkpoints/dictionary_learning/SAE/multi_model_sparsity_test2/")
    all_checkpoint_paths = list(save_dir.glob("**/ae.pt"))

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
        else:
            raise

        main_generation(
            model_path = p,
            output_path = [
                str((parent_path / "eval_embed.recon.jsonl")),
                str((parent_path / "eval_embed.sparse.jsonl")),
            ],
            model_type=model_type
        )
        print("completed generation in:")
        print(str((parent_path / "eval_embed.recon.jsonl")))
        print(str((parent_path / "eval_embed.sparse.jsonl")))


if __name__ == "__main__":
    #main()
    #main_generation()
    main_multi_generation()
    pass
