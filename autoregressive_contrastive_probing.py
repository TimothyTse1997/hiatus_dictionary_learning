import re
import random
import click
import json
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .train import prepare_embedding_dataset, MistralDataset

@torch.no_grad()
def main_unpool_generation(
    model_path = "/scratch/tltse/checkpoints/dictionary_learning/SAE/debug2/trainer_0/ae.pt",
    output_path = [
        "/scratch/tltse/checkpoints/dictionary_learning/SAE/debug2/eval_embed.source_contrastive.jsonl",
        "/scratch/tltse/checkpoints/dictionary_learning/SAE/debug2/eval_embed.recon_contrastive.jsonl",
        "/scratch/tltse/checkpoints/dictionary_learning/SAE/debug2/eval_embed.sparse_contrastive.jsonl",
    ],
    model_type=AutoEncoder
):
    device = "cuda"

    print("loading model")

    batch_size = 128
    print(model_path)
    print(model_type)
    ae = model_type.from_pretrained(model_path)
    _ = ae.eval()
    _ = ae.to(device)

    print("dataset")
    data_path = "/project/def-lingjzhu/tltse/official_data/english.preds.jsonl"
    mistral_dataset = MistralDataset(data_path)
    mistral_dataloader = DataLoader(
        mistral_dataset, batch_size=1, shuffle=False
    )
    #with open(output_path, 'w', encoding='utf8') as f:
    ( 
    f1_source_contrastive,
    f2_recon_contrastive,
    f3_sparse_contrastive) = [
        open(path, 'w', encoding='utf8') for path in output_path
    ]

    for batch in tqdm(mistral_dataloader):
        doc_ids = batch["documentID"]
        embeds = batch["features"].to(device)
        print(embeds.shape)
        source_contrastive = contrastive_from_embed(embeds[0])

        f_x =  ae.encode(embeds[0])
        x_hat =  ae.decode(f_x)

        recon_contrastive = contrastive_from_embed(x_hat)
        sparse_contrastive = contrastive_from_embed(f_x)


        source_contrastive = source_contrastive.detach().cpu().numpy().tolist()
        sparse_contrastive = sparse_contrastive.detach().cpu().numpy().tolist()
        recon_contrastive = recon_contrastive.detach().cpu().numpy().tolist()

        doc_id = doc_ids[0]
        print(
            json.dumps({"documentID": doc_id, "features": source_contrastive}, ensure_ascii=False),
            file=f1_source_contrastive
        )
        print(
            json.dumps({"documentID": doc_id, "features": sparse_contrastive}, ensure_ascii=False),
            file=f2_recon_contrastive
        )
        print(
            json.dumps({"documentID": doc_id, "features": recon_contrastive}, ensure_ascii=False),
            file=f3_sparse_contrastive
        )
    f1_source_contrastive.close()
    f2_recon_contrastive.close()
    f3_sparse_contrastive.close()

def contrastive_from_embed(embed):
    embed_cumsum = torch.cumsum(embed_cumsum, dim=0)/(torch.arange(0, embed_cumsum.shape[0])+1).unsqueeze(-1)
    gt_embed = embed_cumsum[-1]
    return torch.dot(embed_cumsum, gt_embed)

if __name__ == "__main__":
    main_unpool_generation()
    pass
