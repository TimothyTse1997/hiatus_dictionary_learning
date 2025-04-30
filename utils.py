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
        try:
            total_activation = np.sum(np.abs(self.all_embedding) > self.threshold)
            target_size = int(total_activation * target_fraction)

            count_array = np.sum(np.abs(self.all_embedding) > self.threshold, axis=0)
            count_array = np.sort(count_array)[::-1]
            cumsum_count_array = np.cumsum(count_array)
            freq_activation_count = np.where(cumsum_count_array > target_size)[0][0]
            return {f"freq_activation_count_{target_fraction}": str(freq_activation_count)}
        except Exception as e:
            print(e)
            return {f"freq_activation_count_{target_fraction}": str(0)}

    
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

def convert_to_csr_matrix(array):
    return sparse.csr_matrix(array)

def save_csr_matrix(csr_matrix, fname):
    sparse.save_npz(fname, csr_matrix, compressed=True)


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

class TensorUnpoolDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_path = list(self.data_dir.glob("*.pt"))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        current_path = self.data_path[index]
        #return {
        #    "documentID": str(current_path.name),
        #    "features": torch.load(current_path).mean(0)
        #}
        return torch.load(current_path).mean(0)

class ChunkDataset(Dataset):
    def __init__(self, chunk_dir, chunk_name, return_mean_pool=True, return_dict=False, seq_len=300):
        self.chunk_dir = Path(chunk_dir)
        self.doc_ids = self.load_doc_ids(self.chunk_dir / f"{chunk_name}.json")
        self.embeds_path = self.chunk_dir / f"{chunk_name}.pt"
        self.return_mean_pool = return_mean_pool
        self.return_dict = return_dict
        self.embeds = None
        self.seq_len = seq_len
    
    def construct_unpool_dataset(self):
        chunk_size, seq_len, embed_size = self.embeds.shape
        self.embeds = self.embeds.view(-1, embed_size)
        self.doc_ids = sum([
           [f"{doc_id}_{i}" for i in range(seq_len)] for doc_id in self.doc_ids
        ], [])
        assert(len(self.doc_ids) == self.embeds.shape[0])

    def load_doc_ids(self, fname):
        doc_json = json.load(open(str(fname), 'r'))
        doc_ids = doc_json["doc_ids"]
        return doc_ids

    def log_current_mem(self):
        mem = psutil.virtual_memory()
        print(
            f"available: {mem.available}"
        )
        print(
            f"used: {mem.used}"
        )

    def chunk_init(self):
        embeds = torch.load(self.embeds_path)
        if self.return_mean_pool:
            self.embeds = embeds.mean(1)
        else:
            self.embeds = embeds
            self.construct_unpool_dataset()

    def chunk_free_mem(self):
        self.log_current_mem()
        self.embeds = None
        self.log_current_mem()

    def __len__(self):
        if self.embeds is None:
            chunk_size = len(self.doc_ids)
            data_size = chunk_size if self.return_mean_pool else chunk_size * self.seq_len
            return data_size
        else:
            return self.embeds.shape[0]

    def __getitem__(self, index):
        num_data = self.__len__()
        index = min(index, num_data-1)
        if (not self.return_dict):
            return self.embeds[index]
        return {
            "documentID": self.doc_ids[index],
            "features": self.embeds[index]
        }


def mistral_collate_fn(batch):
    #print(batch)
    result = {"documentID": [b["documentID"] for b in batch]}
    result["features"] = torch.FloatTensor([b["features"] for b in batch])
    return result

def chunk_collate_fn(batch):
    result = {"documentID": [b["documentID"] for b in batch]}
    result["features"] = torch.stack([b["features"] for b in batch])
    return result

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
            embedding_key = "embedding" if "embedding" in item else "features"
            all_embedding.append(item[embedding_key])
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
