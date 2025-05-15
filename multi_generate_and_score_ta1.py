import os
import subprocess
import re
import random
import click
import json
import time
from tqdm import tqdm
from pathlib import Path
from typing import Any, Iterable
import argparse

from generate import run_single_checkpoint_generation
from stat_post_training import main as stat_main
from combine_statistics import main as combine_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train sae / generation."
    )
    parser.add_argument(
        "--dict_learning_model_dir",
        help="path to model dir",
        default=None
    )
    parser.add_argument(
        "--eval_dataset",
        help="path to eval embed",
    )
    parser.add_argument(
        "--eval_token_dataset",
        help="path to eval of token embed",
        default="/scratch/tltse/data/english_preds_eval/data-{00000..00607}.tar"
    )
    parser.add_argument(
        "--train_eval_datapath",
        help="path to eval of EV and MSE in train domain (can use train data to replace it if no split is done)",
        default="/scratch/tltse/data/idiolect_embeddings/full/vectors_data/data-{00293..00297}.tar"
    )
    parser.add_argument('--skip_intermediate', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    all_generated_file = []
    if not args.skip_intermediate:
        print("eval on intermediate steps")
        checkpoint_paths = list(Path(args.dict_learning_model_dir).glob("**/ae*.pt"))
    else:
        print("eval ONLY on full training")
        checkpoint_paths = list(Path(args.dict_learning_model_dir).glob("**/ae.pt"))
    
    for model_path in checkpoint_paths:
        parent_dir = model_path.parent
        if bool(re.search("ae\_[0-9]+\.pt", model_path.name)):
            parent_dir = model_path.parent.parent
            checkpoint_step = int(re.split("\_|\.", model_path.name)[1])

            sparse_output_file = parent_dir / f"eval_embed_{checkpoint_step}.sparse.jsonl"
            recon_output_file = parent_dir / f"eval_embed_{checkpoint_step}.recon.jsonl"

            sparse_score_ourput_dir = parent_dir/f"ta1_sparse_score_{checkpoint_step}_result" 
            recon_score_ourput_dir = parent_dir/f"ta1_recon_score_{checkpoint_step}_result" 
        else:
            sparse_output_file = parent_dir / "eval_embed.sparse.jsonl"
            recon_output_file = parent_dir / "eval_embed.recon.jsonl"

            sparse_score_ourput_dir = parent_dir/ "ta1_sparse_score_result" 
            recon_score_ourput_dir = parent_dir/ "ta1_recon_score_result" 
        if recon_output_file.exists() and len(list(recon_score_ourput_dir.glob("*"))) > 0:
            continue
        run_single_checkpoint_generation(
            str(model_path), 
            "/project/def-lingjzhu/tltse/official_data/english.preds.jsonl",
            [recon_output_file, sparse_output_file]
        )
        print("finish generation")
        print(sparse_output_file)
        print(recon_output_file)
        print(model_path)
        print(parent_dir)
        print("sparse scoring")
        all_generated_file += [sparse_output_file, recon_output_file]
        subprocess.run(
            [
                "/project/def-lingjzhu/tltse/ir-aa-master/scripts/score_ta1.sh",
                str(sparse_output_file),
                str(args.eval_dataset),
                "english",
                str(sparse_score_ourput_dir),
                "hrs2"
            ]
        )
        print("recon scoring")
        subprocess.run(
            [
                "/project/def-lingjzhu/tltse/ir-aa-master/scripts/score_ta1.sh",
                str(recon_output_file),
                str(args.eval_dataset),
                "english",
                str(recon_score_ourput_dir),
                "hrs2"
            ]
        )
    
    
    print("now run statistic for all data (explained varience)")
    stat_main(
        save_dir=Path(args.dict_learning_model_dir),
        data_path=args.eval_token_dataset,
        skip_intermediate=args.skip_intermediate,
        train_eval_datapath=args.train_eval_datapath
    )

    print("remove generated files to free disk space")
    for f in all_generated_file:
        if not f.exists(): continue
        os.remove(str(f))

    out_fname = Path(args.dict_learning_model_dir).parent / "full_train_stat.csv"
    combine_result(Path(args.dict_learning_model_dir).parent, str(out_fname))
