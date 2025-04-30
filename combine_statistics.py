import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

def read_metric_from_csv(fname):
    data = np.genfromtxt(fname, delimiter=',')
    return float(data[1, -1])

def get_stat_from_dir(target_dir):
    all_stat = {}
    for eval_stat_path in target_dir.glob("**/*.csv"):
        stat_mode = str(eval_stat_path.parents[0].name)
        score = read_metric_from_csv(eval_stat_path)
        all_stat[stat_mode] = score
    return all_stat

def check_and_round_string(t):
    try:
        t = float(t)
        return round(t, 3)
    except:
        return t


def combine_stat_in_directory(
    stat_file=None,
    recon_eval_result=None,
    sparse_eval_result=None,
):

    current_stat_file = json.load(open(stat_file, 'r'))
    #print(stat_file)
    #assert("ev_score" in current_stat_file)
    current_stat_file.pop('count_activation', None)

    sparse_score = 0
    sparse_total = 0
    for k, v in get_stat_from_dir(sparse_eval_result).items():
        #current_stat_file["sparse_" + k] = v
        sparse_score += float(v)
        sparse_total += 1

    recon_score = 0
    recon_total = 0
    for k, v in get_stat_from_dir(recon_eval_result).items():
        #current_stat_file["recon_" + k] = v
        recon_score += float(v)
        recon_total += 1

    current_stat_file["sparse_avg_score"] = sparse_score/sparse_total
    current_stat_file["recon_avg_score"] = recon_score/recon_total

    #print(current_stat_file.keys())

    current_stat_file = {k:check_and_round_string(v) for k, v in current_stat_file.items()}
    return current_stat_file

def get_stat_from_all_dir(train_checkpoint_dir):
    all_data = []
    for hyper_parameter_dir in train_checkpoint_dir.glob("*"):
        
        hp = str(hyper_parameter_dir.name)
        for model_dir in hyper_parameter_dir.glob("*"):
            model_name = model_dir.name
            for stat_file in model_dir.glob("sparse_stat*.jsonl"):

                data = {"hyper_param": hp}
                if bool(re.search("sparse\_stat\_[0-9]+\.jsonl", stat_file.name)):
                    checkpoint_step = re.split("\_|\.", stat_file.name)[2]
                    data["model_name"] = model_name + f"_{checkpoint_step}"
                    recon_eval_result = model_dir / f"ta1_recon_score_{checkpoint_step}_result"
                    sparse_eval_result = model_dir / f"ta1_sparse_score_{checkpoint_step}_result"
                else:
                    data["model_name"] = model_name
                    recon_eval_result=model_dir / "ta1_recon_score_result"
                    sparse_eval_result=model_dir / "ta1_sparse_score_result"

                print(stat_file)
                print(recon_eval_result)
                print(sparse_eval_result)

                data.update(
                    combine_stat_in_directory(
                        stat_file=stat_file,
                        recon_eval_result=recon_eval_result,
                        sparse_eval_result=sparse_eval_result
                    )
                )
                all_data.append(data)

    return all_data

def convert_result_to_csv(list_of_dict, fname):
    return pd.DataFrame(list_of_dict).to_csv(fname)

def main(train_checkpoint_dir, out_fname):
    stats = get_stat_from_all_dir(train_checkpoint_dir)
    #print(stats[0])
    assert("count_activation" not in stats[0])
    convert_result_to_csv(stats, out_fname)

if __name__ == "__main__":
    #train_checkpoint_dir = "/project/def-lingjzhu/tltse/matryoshka_test/"
    #out_fname = "/project/def-lingjzhu/tltse/matryoshka_test/full_train_stat.csv"

    #train_checkpoint_dir = "/scratch/tltse/matryoshka_test_idiolect_embeddings2/"
    #train_checkpoint_dir = "/scratch/tltse/jump_relu_3mill_test"
    #train_checkpoint_dir = "/scratch/tltse/jump_relu_3mill_unpool_webdataset"

    for train_checkpoint_dir in [
        #Path("/scratch/tltse/jump_relu_3mill_unpool_webdataset/"),
        #Path("/scratch/tltse/matryoshka_3mill_unpool_webdataset/"),
        #Path("/scratch/tltse/gated_vanilla_3mill_unpool_webdataset/")
        #Path("/scratch/tltse/extra_testing_3mill_unpool_webdataset/")
        Path("/scratch/tltse/v2_jumprelu_testing_3mill_unpool_webdataset/")
        #Path("/scratch/tltse/v2_matryoshka_testing_3mill_unpool_webdataset/")

    ]:
        out_fname = train_checkpoint_dir / "full_train_stat.csv"
        print(f"processing in {str(train_checkpoint_dir)}")
        print(f"generate file in {str(out_fname)}")
        main(train_checkpoint_dir, str(out_fname))
