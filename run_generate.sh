#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load arrow
module load rust
ENVDIR=/scratch/tltse/python_virtual_env/hiatus2
source $ENVDIR/bin/activate
which python
# set cache directory
# Please set all your cache directory to scratch. If you don't, cache files will be placed in your /home
export HF_DATASETS_CACHE=/scratch/tltse/cache/huggingface
export HF_HOME=/scratch/tltse/cache/huggingface

export PYTHONPATH=/project/def-lingjzhu/tltse/ir-aa-master:$PYTHONPATH
export PYTHONPATH=/project/def-lingjzhu/tltse/dictionary_learning_v2/dictionary_learning:$PYTHONPATH
free -h

#python train.py \
python multi_generate_and_score_ta1.py \
    --dict_learning_model_dir "/scratch/tltse/v2_jumprelu_testing_3mill_unpool_webdataset/webdataset/" \
    --eval_dataset "/project/def-lingjzhu/tltse/official_data/hrs2_release-6-19-24"
