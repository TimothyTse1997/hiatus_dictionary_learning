#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
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
echo "load code from dictionary_learning"
export PYTHONPATH=/project/def-lingjzhu/tltse/dictionary_learning_v2/dictionary_learning:$PYTHONPATH
free -h
echo "training python script starts!"

#python train.py \
/scratch/tltse/python_virtual_env/hiatus2/bin/python train.py \
    --data_path "/scratch/tltse/data/idiolect_embeddings/full/vectors_data/data-{00000..00297}.tar" \
    --save_dir "/scratch/tltse/v2_jumprelu_testing_3mill_unpool_webdataset/" \
    --model_config_fname "/project/def-lingjzhu/tltse/hiatus_dictionary_learning/configs/training_jumprelu_config.json" \
    --training_config_fname "/project/def-lingjzhu/tltse/hiatus_dictionary_learning/configs/default_training_cfg.json" \
    --batch_size  16384 \
    --alpha 0.1 0.01 0.001 0.0001 0.00001
    #--multi_config_parallel_training \