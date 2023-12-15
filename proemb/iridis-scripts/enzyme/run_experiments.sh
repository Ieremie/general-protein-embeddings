#!/bin/bash

# run the enzyme training with the best parameters found in the parameter search
# run across a few lr pairs to get a better idea of the performance

mlp_lrs=(0.0001 0.00019 0.00019 0.0002)
encoder_lrs=(0.00005 0.00003 0.00005 0.00004)

model_path=$1
alphabet_type=$2
augment_prob=$3
noise_type=$4
softmax_tmp=$5
dont_augment_prob=$6

if [ "$7" -eq 1 ]; then
  full_hmm_augment_hits="--full_hmm_augment_hits"
else
  full_hmm_augment_hits=""
fi

script_path="/home/ii1g17/protein-embeddings/proemb/iridis-scripts/enzyme/allgpu.sh"


for i in "${!mlp_lrs[@]}"; do
    mlp_lr=${mlp_lrs[i]}
    encoder_lr=${encoder_lrs[i]}

    # for each mlp/enc lr pair submit job
    sbatch "$script_path"  --alphabet_type=$alphabet_type --augment_prob=$augment_prob --batch_size=32 --dropout_head=0.7 \
    --early_stopping_metric='val_acc' --encoder_lr=$encoder_lr \
    --enzyme_data_path='/scratch/ii1g17/protein-embeddings/data/enzyme' --factor=0.6  \
    --gradient_clip_val=3 --head_type='mlp3' --hidden_dim=1024 --lr=$mlp_lr --max_epochs=500 \
    --model_path=$model_path --noise_type=$noise_type \
    --patience=20 --proj_head=None --remote --scheduler_patience=3  --unfreeze_encoder_epoch=15 --filter_dataset --softmax_tmp=$softmax_tmp \
    --dont_augment_prob=$dont_augment_prob $full_hmm_augment_hits

done



