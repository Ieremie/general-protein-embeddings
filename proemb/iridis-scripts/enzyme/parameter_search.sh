#!/bin/bash

# Submits a slurm job for each parameter set
batch_size=(32)

mlp_lrs=(0.0001)
encoder_lrs=(0.0001 0.00005)

dropout_mlp=(0.7 0.65 0.6 0.55 0.5 0.4 0.3)

unfreeze_encoder=(10)
mlp_hidden=(1024 512)
augment_prob=(0 0.05 0.1 0.15 0.2 0.25 0.3)
# proj_head_names=("cloze" "sim")
noise_type=("uniref90" "pfam-hmm")

#model_path="/home/ii1g17/protein-embeddings/proemb/iridis-scripts/saved_models/multitask/rtx8000-4gpu/2422084/iter_24000_checkpoint.pt"
model_path="/home/ii1g17/protein-embeddings/proemb/iridis-scripts/saved_models/multitask/rtx8000-4gpu/2324180/iter_240000_checkpoint.pt"
enzyme_data_path="/scratch/ii1g17/protein-embeddings/data/enzyme"

# run a parameter search
for bs in ${batch_size[@]}; do
    for mlp_lr in ${mlp_lrs[@]}; do
        for enc_lr in ${encoder_lrs[@]}; do
            for ml_dp in ${dropout_mlp[@]}; do
                for unfreeze_encoder_epoch in ${unfreeze_encoder[@]}; do
                    for mlp_dim in ${mlp_hidden[@]}; do
                        for aug_prob in ${augment_prob[@]}; do
                          #for proj_head in ${proj_head_names[@]}; do
                          for noise in ${noise_type[@]}; do

                            # printing informative message
                            echo "Submitting job for batch_size: $bs, mlp_lr: $mlp_lr, encoder_lr: $enc_lr,
                            dropout_mlp: $ml_dp, unfreeze_encoder: $unfreeze_encoder and mlp_dim: $mlp_dim"

                           c --model_path=$model_path --enzyme_data_path=$enzyme_data_path --batch_size=$bs \
                            --hidden_dim=$mlp_dim --dropout_head=$ml_dp --encoder_lr=$enc_lr --lr=$mlp_lr  \
                            --max_epochs=500 --remote=True --unfreeze_encoder_epoch=$unfreeze_encoder_epoch \
                            --augment_prob=$aug_prob --noise_type=$noise

                            # wait for 1 second before submitting next job
                            sleep 1

                            done
                        done
                    done
                done
            done
        done
    done
done
