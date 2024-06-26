#!/usr/bin/env bash
set -ex

# Evaluation 
# The current run file performs inference for one test dataset of RAFT32-PIV and RAFT256-PIV.
# Please modify to obtain results of other datasets and pretrained models.

# RAFT32-PIV
python RAFT32-PIV_test.py --nodes 1 --gpus 1 --name RAFT32-PIV_test_backstep \
--input_path_ckpt ./precomputed_ckpts/RAFT32-PIV_ProbClass1/ckpt.tar --test_dataset backstep \
--plot_results True --output_dir_results ../results/

# RAFT256-PIV
# python RAFT256-PIV_test.py --nodes 1 --gpus 1 --name RAFT256-PIV_test_cylinder \
# --input_path_ckpt ./precomputed_ckpts/RAFT256-PIV_ProbClass2/ckpt.tar --test_dataset cylinder \
# --plot_results True --output_dir_results ../results/

# Training
# The current run file performs training for one specific training configuration of RAFT32-PIV 
# and RAFT256-PIV using the provided minimal datasets. 
# Please uncomment if you wish to train on the minimal datasets. However, we do not recommend
# training via the Code Ocean platform due to the time required.

# RAFT32-PIV 
#python RAFT32-PIV_train.py --nodes 1 --gpus 1 --name RAFT32-PIV_newModel_ProbClass2 \
#--batch_size 25 --epochs 10 --recover False --output_dir_ckpt ../results/ \
#--train_tfrecord ../data/minimal_training_dataset_ProbClass2_32px.tfrecord-00000-of-00001 \
#--train_tfrecord_idx ../data/idx_files/minimal_training_dataset_ProbClass2_32px.idx \
#--val_tfrecord ../data/minimal_validation_dataset_ProbClass2_32px.tfrecord-00000-of-00001 \
#--val_tfrecord_idx ../data/idx_files/minimal_validation_dataset_ProbClass2_32px.idx

# RAFT256-PIV
#python RAFT256-PIV_train.py --nodes 1 --gpus 1 --name RAFT256-PIV_newModel_ProbClass1 \
#--batch_size 5 --epochs 50 --recover False --output_dir_ckpt ../results/ \
#--train_tfrecord ../data/minimal_training_dataset_ProbClass1_256px.tfrecord-00000-of-00001 \
#--train_tfrecord_idx ../data/idx_files/minimal_training_dataset_ProbClass1_256px.idx \
#--val_tfrecord ../data/minimal_validation_dataset_ProbClass1_256px.tfrecord-00000-of-00001 \
#--val_tfrecord_idx ../data/idx_files/minimal_validation_dataset_ProbClass1_256px.idx