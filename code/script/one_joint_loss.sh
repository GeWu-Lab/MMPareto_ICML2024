#!/bin/bash
python one_joint_loss.py \
--dataset CREMAD \
--model one_joint_loss \
--gpu_ids 1 \
--n_classes 6 \
--train \
| tee log_print/one_joint_loss.log