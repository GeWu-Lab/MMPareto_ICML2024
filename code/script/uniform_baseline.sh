#!/bin/bash
python uniform_baseline.py \
--dataset CREMAD \
--model uniform_baseline \
--gpu_ids 2 \
--n_classes 6 \
--train \
| tee log_print/uniform_baseline.log