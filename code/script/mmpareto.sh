#!/bin/bash
python MMPareto.py \
--dataset CREMAD \
--model MMPareto \
--gpu_ids 3 \
--n_classes 6 \
--train \
| tee log_print/MMPareto.log