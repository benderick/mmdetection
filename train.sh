#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3 PORT=8787 ./tools/dist_train.sh projects/RS/SparseInst/configs/sparseinst.py 2  --work-dir ./log/spare 