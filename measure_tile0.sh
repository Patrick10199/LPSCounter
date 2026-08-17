#!/bin/bash
# Measure the REAL candidate economics of the near-150k workload on one GPU,
# instead of estimating them. Runs a bounded prefix of the campaign (a few
# x-blocks of window 0, the worst case) and reports:
#   - true 64-bit candidate count
#   - kernel time vs host recovery time
#   - recoveries/sec and candidates/iteration
# Then projects the full campaign from measured numbers.
set -e
cd /root/lps632
rm -rf /root/lps632/measure_w0

# --tiles raises the x-block count so a prefix is a small, well-defined slice.
# --max-tiles bounds the run. Both are config-hashed, so this is its own run id.
CUDA_VISIBLE_DEVICES=1 nice -n 5 ./lps632gpu search \
  --B 150001 --near 1 --devices 0 --windows 5 --tiles 46870 --max-tiles 6 \
  --sample-shift -1 --cand-cap 134217728 \
  --outdir /root/lps632/measure_w0 2>&1 | tail -25
