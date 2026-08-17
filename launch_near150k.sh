#!/bin/bash
# Dual-5090 near-150k campaign launcher (v2).
# RESTORE AFTER CAMPAIGN:  docker start vllm-qwen-32b && docker start infinity-embed
set -e
cd /root/lps632

echo "=== stopping GPU tenants (restore: docker start vllm-qwen-32b infinity-embed) ==="
docker stop vllm-qwen-32b 2>/dev/null || echo "vllm already stopped"
docker stop infinity-embed 2>/dev/null || echo "infinity already stopped"
sleep 6

echo "=== SELinux-safe binary location ==="
cp -f /root/lps632/lps632gpu /usr/local/bin/lps632gpu
restorecon -v /usr/local/bin/lps632gpu || true

echo "=== resources ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,power.limit --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
free -g | head -2

echo "=== launching dual-GPU near-150k ==="
systemctl reset-failed lps632-near150k 2>/dev/null || true
systemd-run --unit=lps632-near150k --nice=5 \
  --property=WorkingDirectory=/root/lps632 \
  --property=Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /usr/local/bin/lps632gpu search \
    --B 150001 --near 1 --devices 0,1 --sample-shift 12 --delta-bits 40 \
    --cand-cap 134217728 --outdir /root/lps632/run_near_150k --auto-resume
sleep 25
systemctl is-active lps632-near150k
journalctl -u lps632-near150k --no-pager | tail -12
