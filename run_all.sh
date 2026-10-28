#!/bin/bash
cd /root/IL/ZFCIL
PY=/root/anaconda3/envs/bimc/bin/python
mkdir -p logs

for cfg in config/umerced.yaml config/aid.yaml config/rsi.yaml config/nwpu.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "===== START $cfg $(date) ====="
  $PY -u main.py --config "$cfg" 2>&1 | tee "logs/$name.log"
  rc=${PIPESTATUS[0]}
  echo "===== DONE $cfg exit=$rc $(date) ====="
done

echo "ALL DONE"
