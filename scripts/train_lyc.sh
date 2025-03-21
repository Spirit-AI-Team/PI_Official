CONFIG=spi0_aloha_multitask4_full
TASK=multitask4_0312_full
CKPT=99999

cd /pfstem/lyc/project/PI_Official
source .venv/bin/activate
export XDG_CACHE_HOME=/pfstem/lyc/.cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# # compute norm
# CUDA_VISIBLE_DEVICES=2 python scripts/compute_norm_stats.py --config_name $CONFIG --max_frames 10000

#train model
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py $CONFIG --exp-name=$TASK --resume

#zip ckpt
cd /pfstem/lyc/checkpoints/$CONFIG/$TASK
mv $CKPT/train_state .
tar -cf - $CKPT | pigz > $TASK.tar.gz
mv train_state $CKPT/

#oss upload
expect <<EOF
spawn oss login
expect {
    "Username:" { send "\b\b\b\b\b\b\b\b\b\b\b15600155670\r" }
}
expect "Password:" { send "spirit-ai\r" }
expect eof
EOF

oss ls -s -d oss://lyc
oss cp $TASK.tar.gz oss://lyc/weights/
cd /pfstem/lyc/project/PI_Official