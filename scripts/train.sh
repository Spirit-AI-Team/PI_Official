CONFIG=spi0_aloha_s2pretrain_full
TASK=shirt_pretrains2_0225
# TASK=shirtflatten_0225_base0223
CKPT=10000

cd /root/PI_Official
source .venv/bin/activate

#compute norm
# CUDA_VISIBLE_DEVICES=2 python scripts/compute_norm_stats.py --config_name $CONFIG --max_frames 10000

#train model
CUDA_VISIBLE_DEVICES=6,7 python scripts/train.py $CONFIG --exp-name=$TASK --resume

#zip ckpt
cd /pfstem/likaiyu/resources/checkpoints/$CONFIG/$TASK
zip -r $TASK.zip $CKPT

#oss upload
expect <<EOF
spawn oss login
expect {
    "Username:" { send "\b\b\b\b\b\b\b\b\b\b\b15600155670\r" }
}
expect "Password:" { send "spirit-ai\r" }
expect eof
EOF

oss ls -s -d oss://likaiyu
oss cp $TASK.zip oss://likaiyu/weights/
cd /root/PI_Official