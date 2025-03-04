CONFIG=spi0_aloha_s2pretrain_full
# TASK=shirt_pretrains2_0225
TASK=shirt_pretrains2_halfall_retrain
CKPT=9999

cd /root/PI_Official
source .venv/bin/activate
export XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache

#compute norm
# CUDA_VISIBLE_DEVICES=0 python scripts/compute_norm_stats.py --config_name $CONFIG --max_frames 20000

#train model
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py $CONFIG --exp-name=$TASK --resume

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