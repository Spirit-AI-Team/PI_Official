CONFIG=spi0_aloha_eef_multi_full2
# TASK=shirt_pretrains2_0225
# TASK=FlattenShirt_EEF_0306_11_debug
TASK=YC_MultiTask_PutPlateOnRack_0510_12
CKPT=29999

cd /root/PI_Official
source .venv/bin/activate
export XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache

#compute norm
CUDA_VISIBLE_DEVICES=2 python scripts/compute_norm_stats_multi.py --config_name $CONFIG --max_frames 10000

#train model
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py $CONFIG --exp-name=$TASK --resume

cd /pfstem/likaiyu/mozbrain
python lerobot/common/policies/pi0/conversion_scripts/convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir /pfstem/likaiyu/mozbrain/data/checkpoints/$CONFIG/$TASK/$CKPT/params \
    --output_path /pfstem/likaiyu/mozbrain/data/lerobot_ckpts/$TASK

python /pfstem/likaiyu/mozbrain/add_norm_stats_to_model.py \
    --asset_path /pfstem/likaiyu/mozbrain/data/checkpoints/$CONFIG/$TASK/$CKPT/assets \
    --test_model_path /pfstem/likaiyu/mozbrain/data/lerobot_ckpts/$TASK

cd /pfstem/likaiyu/mozbrain/data/lerobot_ckpts
zip -r "$TASK"_lerobot.zip $TASK

expect <<EOF
spawn oss login
expect {
    "Username:" { send "\b\b\b\b\b\b\b\b\b\b\b18401132402\r" }
}
expect "Password:" { send "spirit-ai\r" }
expect eof
EOF
oss cp "$TASK"_lerobot.zip oss://likaiyu/weights/
rm "$TASK"_lerobot.zip
cd /pfstem/likaiyu/mozbrain
# #zip ckpt
# cd /pfstem/likaiyu/resources/checkpoints/$CONFIG/$TASK
# mv $CKPT/train_state ./
# zip -r $TASK.zip $CKPT

# #oss upload
# expect <<EOF
# spawn oss login
# expect {
#     "Username:" { send "\b\b\b\b\b\b\b\b\b\b\b18401132402\r" }
# }
# expect "Password:" { send "spirit-ai\r" }
# expect eof
# EOF

# oss ls -s -d oss://likaiyu
# oss cp $TASK.zip oss://likaiyu/weights/
# cd /root/PI_Official