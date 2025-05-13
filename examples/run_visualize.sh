export XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache
# export HF_HOME=/pfstem/wenxuan/resources/.cache/huggingface
# export PYTHONPATH=./
export HF_HUB_OFFLINE=1

TASK=HRPI_MultiTask_PutPlateOnRack_0423_29
POLICY=/root/PI_Official/data/lerobot_ckpts/HRPI_MultiTask_PutPlateOnRack_0423_29
# REPO_ID=lerobot/$TASK
DATAPATH=/root/PI_Official/data/lerobot/HRPI_MultiTask_PutPlateOnRack_0423_29

cd /root/PI_Official
source .venv/bin/activate
# CUDA_VISIBLE_DEVICES=0 python lerobot/scripts/infer_rjointtoeef.py --policy.path=$POLICY --env.type=pusht --dataset.repo_id=$REPO_ID --dataset.root=$DATAPATH #--dataset.episodes="[260,261]"
CUDA_VISIBLE_DEVICES=7 python examples/visualize_3d_trace.py --policy.path=$POLICY --env.type=pusht --dataset.repo_id=$TASK --dataset.root=$DATAPATH #--dataset.episodes="[260,261]"



