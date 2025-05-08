CONFIG=spi0_moz1_eef_y_dyfv2_step0to1_0507 # config的名字
TASK=spi0_moz1_eef_y_dyfv2_step0to1_0507 # wandb的任务名，需要每次有区别，要不然会报错或者resume

source /pfstem/yifeng/PI_Official/.venv/bin/activate # official pi的环境，自己安装
cd /pfstem/yifeng/PI_Official # 代码位置
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=/pfstem/yifeng/resources/.cache # 缓存位置，不要放到系统盘，会崩掉

#train model
#配置好卡数ID，就可以开始训练了
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py $CONFIG --exp-name=$TASK --resume