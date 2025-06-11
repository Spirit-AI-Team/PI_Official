#!/bin/bash

NAME=  # 改成自己的用户名
IMAGE=ccr-2z42czyz-vpc.cnc.bj.baidubce.com/gpua800/mozbrain:v20250407  # 改成需要使用的最新镜像

# 创建个人文件夹并挂在进容器，后续容器内/root/xiejunyuan文件夹内的文件会同步到容器外

docker run -d --name=$NAME \
    -v /mnt/vepfs01/:/mnt/vepfs01/:rw \
    --gpus=all \
    --shm-size=980g \
    $IMAGE \
    tail -f /dev/null