#!/usr/bin/env sh

PYTHON="/home/mengjian/anaconda3/bin/python3"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=squeezenet1_1_grey
dataset=eyeclosure
epochs=50
batch_size=32
optimizer=SGD

# add more labels as additional info into the saving path
label_info=

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/SqueezeNet/decay0.0002_w32_a32_eyeclosure \
    --epochs ${epochs}  --learning_rate  0.01 \
    --optimizer ${optimizer} \
    --schedule 30 40   --gammas 0.1 0.1\
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 2 \
    --print_freq 100 --decay 0.0001