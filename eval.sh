#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found!  Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval. sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

#iSAID_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_iSAID \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/iSAID.json" \
 DATASETS.TEST \(\"iSAID_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#DLRSD_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_DLRSD \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/DLRSD.json" \
 DATASETS.TEST \(\"DLRSD_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Potsdam_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_Potsdam \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Potsdam.json" \
 DATASETS.TEST \(\"Potsdam_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts


#Vaihingen_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_Vaihingen \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Vaihingen.json" \
 DATASETS.TEST \(\"Vaihingen_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#LoveDA_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_LoveDA \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/LoveDA.json" \
 DATASETS.TEST \(\"LoveDA_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#UDD5_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_UDD \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/UDD5.json" \
 DATASETS.TEST \(\"UDD5_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#VDD_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_VDD \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/VDD.json" \
 DATASETS.TEST \(\"VDD_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#uavid_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval_uavid \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/uavid.json" \
 DATASETS.TEST \(\"uavid_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# 打印所有数据集的评估结果
echo "========== iSAID Results =========="
cat $output/eval_iSAID/log.txt | grep copypaste

echo "========== DLRSD Results =========="
cat $output/eval_DLRSD/log.txt | grep copypaste

echo "========== Potsdam Results =========="
cat $output/eval_Potsdam/log.txt | grep copypaste

echo "========== Vaihingen Results =========="
cat $output/eval_Vaihingen/log.txt | grep copypaste

echo "========== LoveDA Results =========="
cat $output/eval_LoveDA/log.txt | grep copypaste

echo "========== UDD5 Results =========="
cat $output/eval_UDD5/log.txt | grep copypaste

echo "========== VDD Results =========="
cat $output/eval_VDD/log.txt | grep copypaste

echo "========== uavid Results =========="
cat $output/eval_uavid/log.txt | grep copypaste

