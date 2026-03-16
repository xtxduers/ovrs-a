output=$1
if [ -z $output ]
then
    echo "No output directory found! Run with "sh show_result.sh [OUTPUT_DIR]""
    exit 0
fi

python train_net.py --config configs/vitl_336_DLRSD.yaml \
    --num-gpus 2 \
    --eval-only \
    OUTPUT_DIR $output/eval_DLRSD \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/DLRSD.json" \
    DATASETS.TEST "(\"DLRSD_all_sem_seg\",)" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS $output/model_final.pth