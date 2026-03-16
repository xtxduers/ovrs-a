output=$1
if [ -z $output ]
then
    echo "No output directory found! Run with "sh show_result.sh [OUTPUT_DIR]""
    exit 0
fi


python train_net.py --config configs/vitl_336_DLRSD.yaml \
    --num-gpus 4 \
    --eval-only \
    OUTPUT_DIR $output/eval_iSAID \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/iSAID.json" \
    DATASETS.TEST "(\"iSAID_all_sem_seg\",)" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS $output/model_final.pth