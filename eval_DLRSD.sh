python train_net.py --config configs/vitl_336_DLRSD.yaml \
    --num-gpus 2 \
    --eval-only \
    OUTPUT_DIR output_paper_DLRSD_test/eval_DLRSD \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/DLRSD.json" \
    DATASETS.TEST "(\"DLRSD_all_sem_seg\",)" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS output_paper_DLRSD_test/model_final.pth