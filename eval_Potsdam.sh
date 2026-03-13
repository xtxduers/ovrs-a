python train_net.py --config configs/vitl_336_DLRSD.yaml \
    --num-gpus 4 \
    --eval-only \
    OUTPUT_DIR output_paper_DLRSD_test/eval_Potsdam \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Potsdam.json" \
    DATASETS.TEST "(\"Potsdam_all_sem_seg\",)" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS output_paper_DLRSD_test/model_final.pth