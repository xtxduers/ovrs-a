python train_net.py --config configs/vitl_336_DLRSD.yaml \
    --num-gpus 1 \
    --eval-only \
    OUTPUT_DIR output_DLRSD_addRCS/eval_Potsdam01 \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Potsdam.json" \
    DATASETS.TEST "(\"Potsdam_all_sem_seg\",)" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS output_DLRSD_addRCS/model_final.pth