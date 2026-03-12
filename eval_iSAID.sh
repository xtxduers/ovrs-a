python train_net.py --config configs/your_config.yaml \
    --num-gpus 2 \
    --eval-only \
    OUTPUT_DIR output_orignal_DLRSD/eval_iSAID \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/iSAID.json" \
    DATASETS.TEST "(\"iSAID_all_sem_seg\",)" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS output_orignal_DLRSD/model_final.pth