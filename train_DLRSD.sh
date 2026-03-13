python train_net.py --config-file configs/vitl_336_DLRSD.yaml --num-gpus 4 \
    OUTPUT_DIR ./output_paper_DLRSD_test \
    SOLVER.MAX_ITER 100000 \
    SOLVER.WEIGHT_DECAY 0.001 \
    INPUT.MIN_SIZE_TEST 384 \
    TEST.EVAL_PERIOD 1000000 
