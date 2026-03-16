CUDA_VISIBLE_DEVICES=4 python train_net.py --config-file configs/vitl_336_DLRSD.yaml --num-gpus 1 \
    OUTPUT_DIR ./output_DLRSD_fixattention \
    SOLVER.MAX_ITER 100000 \
    SOLVER.WEIGHT_DECAY 0.001 \
    INPUT.MIN_SIZE_TEST 384 \
    TEST.EVAL_PERIOD 1000000 
