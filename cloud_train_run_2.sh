python3 ./code/run_robust_dp_mnist_v2.py --train --BATCH_SIZE 8 --BATCHES_PER_LOT 1\
    --EVAL_TRAIN_FREQUENCY 50 --EVAL_VALID_FREQUENCY 2\
    --GPU_INDEX -1\
    --DATA_DIR ../mnist-new\
    --TRAIN_LOG_FILENAME mnist_train_log.txt\
    --VALID_LOG_FILENAME mnist_valid_log.txt\
    --NUM_EPOCHS 200 --NUM_CLASSES 10 --load_model=False\
    --MAX_PARAM_SIZE 18 --IS_MGM_LAYERWISED=True\
    --BETA 200.0 --LEARNING_RATE 1e-3 --LEARNING_DECAY_RATE 0.98 --LEARNING_DECAY_STEPS 10000\
    --INPUT_SIGMA 0.2 --INPUT_DP_SIGMA_THRESHOLD 0.8 --TOTAL_DP_SIGMA 1.0 --TOTAL_DP_DELTA 1e-5 --TOTAL_DP_EPSILON 1.0\
    --MAX_ITERATIONS 800000\
    --DP_GRAD_CLIPPING_L2NORM 1.0\