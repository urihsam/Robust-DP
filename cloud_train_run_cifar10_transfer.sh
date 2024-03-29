python3 ./code/run_robust_dp_cifar10_transfer.py --train\
    --GPU_INDEX -1\
    --IMAGE_ROWS 32 --IMAGE_COLS 32 --NUM_CHANNELS 3 --NUM_CLASSES 10\
    --load_model=False\
    --CNN_CKPT_RESTORE_NAME robust_dp_cnn.epoch124.vloss221.982345.vacc0.824167.input_sigma5.0000.total_sigma3.9236.dp_eps1.000000.dp_delta0.000000.ckpt\
    --load_pretrained=True\
    --BETA 1000.0 --REG_SCALE 1.0\
    --BETA_BOTT 100.0 --REG_SCALE_BOTT 1e-2 --BOTT_TRAIN_FREQ_TOTAL 2 --BOTT_TRAIN_FREQ 1\
    --LEARNING_RATE_0 1e-4 --LEARNING_RATE_1 1e-3 --LEARNING_RATE_2 1e-3\
    --LEARNING_DECAY_RATE_0 0.99 --LEARNING_DECAY_RATE_1 0.99 --LEARNING_DECAY_RATE_2 0.99\
    --LEARNING_DECAY_STEPS_0 20000 --LEARNING_DECAY_STEPS_1 20000 --LEARNING_DECAY_STEPS_2 20000\
    --BATCH_SIZE 16 --BATCHES_PER_LOT 4 --NUM_EPOCHS 2000\
    --EVAL_TRAIN_FREQUENCY 10 --EVAL_VALID_FREQUENCY 50\
    --TRAIN_LOG_FILENAME cifar10_train_log.txt\
    --VALID_LOG_FILENAME cifar10_valid_log.txt\
    --TEST_LOG_FILENAME cifar10_test_log.txt\
    --TOTAL_DP_SIGMA 3.0 --TOTAL_DP_DELTA 1e-5 --TOTAL_DP_EPSILON 1.0\
    --TOTAL_DP_SIGMA_DECAY_RATE 0.997 --TOTAL_DP_SIGMA_DECAY_EPOCH 1 --MIN_TOTAL_DP_SIGMA 1.0\
    --INPUT_SIGMA 0.2 --MAX_PARAM_SIZE 18 --IS_MGM_LAYERWISED=False\
    --MAX_ITERATIONS 20000\
    --DP_GRAD_CLIPPING_L2NORM_BOTT 1.0 --DP_GRAD_CLIPPING_L2NORM_1 1e-3 --DP_GRAD_CLIPPING_L2NORM_2 1e-3\