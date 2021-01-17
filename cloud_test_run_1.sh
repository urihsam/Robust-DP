python3 ./code/run_robust_dp_mnist.py --BATCH_SIZE 128 --BATCHES_PER_LOT 1\
    --GPU_INDEX 1\
    --DATA_DIR ../mnist-new\
    --NUM_CLASSES 10 --load_model=True\
    --CNN_CKPT_RESTORE_NAME robust_dp_cnn.epoch199.vloss0.013280.vacc0.981658.noise_eps0.224806.noise_delta0.000010.sgd_eps0.000000.sgd_delta1.000000.ckpt\
    --DP_INFO_NPY sgd_dp_info.sgd_eps0.000000.sgd_delta1.000000.npy\
    --ATTACK_SIZE 1.0 --INPUT_SIGMA 1.0 --NUM_SAMPLING 200