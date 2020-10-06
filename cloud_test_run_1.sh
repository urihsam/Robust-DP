python3 ./code/run_robust_dp_mnist.py --BATCH_SIZE 128 --BATCHES_PER_LOT 1\
    --NUM_CLASSES 10 --load_model=True\
    --DATA_DIR ../mnist \
    --CNN_CKPT_RESTORE_NAME robust_dp_cnn.epoch33.vloss0.066208.vacc0.904382.noise_eps0.065728.noise_delta0.000002.sgd_eps0.934272.sgd_delta0.000008.ckpt\
    --SGD_DP_INFO_NPY sgd_dp_info.sgd_eps0.934272.sgd_delta0.000008.npy\
    --NOISE_DP_INFO_NPY noise_dp_info.noise_eps0.065728.noise_delta0.000002.npy\
    --ATTACK_SIZE 1.0 --ROBUST_SIGMA 0.1 --ROBUST_SAMPLING_N 2000\
    --TOTAL_EPS 1.0 --DP_GRAD_CLIPPING_L2NORM 1.0\
    --MAX_NOISE_DELTA 2e-6 --G_LIP 0.1 --K_SCONVEX 50.0 --LOSS_LOWER_BOUND 1e-2\
    --MAX_SGD_DELTA 8e-6