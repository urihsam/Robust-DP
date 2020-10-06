python3 ./code/run_robust_dp_mnist.py --BATCH_SIZE 128 --BATCHES_PER_LOT 1\
    --NUM_CLASSES 10 --load_model=True\
    --DATA_DIR ../mnist \
    --CNN_CKPT_RESTORE_NAME robust_dp_cnn.epoch199.vloss0.013280.vacc0.981658.noise_eps0.224806.noise_delta0.000010.sgd_eps0.000000.sgd_delta1.000000.ckpt\
    --SGD_DP_INFO_NPY sgd_dp_info.sgd_eps0.000000.sgd_delta1.000000.npy\
    --NOISE_DP_INFO_NPY noise_dp_info.noise_eps0.224806.noise_delta0.000010.npy\
    --ATTACK_SIZE 1.0 --ROBUST_SIGMA 0.5 --ROBUST_SAMPLING_N 2000\
    --TOTAL_EPS 1.0 --DP_GRAD_CLIPPING_L2NORM 1.0\
    --MAX_NOISE_DELTA 1e-5 --G_LIP 0.5 --K_SCONVEX 50.0 --LOSS_LOWER_BOUND 1e-2\
    --MAX_SGD_DELTA 0.0