from dependency import *
import os

def set_flags():
    flags.DEFINE_bool("train", False, "Train and save the ATN model.")
    flags.DEFINE_bool("local", False, "Run this model locally or on the cloud")
    # GPU params
    flags.DEFINE_string("GPU_INDEX", "0", "GPU index number")
    # Path
    flags.DEFINE_string("CNN_PATH", "./models/CNN", "Path of MNIST cnn")
    flags.DEFINE_string("PRETRAINED_CNN_PATH", "./models/CNN", "Path of MNIST cnn")
    flags.DEFINE_string("CNN_CKPT_RESTORE_NAME", "robust_dp_cnn.ckpt", "Path of MNIST cnn")
    flags.DEFINE_string("PRETRAINED_CNN_CKPT_RESTORE_NAME", "robust_dp_cnn.ckpt", "Path of MNIST cnn")
    flags.DEFINE_string("TRAIN_LOG_PATH", "./graphs/train", "Path of log for training")
    flags.DEFINE_string("VALID_LOG_PATH", "./graphs/valid", "Path of log for validation")
    flags.DEFINE_string("TEST_LOG_PATH", "./graphs/test", "Path of log for testing")
    flags.DEFINE_string("TRAIN_LOG_FILENAME", "mnist_train_log.txt", "Path of log for training")
    flags.DEFINE_string("VALID_LOG_FILENAME", "mnist_valid_log.txt", "Path of log for validation")
    flags.DEFINE_string("TEST_LOG_FILENAME", "mnist_test_log.txt", "Path of log for testing")
    flags.DEFINE_string("FINETUNE_TRAIN_LOG_FILENAME", "mnist_fine_train_log.txt", "Path of log for training")
    flags.DEFINE_string("FINETUNE_VALID_LOG_FILENAME", "mnist_fine_valid_log.txt", "Path of log for validation")
    flags.DEFINE_string("FINETUNE_TEST_LOG_FILENAME", "mnist_fine_test_log.txt", "Path of log for testing")
    flags.DEFINE_string("ROBUST_LOG_FILENAME", "robust_log_.txt", "Path of log for robust")
    flags.DEFINE_string("DATA_DIR", "/Users/mashiru/Life/My-Emory/Research/Research-Project/Data/mnist", "Data dir")
    flags.DEFINE_string("ADV_PATH_PREFIX", "", "Prefix path for adv examples")
    # Data description
    flags.DEFINE_bool("NORMALIZE", True, "Data is normalized to [0, 1]")
    flags.DEFINE_bool("BIASED", False, "Data is shifted to [-1, 1]")
    flags.DEFINE_integer("NUM_CLASSES", 10, "Number of classification classes")
    flags.DEFINE_integer("IMAGE_ROWS", 28, "Input row dimension")
    flags.DEFINE_integer("IMAGE_COLS", 28, "Input column dimension")
    flags.DEFINE_integer("NUM_CHANNELS", 1, "Input depth dimension")
    # Training params
    flags.DEFINE_integer("NUM_EPOCHS", 1, "Number of epochs") # 200
    flags.DEFINE_integer("NUM_FINETUNE_EPOCHS", 1, "Number of epochs") # 200
    flags.DEFINE_integer("NUM_ACCUM_ITERS", 2, "Number of accumulation") # 2
    flags.DEFINE_integer("BATCH_SIZE", 128, "Size of training batches")# 128
    flags.DEFINE_integer("BATCHES_PER_LOT", 1, "Number of batches per lot")
    flags.DEFINE_integer("FINETUNE_BATCH_SIZE", 128, "Size of training batches")# 128
    flags.DEFINE_integer("FINETUNE_BATCHES_PER_LOT", 1, "Number of batches per lot")
    flags.DEFINE_integer("EVAL_TRAIN_FREQUENCY", 1, "Frequency for evaluation") # 25
    flags.DEFINE_integer("EVAL_VALID_FREQUENCY", 1, "Frequency for evaluation") # 25
    flags.DEFINE_integer("EVAL_FINETUNE_TRAIN_FREQUENCY", 1, "Frequency for evaluation") # 25
    flags.DEFINE_integer("EVAL_FINETUNE_VALID_FREQUENCY", 1, "Frequency for evaluation") # 25
    flags.DEFINE_bool("TRAIN_BEFORE_FINETUNE", True, "Train model before finetune")
    flags.DEFINE_bool("load_model", False, "Load model from the last training result or not")
    flags.DEFINE_bool("load_pretrained", False, "Load model from the last training result or not")
    flags.DEFINE_bool("early_stopping", False, "Use early stopping or not")
    flags.DEFINE_integer("EARLY_STOPPING_THRESHOLD", 10, "Early stopping threshold")
    # Loss param
    flags.DEFINE_float("BETA", 100.0, "Coefficient for loss")
    flags.DEFINE_float("FINETUNE_BETA", 100.0, "Coefficient for loss")
    # Optimization params
    flags.DEFINE_string("OPT_TYPE", "ADAM", "The type of optimization") # ADAM, MOME, NEST
    flags.DEFINE_float("BATCH_MOME", 0.99, "Momentum for the moving average")
    flags.DEFINE_float("BATCH_EPSILON", 0.001, "Small float added to variance to avoid dividing by zero")
    flags.DEFINE_float("LEARNING_RATE", 1e-4, "Learning rate of optimization")
    flags.DEFINE_float("LEARNING_DECAY_RATE", 0.99, "Decay rate of learning rate")
    flags.DEFINE_integer("LEARNING_DECAY_STEPS", int(2.5*1e3), "Decay steps of learning rate")
    flags.DEFINE_float("FINETUNE_LEARNING_RATE", 1e-4, "Learning rate of optimization")
    flags.DEFINE_float("FINETUNE_LEARNING_DECAY_RATE", 0.99, "Decay rate of learning rate")
    flags.DEFINE_integer("FINETUNE_LEARNING_DECAY_STEPS", int(2.5*1e3), "Decay steps of learning rate")
    flags.DEFINE_bool("IS_GRAD_CLIPPING", False, "Use gradient clipping or not")
    flags.DEFINE_float("GRAD_CLIPPING_NORM", 10.0, "Gradient clipping norm")
    # Robust params
    flags.DEFINE_bool("LOAD_ADVS", False, "Whether to load adv examples or not")
    flags.DEFINE_float("ATTACK_SIZE", 1.0, "Attack size")
    flags.DEFINE_integer("ROBUST_SAMPLING_N", 2000, "Number of sampling")
    # DP params
    flags.DEFINE_integer("MAX_ITERATIONS", 500000, "Max number of perturbation")
    flags.DEFINE_integer("MAX_FINETUNE_ITERATIONS", 500000, "Max number of perturbation")
    flags.DEFINE_string("DP_INFO_NPY", "sgd_dp_info_.npy", "npy file name for dp sgd")
    flags.DEFINE_string("FINETUNE_DP_INFO_NPY", "finetune.sgd_dp_info_.npy", "npy file name for dp sgd")
    flags.DEFINE_float("DP_GRAD_CLIPPING_L2NORM", 1.0, "DP gradient clipping in L2 norm")
    flags.DEFINE_float("FINETUNE_DP_GRAD_CLIPPING_L2NORM", 1.0, "DP gradient clipping in L2 norm")
    flags.DEFINE_integer("MAX_PARAM_SIZE", 10, "DP gradient clipping in L2 norm")
    flags.DEFINE_integer("ACCOUNT_NUM", 10, "Num of param layers for account")
    flags.DEFINE_integer("DP_ACCOUNTANT_ITERATION", 20, "The Iteration threshold for dp accountant")
    flags.DEFINE_bool("IS_MGM_LAYERWISED", True, "Whether calculate K norm layerwisedly or not")
    ## perturbation param
    flags.DEFINE_float("INPUT_SIGMA", 1.0, "The sigma of input perturbation")
    flags.DEFINE_float("INPUT_DP_SIGMA_THRESHOLD", 0.6, "The sigma threshold of input perturbation")
    flags.DEFINE_float("GRADIENT_SIGMA", 1.0, "The sigma of gradient perturbation")
    flags.DEFINE_float("TOTAL_DP_SIGMA", 1.0, "The sigma of total perturbation")
    flags.DEFINE_float("TOTAL_DP_EPSILON", 0.1, "The epsilon of total perturbation")
    flags.DEFINE_float("TOTAL_DP_DELTA", 1e-5, "The delta of total perturbation")
    flags.DEFINE_float("TOTAL_DP_SIGMA_DECAY_RATE", 0.99, "The decay rate of total dp sigma")
    flags.DEFINE_integer("TOTAL_DP_SIGMA_DECAY_EPOCH", 1, "The decay epoch of total dp sigma")
    flags.DEFINE_float("MIN_TOTAL_DP_SIGMA", 1.0, "The minnimum of total dp sigma")
    flags.DEFINE_float("TOTAL_FINETUNE_DP_SIGMA", 1.0, "The sigma of total perturbation")
    flags.DEFINE_float("TOTAL_FINETUNE_DP_EPSILON", 0.1, "The epsilon of total perturbation")
    flags.DEFINE_float("TOTAL_FINETUNE_DP_DELTA", 1e-5, "The delta of total perturbation")
    flags.DEFINE_float("TOTAL_FINETUNE_DP_SIGMA_DECAY_RATE", 0.99, "The decay rate of total dp sigma")
    flags.DEFINE_integer("TOTAL_FINETUNE_DP_SIGMA_DECAY_EPOCH", 1, "The decay epoch of total dp sigma")
    flags.DEFINE_float("MIN_TOTAL_FINETUNE_DP_SIGMA", 1.0, "The minnimum of total dp sigma")

    



# Init tensorboard summary writer
def init_writer(LOG_DIR, graph):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    else:
        files = os.listdir(LOG_DIR)
        for log_file in files:
            if log_file.startswith("events"):
                file_path = os.path.join(LOG_DIR, log_file)
                os.remove(file_path)
    writer = tf.summary.FileWriter(LOG_DIR, graph=graph)
    #writer.add_graph(graph)# tensorboard
    return writer

def change_coef(last_value, change_rate, change_itr=None, change_type="STEP"):
    if change_type == "STEP":
        return last_value * change_rate
    elif change_type == "EXP":
        return last_value * np.exp(change_rate)
    elif change_type == "TIME":
        assert change_itr != None
        init_value = last_value * (1.0 + change_rate * (change_itr-1))
        return init_value / (1.0 + change_rate * change_itr)

def _one_hot_encode(inputs, encoded_size):
    def get_one_hot(number):
        on_hot=[0]*encoded_size
        on_hot[int(number)]=1
        return on_hot
    #return list(map(get_one_hot, inputs))
    if isinstance(inputs, list):
        return list(map(get_one_hot, inputs))
    else:
        return get_one_hot(inputs)