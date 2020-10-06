from dependency import *
import os

def set_flags():
    flags.DEFINE_bool("train", False, "Train and save the ATN model.")
    flags.DEFINE_bool("local", False, "Run this model locally or on the cloud")
    flags.DEFINE_string("LBL_NAME", "Labeler", "The name of Label part")
    flags.DEFINE_string("ENC_NAME", "Encoder", "The name of Encoder part")
    flags.DEFINE_string("EMB_NAME", "Embedder", "The name of Embedder part")
    flags.DEFINE_string("DEC_NAME", "Decoder", "The name of Decoder part")
    flags.DEFINE_integer("CENTRAL_CHANNEL_SIZE", 128, "The size of the central channel")
    flags.DEFINE_integer("NUM_ENC_RES_BLOCK", 5, "The num of residual block")
    flags.DEFINE_integer("ENC_RES_BLOCK_SIZE", 3, "The num of layers in each block")
    flags.DEFINE_integer("NUM_DEC_RES_BLOCK", 5, "The num of residual block")
    flags.DEFINE_integer("DEC_RES_BLOCK_SIZE", 3, "The num of layers in each block")
    flags.DEFINE_string("ATT_TYPE", "GOOGLE", "TRAD or GOOGLE")
    flags.DEFINE_integer("EMB_SIZE", 512, "The embedding size")
    flags.DEFINE_integer("G_EMB_SIZE", 512, "The embedding size")
    flags.DEFINE_integer("D_EMB_SIZE", 512, "The embedding size")
    flags.DEFINE_integer("EMB_TOPK", 3, "The top k smallest")
    flags.DEFINE_integer("EMB_SUBK", 8, "The sub k smallest")
    flags.DEFINE_string("EMB_TYPE", "MINIMUM", "SOFTMAX or MINIMUM")
    # Path
    flags.DEFINE_string("RESNET18_PATH", "./models/target_classifier/resnet18", "Path of Resnet18")
    flags.DEFINE_string("RESNET50_PATH", "./models/target_classifier/resnet_v2_50", "Path of Resnet50")
    flags.DEFINE_string("AE_PATH", "./models/AE", "Path of AAN")
    flags.DEFINE_string("CNN_PATH", "./models/target_classifier/basic_CNN", "Path of CNN")
    flags.DEFINE_string("TRAIN_LOG_PATH", "./graphs/train", "Path of log for training")
    flags.DEFINE_string("VALID_LOG_PATH", "./graphs/valid", "Path of log for validation")
    flags.DEFINE_string("TEST_LOG_PATH", "./graphs/test", "Path of log for testing")
    flags.DEFINE_string("DATA_DIR", "/Users/mashiru/Life/My-Emory/Research/Research-Project/Data/tiny-imagenet-200", "Data dir")
    # Data description
    flags.DEFINE_bool("NORMALIZE", True, "Data is normalized to [0, 1]")
    flags.DEFINE_bool("BIASED", False, "Data is shifted to [-1, 1]")
    flags.DEFINE_integer("NUM_CLASSES", 1000, "Number of classification classes")
    flags.DEFINE_integer("LBL_STATES_SIZE", 256, "Size of label states")
    flags.DEFINE_integer("IMAGE_ROWS", 224, "Input row dimension")
    flags.DEFINE_integer("IMAGE_COLS", 224, "Input column dimension")
    flags.DEFINE_integer("NUM_CHANNELS", 3, "Input depth dimension")
    # Training params
    flags.DEFINE_integer("NUM_EPOCHS", 1, "Number of epochs") # 200
    flags.DEFINE_integer("NUM_PRE_EPOCHS", 2, "Number of epochs") # 200
    flags.DEFINE_integer("NUM_ACCUM_ITERS", 2, "Number of accumulation") # 2
    flags.DEFINE_integer("BATCH_SIZE", 128, "Size of training batches")# 128
    flags.DEFINE_integer("EVAL_FREQUENCY", 1, "Frequency for evaluation") # 25
    flags.DEFINE_integer("PRE_EVAL_FREQUENCY", 1, "Frequency for pre evaluation") # 25
    flags.DEFINE_bool("load_AE", False, "Load AE from the last training result or not")
    flags.DEFINE_bool("train_label", True, "Train and get label states, save into ckpt")
    flags.DEFINE_bool("early_stopping", False, "Use early stopping or not")
    flags.DEFINE_integer("EARLY_STOPPING_THRESHOLD", 10, "Early stopping threshold")
    # AE type
    flags.DEFINE_string("AE_TYPE", "TRAD", "The type of Autoencoder") # SPARSE, VARI(ATIONAL), TRAD(ITIONAL), ATTEN(TIVE)
    flags.DEFINE_bool("SPARSE", False, "The type of Autoencoder") # SPARSE, VARI(ATIONAL), TRAD(ITIONAL), ATTEN(TIVE)
    flags.DEFINE_bool("VARI", False, "The type of Autoencoder") # SPARSE, VARI(ATIONAL), TRAD(ITIONAL), ATTEN(TIVE)
    # AE params
    flags.DEFINE_integer("BOTTLENECK", 2048, "The size of bottleneck")
    flags.DEFINE_bool("USE_LABEL_MASK", False, "Whether use label mask or not")
    # Loss params
    flags.DEFINE_string("LOSS_MODE_TRANS", "C_W2", "How to calculate loss from fake imgae") # ENTRO, C_W
    flags.DEFINE_string("LOSS_MODE_FAKE", "C_W", "How to calculate loss from fake imgae") # LOGITS, PREDS, ENTRO, C_W
    flags.DEFINE_string("LOSS_MODE_CLEAN", "C_W2", "How to calculate loss from clean image") # LOGITS, PREDS, ENTRO, C_W
    ##
    flags.DEFINE_string("NORM_TYPE", "L2", "The norm type") # INF, L2, L1
    flags.DEFINE_float('REG_SCALE', 0.01, 'The scale of regularization')
    ## Gamma for rho distance
    flags.DEFINE_float("SPARSE_RHO", 10, "The sparse threshold for central states of AE")
    flags.DEFINE_float("GAMMA_S", 1e-5, "Coefficient for RHO distance") # 0.01
    ## Gamma for variational kl distance
    flags.DEFINE_float("GAMMA_V", 1e-4, "Coefficient for KL distance") # 1e-3
    ## Gamma for reconstruction loss
    flags.DEFINE_float("GAMMA_R", 1e-2, "Coefficient for reconstruction loss")
    ## Gamma for label loss
    flags.DEFINE_float("GAMMA_L", 1.0, "Coefficient for label loss")
    ## Gamma for label loss
    flags.DEFINE_float("GAMMA_PRE_L", 1.0, "Coefficient for label pre traini loss")
    ## loss x
    flags.DEFINE_string("PARTIAL_LOSS", "FULL_LOSS", "Use loss x or loss y") # FULL_LOSS, LOSS_X, LOSS_Y
    flags.DEFINE_integer("LOSS_CHANGE_FREQUENCY", 5, "The frequency of changing loss") # 5
    flags.DEFINE_float("LOSS_X_THRESHOLD", 500, "Threshold for loss x") # 500
    flags.DEFINE_float("LOSS_X_THRE_CHANGE_RATE", 0.8, "Change rate of Loss x threshold") # 0.8
    flags.DEFINE_float("LOSS_X_THRE_CHANGE_EPOCHS", 10, "Change epochs of Loss x threshold") # 10
    flags.DEFINE_float("MIN_LOSS_X_THRE", 50, "Minimum threshold for loss x") # 50
    flags.DEFINE_float("MAX_LOSS_X_THRE", 1500, "Maximum threshold for loss x") # 150
    ## Loss Y
    flags.DEFINE_float("LOSS_Y_LOW_BOUND_F", -20.0, "The lower bound of loss y for fake")
    flags.DEFINE_float("LOSS_Y_UP_BOUND_F", 500.0, "The up bound of loss y for fake")
    flags.DEFINE_float("LOSS_Y_LOW_BOUND_T", -20.0, "The lower bound of loss y for trans")
    flags.DEFINE_float("LOSS_Y_UP_BOUND_T", 500.0, "The up bound of loss y for trans")
    flags.DEFINE_float("LOSS_Y_LOW_BOUND_C", -20.0, "The lower bound of loss y for clean")
    flags.DEFINE_float("LOSS_Y_UP_BOUND_C", 500.0, "The up bound of loss y for clean")
    ## Kappa: C_W loss
    flags.DEFINE_integer("MODIFY_KAPPA_THRESHOLD", 5, "Modify beta y threshold")
    flags.DEFINE_float("KAPPA_FOR_TRANS", 8, "The min logits distance") # 8
    flags.DEFINE_float("KAPPA_TRANS_CHANGE_RATE", 1.2, "The change rate of KAPPA TRANS")
    flags.DEFINE_float("KAPPA_FOR_FAKE", 8, "The min logits distance") # 8
    flags.DEFINE_float("KAPPA_FAKE_CHANGE_RATE", 1.2, "The change rate of KAPPA FAKE")
    flags.DEFINE_float("KAPPA_FOR_CLEAN", 8, "The min logits distance") # 8
    flags.DEFINE_float("KAPPA_CLEAN_CHANGE_RATE", 1.2, "The change rate of KAPPA CLEAN")
    ## Beta x true
    flags.DEFINE_float("BETA_X_TRUE", 0.1, "Coefficient for loss of X") # 1
    flags.DEFINE_string("BETA_X_TRUE_CHANGE_TYPE", "STEP", "Change type of Beta x") # STEP, EXP, TIME
    flags.DEFINE_float("BETA_X_TRUE_CHANGE_RATE", 1.2, "Change rate of Beta x") # 1.2
    flags.DEFINE_float("BETA_X_TRUE_CHANGE_EPOCHS", 10, "Change epochs of Beta x") # 10
    flags.DEFINE_float("MIN_BETA_X_TRUE", 0.01, "Minimum of beta x") # 0.01
    flags.DEFINE_float("MAX_BETA_X_TRUE", 1, "Maximum of beta x") # 1
    ## Beta x fake
    flags.DEFINE_float("BETA_X_FAKE", 0.1, "Coefficient for loss of X") # 1
    flags.DEFINE_string("BETA_X_FAKE_CHANGE_TYPE", "STEP", "Change type of Beta x") # STEP, EXP, TIME
    flags.DEFINE_float("BETA_X_FAKE_CHANGE_RATE", 1.2, "Change rate of Beta x") # 1.2
    flags.DEFINE_float("BETA_X_FAKE_CHANGE_EPOCHS", 10, "Change epochs of Beta x") # 10
    flags.DEFINE_float("MIN_BETA_X_FAKE", 0.01, "Minimum of beta x") # 0.01
    flags.DEFINE_float("MAX_BETA_X_FAKE", 1, "Maximum of beta x") # 1    
    ## Beta y TRANS
    flags.DEFINE_float("BETA_Y_TRANS", 1, "Coefficient for loss of Y TRANS") # 5
    flags.DEFINE_string("BETA_Y_TRANS_CHANGE_TYPE", "STEP", "Change type of Beta Y TRANS") # STEP, EXP, TIME
    flags.DEFINE_float("BETA_Y_TRANS_CHANGE_RATE", 1.1, "Change rate of Beta Y TRANS") # 1
    flags.DEFINE_float("BETA_Y_TRANS_CHANGE_EPOCHS", 50, "Change epochs of Beta Y TRANS") # 50
    flags.DEFINE_float("MIN_BETA_Y_TRANS", 1, "Minimum of beta Y TRANS") # 50
    flags.DEFINE_float("MAX_BETA_Y_TRANS", 100, "Maximum of beta Y TRANS") # 100
    ## Beta y fake
    flags.DEFINE_float("BETA_Y_FAKE", 0, "Coefficient for loss of Y FAKE") # 0
    flags.DEFINE_string("BETA_Y_FAKE_CHANGE_TYPE", "STEP", "Change type of Beta Y FAKE") # STEP, EXP, TIME
    flags.DEFINE_float("BETA_Y_FAKE_CHANGE_RATE", 1.2, "Change rate of Beta Y FAKE") # 1.2
    flags.DEFINE_float("BETA_Y_FAKE_CHANGE_EPOCHS", 10, "Change epochs of Beta Y FAKE") # 10
    flags.DEFINE_float("MIN_BETA_Y_FAKE", 0, "Minimum of beta Y FAKE") # 50
    flags.DEFINE_float("MAX_BETA_Y_FAKE", 0, "Maximum of beta Y FAKE") # 120
    ## Beta y clean
    flags.DEFINE_float("BETA_Y_CLEAN", 1, "Coefficient for loss of Y CLEAN") # 1
    flags.DEFINE_string("BETA_Y_CLEAN_CHANGE_TYPE", "STEP", "Change type of Beta Y CLEAN") # STEP, EXP, TIME
    flags.DEFINE_float("BETA_Y_CLEAN_CHANGE_RATE", 1.1, "Change rate of Beta Y CLEAN") # 1
    flags.DEFINE_float("BETA_Y_CLEAN_CHANGE_EPOCHS", 50, "Change epochs of Beta Y CLEAN") # 50
    flags.DEFINE_float("MIN_BETA_Y_CLEAN", 1, "Minimum of beta Y CLEAN") # 1
    flags.DEFINE_float("MAX_BETA_Y_CLEAN", 100, "Maximum of beta Y CLEAN") # 1
    # Optimization params
    flags.DEFINE_string("OPT_TYPE", "ADAM", "The type of optimization") # ADAM, MOME, NEST
    flags.DEFINE_string("ENC_NORM", "NONE", "Use batch normalization or layer normalization or none")
    flags.DEFINE_string("ENC_OUT_NORM", "NONE", "Use batch normalization or layer normalization or none")
    flags.DEFINE_string("EMB_NORM", "LAYER", "Use batch normalization or layer normalization or none")
    flags.DEFINE_string("DEC_IN_NORM", "NONE", "Use batch normalization or layer normalization or none")
    flags.DEFINE_string("DEC_NORM", "NONE", "Use batch normalization or layer normalization or none")
    flags.DEFINE_float("BATCH_MOME", 0.99, "Momentum for the moving average")
    flags.DEFINE_float("BATCH_EPSILON", 0.001, "Small float added to variance to avoid dividing by zero")
    flags.DEFINE_float("LEARNING_RATE", 1e-4, "Learning rate of optimization")
    flags.DEFINE_float("LEARNING_DECAY_RATE", 0.99, "Decay rate of learning rate")
    flags.DEFINE_integer("LEARNING_DECAY_STEPS", int(2.5*1e3), "Decay steps of learning rate")
    flags.DEFINE_bool("IS_GRAD_CLIPPING", False, "Use gradient clipping or not")
    flags.DEFINE_float("GRAD_CLIPPING_NORM", 10.0, "Gradient clipping norm")
    # Non-linear func params
    flags.DEFINE_float("PIXEL_BOUND", 0.01, "Bound for pixel distance") # 0.01
    flags.DEFINE_string("BOUND_CHANGE_TYPE", "step", "Bound change type") # STEP
    flags.DEFINE_float("BOUND_CHANGE_RATE", 0.8, "Bound change rate") # 0.8
    flags.DEFINE_float("BOUND_CHANGE_EPOCHS", 2, "Num of epochs per bound change") # 2
    flags.DEFINE_float("MIN_BOUND", 0.001, "Minimum bound for pixel distance") # 4
    flags.DEFINE_float("MAX_BOUND", 1, "Maximum bound for pixel distance") # 128
    # Attack params
    flags.DEFINE_float("EPSILON", 1, "Epsilon for fgm attack") # 128
    flags.DEFINE_float("EPSILON_CHANGE_RATE", 0.8, "Epsilon change rate") # 128
    flags.DEFINE_float("EPSILON_CHANGE_EPOCHS", 100, "Num of epochs for epsilon change") # 100
    flags.DEFINE_float("MIN_EPSILON", 0.1, "Minimum epsilon for fgm attack") # 4
    flags.DEFINE_float("MAX_EPSILON", 128, "Maximum epsilon for fgm attack") # 128
    flags.DEFINE_integer("FGM_ITERS", 1, "Iteration for fgm attack") # 1
    


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

def change_coef(init_value, change_rate, change_itr, change_type="STEP"):
    if change_type == "STEP":
        return init_value * change_rate ** change_itr
    elif change_type == "EXP":
        return init_value * np.exp(-1.0 * change_rate * change_itr)
    elif change_type == "TIME":
        return init_value / (1.0 + change_rate * change_itr)