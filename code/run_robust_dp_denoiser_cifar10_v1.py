import nn.robust_dp_denoiser_cifar10_v1 as model_cifar10
import os, math
from PIL import Image
from dependency import *
from tensornets.preprocess import keras_resnet_preprocess
import utils.model_utils_cifar10 as  model_utils
from utils.data_utils_cifar10 import dataset
from robust.randomized_smoothing import Smooth
import differential_privacy.utils as dp_utils
import differential_privacy.privacy_accountant.tf.accountant_v2 as accountant
import differential_privacy.dp_sgd.dp_optimizer.sanitizer as sanitizer
# attacks
from cleverhans.attacks import BasicIterativeMethod, FastGradientMethod, MadryEtAl, MomentumIterativeMethod
from cleverhans.attacks_tf import fgm, fgsm
from cleverhans.model import CallableModelWrapper, CustomCallableModelWrapper

model_utils.set_flags()

data = dataset()
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_INDEX




def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test_info(sess, model, is_valid, graph_dict, dp_info, log_file, total_batch=None):
    # vars
    enc_bott_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.enc_bott_scope.name)
    enc_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.enc_scope.name)
    dec_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.dec_scope.name)
    dec_top_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.dec_top_scope.name)

    model_clean_accuracy = model.clean_accuracy
    model_recon_accuracy = model.recon_accuracy
    
    # loss
    model_loss = model.loss(graph_dict["beta_holder"])
    model_loss_clean = model.loss_clean(graph_dict["beta_holder"])

    # reg
    model_reg_enc_bott = model.loss_reg(enc_bott_opt_vars)
    model_reg_enc = model.loss_reg(enc_opt_vars)
    model_reg_dec = model.loss_reg(dec_opt_vars)
    model_reg_dec_top = model.loss_reg(dec_top_opt_vars)

    model_regs = [model_reg_enc_bott, model_reg_enc, model_reg_dec, model_reg_dec_top]
    
    input_sigma = dp_info["input_sigma"]
    

    full_data = False
    
    batch_size = FLAGS.BATCH_SIZE
    if total_batch is None:
        total_batch = int(data.test_size/batch_size)
        full_data = True
    else: total_batch = total_batch

    
    loss = 0.0
    regs = [0.0] *len(model_regs)
        
    clean_acc = 0.0
    recon_acc = 0.0
    recon_clean_acc = 0.0

    data.shuffle_test()
    x_ = data.x_test
    y_ = data.y_test
    
    for b_idx in range(total_batch):
        batch_xs = x_[b_idx*batch_size:(b_idx+1)*batch_size]
        batch_ys = y_[b_idx*batch_size:(b_idx+1)*batch_size]
        #noise = np.random.normal(loc=0.0, scale=input_sigma, size=batch_xs.shape)
        noise = np.random.normal(loc=0.0, scale=FLAGS.INFER_INPUT_SIGMA, size=batch_xs.shape)
        feed_dict = {
            graph_dict["data_holder"]: batch_xs/255.0,
            graph_dict["noised_data_holder"]: np.clip(batch_xs/255.0+noise, 0.0, 1.0),
            graph_dict["label_holder"]: batch_ys,
            graph_dict["beta_holder"]: FLAGS.BETA,
            graph_dict["is_training"]: False
        }
        '''
        fetches = [model.resized_data, model.resized_recon]
        data_infer, recon_infer = sess.run(fetches=fetches, feed_dict=feed_dict)
        feed_dict[graph_dict["data_infer_holder"]] = keras_resnet_preprocess(data_infer)
        feed_dict[graph_dict["recon_infer_holder"]] = keras_resnet_preprocess(recon_infer)
        '''
        fetches = [model_loss, model_regs, model_clean_accuracy, model_recon_accuracy, model.recon_clean_accuracy]
        batch_loss, batch_regs, batch_clean_acc, batch_recon_acc, batch_recon_clean_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        
        loss += batch_loss
        clean_acc += batch_clean_acc
        recon_acc += batch_recon_acc
        recon_clean_acc += batch_recon_clean_acc
        for r_idx in range(len(regs)):
            regs[r_idx] += batch_regs[r_idx]
        
    loss /= total_batch
    for r_idx in range(len(regs)):
        regs[r_idx] /= total_batch
    clean_acc /= total_batch
    recon_acc /= total_batch
    recon_clean_acc /= total_batch


    # Print info
    print("Loss: {:.4f}".format(loss))
    for idx in range(len(regs)):
        print("Reg loss for {}: {}".format(idx, regs[idx]))
    print("Beta: {:.4f}".format(FLAGS.BETA))
    print("Clean Acc: {:.4f}, Recon Acc: {:.4f}, Recon Clean Acc: {:.4f}".format(clean_acc, recon_acc, recon_clean_acc))
    print("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}".format(
        dp_info["eps"], dp_info["delta"], dp_info["total_sigma"], FLAGS.INFER_INPUT_SIGMA))
    
    with open(log_file, "a+") as file: 
        if full_data:
            file.write("FULL DATA\n")
        file.write("Loss: {:.4f}\n".format(loss))
        for idx in range(len(regs)):
            file.write("Reg loss for {}: {}\n".format(idx, regs[idx]))
        file.write("Beta: {:.4f}".format(FLAGS.BETA))
        file.write("Clean Acc: {:.4f}, Recon Acc: {:.4f}, Recon Clean Acc: {:.4f}\n".format(clean_acc, recon_acc, recon_clean_acc))
        file.write("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}\n".format(
        dp_info["eps"], dp_info["delta"], dp_info["total_sigma"], FLAGS.INFER_INPUT_SIGMA))
        file.write("---------------------------------------------------\n")
    
    res_dict = {"clean_acc": clean_acc,
                "recon_acc": recon_acc
                }
    return res_dict


def robust_info(sess, model, recon_outputs, graph_dict, log_file):
    data_holder = graph_dict["data_holder"]
    noised_data_holder = graph_dict["noised_data_holder"]
    data_infer_holder = graph_dict["data_infer_holder"]
    recon_infer_holder = graph_dict["recon_infer_holder"]
    label_holder = graph_dict["label_holder"]
    sgd_sigma_holder = graph_dict["sgd_sigma_holder"]
    act_sigma_holder = graph_dict["act_sigma_holder"]
    beta_holder = graph_dict["beta_holder"]
    is_training = graph_dict["is_training"]

    batch_size = FLAGS.BATCH_SIZE

    if not FLAGS.LOAD_ADVS:
        # generate adversarial examples
        data.shuffle_test()
        x_ = data.x_test
        y_ = data.y_test
        #total_batch = int(data.test_size/batch_size)
        total_batch = 100

        ys = []; xs = []; adv_fgsm = []; adv_ifgsm = []; adv_mim = []; adv_madry = []
        for idx in range(total_batch):
            batch_xs = x_[idx*batch_size:(idx+1)*batch_size]
            batch_ys = y_[idx*batch_size:(idx+1)*batch_size]
            batch_xs /= 255.0
            ys.append(batch_ys)
            xs.append(batch_xs)
            adv_fgsm.append(sess.run(graph_dict["x_advs"]["fgsm"], feed_dict = {
                                graph_dict["data_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))
            adv_ifgsm.append(sess.run(graph_dict["x_advs"]["ifgsm"], feed_dict = {
                                graph_dict["data_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))
            adv_mim.append(sess.run(graph_dict["x_advs"]["mim"], feed_dict = {
                                graph_dict["data_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))
            adv_madry.append(sess.run(graph_dict["x_advs"]["madry"], feed_dict = {
                                graph_dict["data_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))

        ys = np.concatenate(ys, axis=0)
        advs = {}
        advs["clean"] = np.concatenate(xs, axis=0)
        advs["fgsm"] = np.concatenate(adv_fgsm, axis=0)
        advs["ifgsm"] = np.concatenate(adv_ifgsm, axis=0)
        advs["mim"] = np.concatenate(adv_mim, axis=0)
        advs["madry"] = np.concatenate(adv_madry, axis=0)
        advs["ys"] = ys
        np.save("adv_examples.npy", advs)
    else:
        advs = np.load("adv_examples.npy", allow_pickle=True).item()
        ys = advs["ys"]

    is_acc = {
        "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
    }
    is_robust = {
        "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
    }
    
    robust_pred = {
        "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
    }
    robust_radius = {
        "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
    }
    adv_acc = {}; adv_robust = {}; adv_robust_acc = {}
    # Randomized Smoothing
    smoother = Smooth(recon_outputs[1], noised_data_holder, is_training, FLAGS.NUM_CLASSES, FLAGS.INFER_INPUT_SIGMA)

    print("Robust defense with input std: {}".format(FLAGS.INFER_INPUT_SIGMA))
    for key, adv in advs.items():
        for x in adv:
            prediction, radius = smoother.certify(sess, x, FLAGS.NUM_SAMPLING_0, FLAGS.NUM_SAMPLING, FLAGS.ROBUST_ALPHA, FLAGS.BATCH_SIZE)
            robust_pred[key].append(prediction)
            robust_radius[key].append(radius)
        adv_acc[key] = np.sum(np.array(robust_pred[key])==ys.argmax(axis=1))*1.0/adv.shape[0]
        adv_robust[key] = np.sum(np.array(robust_radius[key])>=FLAGS.ATTACK_NORM_BOUND)*1.0/adv.shape[0]
        adv_robust_acc[key] = np.sum(
                np.logical_and(
                    np.array(robust_pred[key])==ys.argmax(axis=1), 
                    np.array(robust_radius[key])>=FLAGS.ATTACK_NORM_BOUND
                )
            )*1.0/adv.shape[0]

        # Print info
        print("{}:".format(key))
        print("accuracy: {}, robustness: {}, robust_accuracy: {}".format(
            adv_acc[key], adv_robust[key], adv_robust_acc[key]))
        print("avg. bound is {} at sigma = {}\n".format(np.mean(robust_radius[key]), FLAGS.INFER_INPUT_SIGMA))
        print()
        print()
    
        with open("{}.std{}".format(log_file, FLAGS.INFER_INPUT_SIGMA), "a+") as file: 
            file.write("{}:\n".format(key))
            file.write("accuracy: {}, robustness: {}, robust_accuracy: {}\n".format(
                adv_acc[key], adv_robust[key], adv_robust_acc[key]))
            file.write("avg. bound is {} at sigma = {}\n".format(np.mean(robust_radius[key]), FLAGS.INFER_INPUT_SIGMA))
            file.write("===================================================\n")
    
    res_dict = {"adv_acc": adv_acc, 
                "adv_robust": adv_robust,
                "adv_robust_acc": adv_robust_acc
                }
    return res_dict



def test():
    """
    """
    input_sigma = FLAGS.INPUT_SIGMA
    total_dp_sigma = FLAGS.TOTAL_DP_SIGMA
    total_dp_delta = FLAGS.TOTAL_DP_DELTA
    total_dp_epsilon = FLAGS.TOTAL_DP_EPSILON

    batch_size = FLAGS.BATCH_SIZE
    tf.reset_default_graph()
    g = tf.get_default_graph()
    # attack_target = 8
    with g.as_default():
        # Placeholder nodes.
        data_holder = tf.placeholder(tf.float32, [batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        noised_data_holder = tf.placeholder(tf.float32, [batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        data_infer_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, FLAGS.NUM_CHANNELS])
        recon_infer_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [batch_size, FLAGS.NUM_CLASSES])
        sgd_sigma_holder = tf.placeholder(tf.float32, ())
        act_sigma_holder = tf.placeholder(tf.float32, ())
        beta_holder = tf.placeholder(tf.float32, ())
        dp_grad_clipping_norm_holder = tf.placeholder(tf.float32, ())
        dp_grad_clipping_norm_holder_1 = tf.placeholder(tf.float32, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = model_cifar10.DP_DENOISER(data=data_holder, label=label_holder, is_training=is_training, noised_data=noised_data_holder,
                                        data_infer=data_infer_holder, recon_infer=recon_infer_holder)


        
        data_recon, recon_outputs, _ = model(noised_data_holder, data_holder)

        model_clean_accuracy = model.clean_accuracy
        model_recon_accuracy = model.recon_accuracy
        # vars
        enc_bott_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.enc_bott_scope.name)
        enc_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.enc_scope.name)
        dec_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.dec_scope.name)
        dec_top_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.dec_top_scope.name)


        # loss
        model_loss = model.loss(beta_holder)
        model_loss_clean = model.loss_clean(beta_holder)
        # reg
        model_reg_enc_bott = model.loss_reg(enc_bott_opt_vars)
        model_reg_enc = model.loss_reg(enc_opt_vars)
        model_reg_dec = model.loss_reg(dec_opt_vars)
        model_reg_dec_top = model.loss_reg(dec_top_opt_vars)

        model_regs = [model_reg_enc_bott, model_reg_enc, model_reg_dec, model_reg_dec_top]

        # robust
        def inference(x): 
            logits, _, _, _ = model.prediction(x, True)
            return logits
        def inference_prob(x):
            _, probs, _, _ = model.prediction(x, True)
            return probs
        

        graph_dict = {}
        graph_dict["data_holder"] = data_holder
        graph_dict["noised_data_holder"] = noised_data_holder
        graph_dict["data_infer_holder"] = data_infer_holder
        graph_dict["recon_infer_holder"] = recon_infer_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["sgd_sigma_holder"] = sgd_sigma_holder
        graph_dict["act_sigma_holder"] = act_sigma_holder
        graph_dict["beta_holder"] = beta_holder
        graph_dict["is_training"] = is_training
        

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # load model
        model.tf_load(sess, name=FLAGS.CNN_CKPT_RESTORE_NAME)
        model.tf_load_classifier(sess, name=FLAGS.PRETRAINED_CNN_CKPT_RESTORE_NAME)

        
        # adv test
        ####################################################################################################
        x_advs = {}
        ch_model_logits = CallableModelWrapper(callable_fn=inference, output_layer='logits')
        ch_model_probs = CallableModelWrapper(callable_fn=inference_prob, output_layer='probs')
        # FastGradientMethod
        fgsm_obj = FastGradientMethod(model=ch_model_probs, sess=sess)
        x_advs["fgsm"] = fgsm_obj.generate(x=data_holder, ord=2,
            eps=FLAGS.ATTACK_NORM_BOUND, clip_min=0.0, clip_max=1.0) # testing now

        # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
        # default: eps_iter=0.05, nb_iter=10
        ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
        x_advs["ifgsm"] = ifgsm_obj.generate(x=data_holder, ord=2,
            eps=FLAGS.ATTACK_NORM_BOUND, eps_iter=FLAGS.ATTACK_NORM_BOUND/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
        
        # MomentumIterativeMethod
        # default: eps_iter=0.06, nb_iter=10
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        x_advs["mim"] = mim_obj.generate(x=data_holder, ord=2,
            eps=FLAGS.ATTACK_NORM_BOUND, eps_iter=FLAGS.ATTACK_NORM_BOUND/10, nb_iter=10, decay_factor=1.0, clip_min=0.0, clip_max=1.0)

        # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
        # default: eps_iter=0.01, nb_iter=40
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        x_advs["madry"] = madry_obj.generate(x=data_holder, ord=2,
            eps=FLAGS.ATTACK_NORM_BOUND, eps_iter=FLAGS.ATTACK_NORM_BOUND/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
        graph_dict["x_advs"] = x_advs
        ####################################################################################################

        # tensorboard writer
        #test_writer = model_utils.init_writer(FLAGS.TEST_LOG_PATH, g)
        print("\nTest")
        if FLAGS.local:
            total_test_batch = 2
        else:
            total_test_batch = None
        dp_info = np.load(FLAGS.DP_INFO_NPY, allow_pickle=True).item()
        test_info(sess, model, True, graph_dict, dp_info, FLAGS.TEST_LOG_FILENAME, 
            total_batch=total_test_batch)
        robust_info(sess, model, recon_outputs, graph_dict, FLAGS.ROBUST_LOG_FILENAME)



def __compute_S_min(M, param_size):
    ss = np.linalg.svd(M, compute_uv=False)
    #import pdb; pdb.set_trace()
    #S_min = min(np.sqrt(FLAGS.MAX_ITERATIONS*ss[:, -1]))
    S_min = np.sqrt(FLAGS.MAX_ITERATIONS*ss[-1])
    return S_min

def __compute_S_min_from_M(M):
    param_size = M.shape[0]
    S_min = __compute_S_min(M, param_size)

    return S_min

def compute_S_min_from_M(M, is_layerwised=True):
    if is_layerwised:
        batch_S_min_layerwised = []
        for M_ in M:
            batch_S_min_layerwised.append(__compute_S_min_from_M(M_))
        batch_S_min_layerwised = np.stack(batch_S_min_layerwised, axis=1)
        return batch_S_min_layerwised
    else:
        return __compute_S_min_from_M(M)

def cal_sigmas(lot_M, input_sigma, clipping_norm):
    lot_M = sum(lot_M) / (FLAGS.BATCHES_PER_LOT**2)
    lot_S_min = compute_S_min_from_M(lot_M, FLAGS.IS_MGM_LAYERWISED)/clipping_norm
    #import pdb; pdb.set_trace()
    min_S_min = lot_S_min
    sigma_trans = input_sigma * min_S_min
    
    if sigma_trans >= FLAGS.TOTAL_DP_SIGMA:
        sgd_sigma = 0.0
    else: 
        sgd_sigma = FLAGS.TOTAL_DP_SIGMA - sigma_trans
        sigma_trans = FLAGS.TOTAL_DP_SIGMA
    return min_S_min, sgd_sigma, sigma_trans



def train():
    """
    """
    import time
    input_sigma = FLAGS.INPUT_SIGMA
    total_dp_sigma = FLAGS.TOTAL_DP_SIGMA
    total_dp_delta = FLAGS.TOTAL_DP_DELTA
    total_dp_epsilon = FLAGS.TOTAL_DP_EPSILON

    batch_size = FLAGS.BATCH_SIZE
    tf.reset_default_graph()
    g = tf.get_default_graph()
    # attack_target = 8
    with g.as_default():
        # Placeholder nodes.
        data_holder = tf.placeholder(tf.float32, [batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        noised_data_holder = tf.placeholder(tf.float32, [batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        data_infer_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, FLAGS.NUM_CHANNELS])
        recon_infer_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [batch_size, FLAGS.NUM_CLASSES])
        sgd_sigma_holder = tf.placeholder(tf.float32, ())
        act_sigma_holder = tf.placeholder(tf.float32, ())
        beta_holder = tf.placeholder(tf.float32, ())
        dp_grad_clipping_norm_holder = tf.placeholder(tf.float32, ())
        dp_grad_clipping_norm_holder_1 = tf.placeholder(tf.float32, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = model_cifar10.DP_DENOISER(data=data_holder, label=label_holder, is_training=is_training, noised_data=noised_data_holder,
                                        data_infer=data_infer_holder, recon_infer=recon_infer_holder)
        priv_accountant = accountant.GaussianMomentsAccountant(data.valid_size)

        data_recon, _, _ = model(noised_data_holder, data_holder)
        model_clean_accuracy = model.clean_accuracy
        model_recon_accuracy = model.recon_accuracy
        
        # vars
        enc_bott_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.enc_bott_scope.name)
        enc_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.enc_scope.name)
        dec_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.dec_scope.name)
        dec_top_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model.dec_top_scope.name)


        # loss
        model_loss = model.loss(beta_holder)
        model_loss_clean = model.loss_clean(beta_holder)
        # reg
        model_reg_enc_bott = model.loss_reg(enc_bott_opt_vars)
        model_reg_enc = model.loss_reg(enc_opt_vars)
        model_reg_dec = model.loss_reg(dec_opt_vars)
        model_reg_dec_top = model.loss_reg(dec_top_opt_vars)

        model_regs = [model_reg_enc_bott, model_reg_enc, model_reg_dec, model_reg_dec_top]
        # training
        # DP-OPT  
        model_op_1, model_lot_op_1, _, model_lr_1 = model.dp_optimization(model_loss+model_reg_enc+model_reg_dec, priv_accountant, sgd_sigma_holder,
                        act_sigma=act_sigma_holder, opt_vars=enc_opt_vars+dec_opt_vars, 
                        learning_rate=FLAGS.LEARNING_RATE_1, lr_decay_steps=FLAGS.LEARNING_DECAY_STEPS_1, lr_decay_rate=FLAGS.LEARNING_DECAY_RATE_1,
                        batches_per_lot=FLAGS.BATCHES_PER_LOT, px_clipping_norm=dp_grad_clipping_norm_holder_1,
                        scope="DP_OPT_1")
        model_zero_op_1, model_accum_op_1, model_avg_op_1 = model_lot_op_1
            
        # Input Perturb  
        model_op_0, model_lot_op_0, _, model_lr_0 = model.dp_optimization(model_loss+model_reg_enc_bott, priv_accountant, sgd_sigma_holder,
                        act_sigma=act_sigma_holder, opt_vars=enc_bott_opt_vars, 
                        learning_rate=FLAGS.LEARNING_RATE_0, lr_decay_steps=FLAGS.LEARNING_DECAY_STEPS_0, lr_decay_rate=FLAGS.LEARNING_DECAY_RATE_0,
                        batches_per_lot=FLAGS.BATCHES_PER_LOT, px_clipping_norm=dp_grad_clipping_norm_holder,
                        scope="DP_OPT_0")
        model_zero_op_0, model_accum_op_0, model_avg_op_0 = model_lot_op_0

        # Input Perturb  
        model_op_2, model_lot_op_2, _, model_lr_2 = model.dp_optimization(model_loss+model_reg_dec_top, priv_accountant, sgd_sigma_holder,
                        act_sigma=act_sigma_holder, opt_vars=dec_top_opt_vars, 
                        learning_rate=FLAGS.LEARNING_RATE_2, lr_decay_steps=FLAGS.LEARNING_DECAY_STEPS_2, lr_decay_rate=FLAGS.LEARNING_DECAY_RATE_2,
                        batches_per_lot=FLAGS.BATCHES_PER_LOT, px_clipping_norm=dp_grad_clipping_norm_holder,
                        scope="DP_OPT_2")
        model_zero_op_2, model_accum_op_2, model_avg_op_2 = model_lot_op_2

        model_lrs = [model_lr_0, model_lr_1, model_lr_2]
                
        # analysis
        model_M_0, _ = model.compute_M_from_input_perturbation(data_holder, model_loss_clean+model_reg_enc_bott, dp_grad_clipping_norm_holder, 
                        var_list=enc_bott_opt_vars, scope="M_0")

        model_M_2, _ = model.compute_M_from_input_perturbation(data_holder, model_loss_clean+model_reg_dec_top, dp_grad_clipping_norm_holder, 
                        var_list=dec_top_opt_vars, scope="M_2")

        


        graph_dict = {}
        graph_dict["data_holder"] = data_holder
        graph_dict["noised_data_holder"] = noised_data_holder
        graph_dict["data_infer_holder"] = data_infer_holder
        graph_dict["recon_infer_holder"] = recon_infer_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["sgd_sigma_holder"] = sgd_sigma_holder
        graph_dict["act_sigma_holder"] = act_sigma_holder
        graph_dict["beta_holder"] = beta_holder
        graph_dict["dp_grad_clipping_norm_holder"] = dp_grad_clipping_norm_holder
        graph_dict["dp_grad_clipping_norm_holder_1"] = dp_grad_clipping_norm_holder_1
        graph_dict["is_training"] = is_training

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        
        
        if FLAGS.load_model:
            model.tf_load(sess, name=FLAGS.CNN_CKPT_RESTORE_NAME)
        
        if FLAGS.local:
            total_train_lot = 2
            total_valid_lot = 2
        else:
            #total_train_lot = int(data.valid_size/batch_size/FLAGS.BATCHES_PER_LOT)
            total_train_lot = int(data.train_size/batch_size/FLAGS.BATCHES_PER_LOT)
            total_valid_lot = None  

        model.tf_load_classifier(sess, name=FLAGS.PRETRAINED_CNN_CKPT_RESTORE_NAME)      
        
        print("Training...")
        itr_count = 0
        itr_start_time = time.time()
        threshold_count_0 = 0
        threshold_count_2 = 0
        update_dp_grad_norm = False
        for epoch in range(FLAGS.NUM_EPOCHS):
            ep_start_time = time.time()
            # Compute A norm
            
            min_S_min = float("inf")

            # shuffle
            data.shuffle_train()
            b_idx = 0

            
            for train_idx in range(total_train_lot):
            #for train_idx in range(1):
                terminate = False
                # dec_top_layers
                lot_feeds = []
                lot_M = []
                if input_sigma < FLAGS.INFER_INPUT_SIGMA:
                    input_sigma *= (1.0 + FLAGS.INFER_INPUT_SIGMA_INC_RATE)
                for _ in range(FLAGS.BATCHES_PER_LOT):
                    #batch_xs = data.x_valid[b_idx*batch_size:(b_idx+1)*batch_size]
                    #batch_ys = data.y_valid[b_idx*batch_size:(b_idx+1)*batch_size]
                    batch_xs = data.x_train[b_idx*batch_size:(b_idx+1)*batch_size]
                    batch_ys = data.y_train[b_idx*batch_size:(b_idx+1)*batch_size]
                    noise = np.random.normal(loc=0.0, scale=input_sigma, size=batch_xs.shape)
                    #import pdb; pdb.set_trace()
                    feed_dict = {
                        data_holder: batch_xs/255.0,
                        noised_data_holder: np.clip(batch_xs/255.0+noise, 0.0, 1.0),
                        label_holder: batch_ys,
                        beta_holder: FLAGS.BETA,
                        dp_grad_clipping_norm_holder: FLAGS.DP_GRAD_CLIPPING_L2NORM,
                        dp_grad_clipping_norm_holder_1: FLAGS.DP_GRAD_CLIPPING_L2NORM_1,
                        is_training: True
                    }

                    batch_M_0 = sess.run(fetches=model_M_0, feed_dict=feed_dict)
                    lot_M.append(batch_M_0)
                    lot_feeds.append(feed_dict)
                    b_idx += 1
                
                min_S_min_0, sgd_sigma_0, act_sigma_0 = cal_sigmas(lot_M, input_sigma, FLAGS.DP_GRAD_CLIPPING_L2NORM)
                # for input transofrmation
                if train_idx % 1 == 0:
                    print("enc_bott_layers:")
                    print("min S_min: ", min_S_min_0)
                    print("Sigma trans: ", act_sigma_0)
                    print("Sigma grads: ", sgd_sigma_0)
                    print("DP grad clipping norm: {}".format(FLAGS.DP_GRAD_CLIPPING_L2NORM))
                    print("DP grad clipping norm 1: {}".format(FLAGS.DP_GRAD_CLIPPING_L2NORM_1))
                    print()

                if sgd_sigma_0 > 1.5:
                    threshold_count_0 += 1
                
                # run op for dec_top_layers
                sess.run(model_zero_op_0)
                for feed_dict in lot_feeds:
                    feed_dict[sgd_sigma_holder] = sgd_sigma_0
                    feed_dict[act_sigma_holder] = act_sigma_0
                    sess.run(fetches=model_accum_op_0, feed_dict=feed_dict)
                sess.run(model_avg_op_0)
                sess.run(model_op_0, feed_dict=feed_dict)

                # run op for enc dec layers
                sess.run(model_zero_op_1)
                for feed_dict in lot_feeds:
                    feed_dict[sgd_sigma_holder] = FLAGS.TOTAL_DP_SIGMA
                    feed_dict[act_sigma_holder] = FLAGS.TOTAL_DP_SIGMA
                    sess.run(fetches=model_accum_op_1, feed_dict=feed_dict)
                sess.run(model_avg_op_1)
                sess.run(model_op_1, feed_dict=feed_dict)

                # enc_bott_layers
                lot_M = []
                for feed_dict in lot_feeds:
                    batch_M_2 = sess.run(fetches=model_M_2, feed_dict=feed_dict)
                    lot_M.append(batch_M_2)

                min_S_min_2, sgd_sigma_2, act_sigma_2 = cal_sigmas(lot_M, input_sigma, FLAGS.DP_GRAD_CLIPPING_L2NORM)
                # for input transofrmation
                if train_idx % 1 == 0:
                    print("dec_top_layers:")
                    print("min S_min: ", min_S_min_2)
                    print("Sigma trans: ", act_sigma_2)
                    print("Sigma grads: ", sgd_sigma_2)
                    print("DP grad clipping norm: {}".format(FLAGS.DP_GRAD_CLIPPING_L2NORM))
                    print("DP grad clipping norm 1: {}".format(FLAGS.DP_GRAD_CLIPPING_L2NORM_1))
                    print()
                
                if sgd_sigma_2 > 1.5:
                    threshold_count_2 += 1


                # run op for dec_top_layers
                sess.run(model_zero_op_2)
                for feed_dict in lot_feeds:
                    feed_dict[sgd_sigma_holder] = sgd_sigma_2
                    feed_dict[act_sigma_holder] = act_sigma_2
                    sess.run(fetches=model_accum_op_2, feed_dict=feed_dict)
                sess.run(model_avg_op_2)
                sess.run(model_op_2, feed_dict=feed_dict)
                

                itr_count += 1
                if itr_count > FLAGS.MAX_ITERATIONS:
                    terminate = True
               
                
                #import pdb; pdb.set_trace()
                spent_eps_delta, selected_moment_orders = priv_accountant.get_privacy_spent(sess, target_eps=[total_dp_epsilon])
                spent_eps_delta = spent_eps_delta[0]
                selected_moment_orders = selected_moment_orders[0]
                if spent_eps_delta.spent_delta > total_dp_delta or spent_eps_delta.spent_eps > total_dp_epsilon:
                    terminate = True

                # Print info
                if train_idx % FLAGS.EVAL_TRAIN_FREQUENCY == (FLAGS.EVAL_TRAIN_FREQUENCY - 1):
                    # optimization
                    fetches = [model_loss, model_regs, model_lrs, model_clean_accuracy, model_recon_accuracy]#, model_lr_bott, model_lr_highway]
                    loss, regs, lrs, clean_acc, recon_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                    
                    print("Epoch: {}".format(epoch))
                    print("Iteration: {}".format(itr_count))
                    #print("Learning rate: {}, Learning rate bott: {}, Learning rate highway: {}".format(lr, lr_bott, lr_highway))
                    for idx in range(len(lrs)):
                        print("Learning rate for {}: {}".format(idx, lrs[idx]))
                    print("Loss: {:.4f}".format(loss))
                    for idx in range(len(regs)):
                        print("Reg loss for {}: {}".format(idx, regs[idx]))
                    print("Beta: {:.4f}".format(FLAGS.BETA))
                    print("Clean Acc: {:.4f}, Recon Acc: {:.4f}".format(clean_acc, recon_acc))
                    print("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}".format(
                        spent_eps_delta.spent_eps, spent_eps_delta.spent_delta, total_dp_sigma, input_sigma))
                    print()
                    #model.tf_save(sess) # save checkpoint

                    with open(FLAGS.TRAIN_LOG_FILENAME, "a+") as file: 
                        file.write("Epoch: {}\n".format(epoch))
                        file.write("Iteration: {}\n".format(itr_count))
                        for idx in range(len(lrs)):
                            file.write("Learning rate for {}: {}\n".format(idx, lrs[idx]))
                        file.write("Loss: {:.4f}\n".format(loss))
                        for idx in range(len(regs)):
                            file.write("Reg loss for {}: {}\n".format(idx, regs[idx]))
                        file.write("Beta: {:.4f}".format(FLAGS.BETA))
                        file.write("Clean Acc: {:.4f}, Recon Acc: {:.4f}\n".format(clean_acc, recon_acc))
                        file.write("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}\n".format(
                            spent_eps_delta.spent_eps, spent_eps_delta.spent_delta, total_dp_sigma, input_sigma))
                        file.write("\n")
                    
                    if threshold_count_0 > 0.5*FLAGS.EVAL_TRAIN_FREQUENCY and threshold_count_2 > 0.5*FLAGS.EVAL_TRAIN_FREQUENCY:
                        FLAGS.DP_GRAD_CLIPPING_L2NORM = FLAGS.DP_GRAD_CLIPPING_L2NORM / 2
                    
                    '''
                    if threshold_count_0 > 0.8*FLAGS.EVAL_TRAIN_FREQUENCY and threshold_count_2 > 0.8*FLAGS.EVAL_TRAIN_FREQUENCY:
                        FLAGS.BETA = FLAGS.BETA * 1.5
                    '''

                    threshold_count_0 = 0
                    threshold_count_2 = 0
                
                if itr_count % FLAGS.EVAL_VALID_FREQUENCY == 0:
                #if train_idx >= 0:
                    end_time = time.time()
                    print('{} iterations completed with time {:.2f} s'.format(itr_count, end_time-itr_start_time))
                    # validation
                    print("\n******************************************************************")
                    print("Epoch {} Validation".format(epoch))
                    dp_info = {
                        "eps": spent_eps_delta.spent_eps,
                        "delta": spent_eps_delta.spent_delta,
                        "total_sigma": total_dp_sigma,
                        "input_sigma": input_sigma
                    }
                    valid_dict = test_info(sess, model, True, graph_dict, dp_info, FLAGS.VALID_LOG_FILENAME, total_batch=100)
                    #np.save(FLAGS.DP_INFO_NPY, dp_info, allow_pickle=True)
                    '''
                    ckpt_name='robust_dp_cnn.epoch{}.vloss{:.6f}.vacc{:.6f}.input_sigma{:.4f}.total_sigma{:.4f}.dp_eps{:.6f}.dp_delta{:.6f}.ckpt'.format(
                            epoch,
                            valid_dict["loss"],
                            valid_dict["acc"],
                            input_sigma, total_dp_sigma,
                            spent_eps_delta.spent_eps,
                            spent_eps_delta.spent_delta
                            )
                    '''
                    #model.tf_save(sess, name=ckpt_name) # extra store

                if terminate:
                    break

                
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch, end_time-ep_start_time))
            # validation
            print("\n******************************************************************")
            print("Epoch {} Validation".format(epoch))
            dp_info = {
                "eps": spent_eps_delta.spent_eps,
                "delta": spent_eps_delta.spent_delta,
                "total_sigma": total_dp_sigma,
                "input_sigma": input_sigma
            }
            valid_dict = test_info(sess, model, True, graph_dict, dp_info, FLAGS.VALID_LOG_FILENAME, total_batch=None)
            np.save(FLAGS.DP_INFO_NPY, dp_info, allow_pickle=True)
            ckpt_name='robust_dp_cnn.epoch{}.vacc{:.6f}.input_sigma{:.4f}.total_sigma{:.4f}.dp_eps{:.6f}.dp_delta{:.6f}.ckpt'.format(
                    epoch,
                    valid_dict["recon_acc"],
                    input_sigma, total_dp_sigma,
                    spent_eps_delta.spent_eps,
                    spent_eps_delta.spent_delta
                    )
            model.tf_save(sess, name=ckpt_name) # extra store
            
            if terminate:
                break

            print("******************************************************************")
            print()
            print()
            
        print("Optimization Finished!")
        dp_info = {
            "eps": spent_eps_delta.spent_eps,
            "delta": spent_eps_delta.spent_delta,
            "total_sigma": total_dp_sigma,
            "input_sigma": input_sigma
        }
        valid_dict = test_info(sess, model, False, graph_dict, dp_info, FLAGS.TEST_LOG_FILENAME, total_batch=None)
        np.save(FLAGS.DP_INFO_NPY, dp_info, allow_pickle=True)
                
        ckpt_name='robust_dp_cnn.epoch{}.vacc{:.6f}.input_sigma{:.4f}.total_sigma{:.4f}.dp_eps{:.6f}.dp_delta{:.6f}.ckpt'.format(
            epoch,
            valid_dict["recon_acc"],
            input_sigma, total_dp_sigma,
            spent_eps_delta.spent_eps,
            spent_eps_delta.spent_delta
        )
        model.tf_save(sess, name=ckpt_name) # extra store



if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()

