import nn.robust_dp_mnist_v2 as model_mnist
import os, math
from PIL import Image
from dependency import *
import utils.model_utils_mnist as  model_utils
from utils.data_utils_mnist import dataset
import robust.additiveNoise as additiveNoise
import differential_privacy.utils as dp_utils
import differential_privacy.privacy_accountant.tf.accountant_v2 as accountant
import differential_privacy.dp_sgd.dp_optimizer.sanitizer as sanitizer
# attacks
from cleverhans.attacks import BasicIterativeMethod, FastGradientMethod, MadryEtAl, MomentumIterativeMethod
from cleverhans.attacks_tf import fgm, fgsm
from cleverhans.model import CallableModelWrapper, CustomCallableModelWrapper

model_utils.set_flags()

data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED, 
    adv_path_prefix=FLAGS.ADV_PATH_PREFIX)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_INDEX


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test_info(sess, model, test_writer, graph_dict, dp_info, log_file, total_batch=None, valid=False):
    model_loss = model.loss()
    model_acc = model.cnn_accuracy
    input_sigma = dp_info["input_sigma"]
    fetches = [model_loss, model_acc]
    if total_batch is None:
        if valid:
            total_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
        else:
            total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    else: total_batch = total_batch

    acc = 0 
    loss = 0
    if FLAGS.IS_MGM_LAYERWISED:
        sgd_sigma = np.zeros([FLAGS.MAX_PARAM_SIZE])
    else:
        sgd_sigma = 0.0
    for idx in range(total_batch):
        if valid:
            batch_xs, batch_ys, _ = data.next_valid_batch(FLAGS.BATCH_SIZE, True)
        else:
            batch_xs, batch_ys, _ = data.next_test_batch(FLAGS.BATCH_SIZE, True)
        noise = np.random.normal(loc=0.0, scale=input_sigma, size=batch_xs.shape)
        #px_noised_xs = np.split(batch_xs+noise, FLAGS.BATCH_SIZE, axis=0)
        feed_dict = {
            graph_dict["noise_holder"]: noise,
            graph_dict["noised_data_holder"]: batch_xs+noise,
            graph_dict["label_holder"]: batch_ys,
            graph_dict["is_training"]: False
        }

        if FLAGS.IS_MGM_LAYERWISED:
            for idx in range(len(sgd_sigma)):
                feed_dict[graph_dict["sgd_sigma_holder"][idx]] = sgd_sigma[idx]
        else:
            feed_dict[sgd_sigma_holder] = sgd_sigma

        batch_loss, batch_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        acc += batch_acc
        loss += batch_loss
        
    acc /= total_batch
    loss /= total_batch

    # Print info
    print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss, acc))
    print("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}".format(
        dp_info["eps"], dp_info["delta"], dp_info["total_sigma"], dp_info["input_sigma"]))
    
    with open(log_file, "a+") as file: 
        file.write("Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, acc))
        file.write("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}\n".format(
        dp_info["eps"], dp_info["delta"], dp_info["total_sigma"], dp_info["input_sigma"]))
        file.write("---------------------------------------------------\n")
    
    res_dict = {"acc": acc, 
                "loss": loss
                }
    return res_dict


def robust_info(sess, model, graph_dict, log_file):
    images_holder = graph_dict["images_holder"]
    label_holder = graph_dict["label_holder"]
    is_training = graph_dict["is_training"]
    logits, _ = model.cnn.prediction(images_holder)

    if not FLAGS.LOAD_ADVS:
        # generate adversarial examples
        total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
        ys = []; xs = []; adv_fgsm = []; adv_ifgsm = []; adv_mim = []; adv_madry = []
        for idx in range(total_batch):
            batch_xs, batch_ys, _ = data.next_test_batch(FLAGS.BATCH_SIZE, True)
            ys.append(batch_ys)
            xs.append(batch_xs)
            adv_fgsm.append(sess.run(graph_dict["x_advs"]["fgsm"], feed_dict = {
                                graph_dict["images_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))
            adv_ifgsm.append(sess.run(graph_dict["x_advs"]["ifgsm"], feed_dict = {
                                graph_dict["images_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))
            adv_mim.append(sess.run(graph_dict["x_advs"]["mim"], feed_dict = {
                                graph_dict["images_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))
            adv_madry.append(sess.run(graph_dict["x_advs"]["madry"], feed_dict = {
                                graph_dict["images_holder"]:batch_xs, 
                                graph_dict["label_holder"] :batch_ys}))

        ys = np.concatenate(ys, axis=0)
        advs = {}
        advs["clean"] = np.concatenate(xs, axis=0)
        advs["fgsm"] = np.concatenate(adv_fgsm, axis=0)
        advs["ifgsm"] = np.concatenate(adv_ifgsm, axis=0)
        advs["mim"] = np.concatenate(adv_mim, axis=0)
        advs["madry"] = np.concatenate(adv_madry, axis=0)
        np.save("adv_examples.npy", advs)
    else:
        advs = np.load("adv_examples.npy", allow_pickle=True).item()

    is_acc = {
        "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
    }
    is_robust = {
        "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
    }
    res_dicts = {}
    adv_acc = {}; adv_robust = {}; adv_robust_acc = {}
    n_sampling = FLAGS.NUM_SAMPLING
    test_size = ys.shape[0]

    print("Robust defense with input std: {}".format(FLAGS.INPUT_SIGMA))
    for key, adv in advs.items():
        one_hot = np.zeros(shape=(test_size, FLAGS.NUM_CLASSES))
        for i in range(n_sampling):
            fetches = logits
            feed_dict = {
                images_holder: adv + np.random.normal(scale=std, size = adv.shape),
                label_holder: ys,
                is_training: False
            }
            logits_np = sess.run(fetches=fetches, feed_dict=feed_dict)
            one_hot[np.arange(test_size), logits_np.argmax(axis=1)] += 1
        
        robust_res[key] = np.apply_along_axis(func1d=additiveNoise.isRobust, axis=1, arr=one_hot, std=std, attack_size=FLAGS.ATTACK_SIZE)
        adv_robust[key] = np.sum(robust_res[key][:,0]==True)/n_sampling
        adv_robust_acc[key] = np.sum(np.logical_and(robust_res[key][:,0]==True, one_hot.argmax(axis=1)==ys.argmax(axis=1)))/n_sampling
        adv_acc[key] = np.sum(one_hot.argmax(axis=1)==ys.argmax(axis=1))/n_sampling

        # Print info
        print("{}:".format(key))
        print("accuracy: {}, robustness: {}, robust_accuracy: {}".format(
            adv_acc[key], adv_robust[key], adv_robust_acc[key]))
        print("avg. bound is {} at sigma = {}\n".format(np.mean(robust_res[key][:,1]), std))
        print()
        print()
    
        with open("{}.std{}".format(log_file, std), "a+") as file: 
            file.write("accuracy: {}, robustness: {}, robust_accuracy: {}\n".format(
                adv_acc[key], adv_robust[key], adv_robust_acc[key]))
            file.write("avg. bound is {} at sigma = {}\n".format(np.mean(robust_res[key][:,1]), std))
            file.write("===================================================\n")
    
    res_dict = {"adv_acc": adv_acc, 
                "adv_robust": adv_robust,
                "adv_robust_acc": adv_robust_acc
                }
    return res_dict



def test():
    """
    """
    tf.reset_default_graph()
    g = tf.get_default_graph()

    with g.as_default():
        # Placeholder nodes.
        images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        is_training = tf.placeholder(tf.bool, ())

        # model
        model = model_mnist.RDPCNN(images_holder, label_holder, FLAGS.INPUT_SIGMA, is_training) # for adv examples

        model_loss = model.loss()
        model_acc = model.cnn_accuracy

        # robust
        def inference(x): 
            logits, _ = model.cnn.prediction(x)
            return logits
        def inference_prob(x):
            _, probs = model.cnn.prediction(x)
            return probs
        

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["is_training"] = is_training
        

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # load model
        model.tf_load(sess, name=FLAGS.CNN_CKPT_RESTORE_NAME)

        # adv test
        ####################################################################################################
        x_advs = {}
        ch_model_logits = CallableModelWrapper(callable_fn=inference, output_layer='logits')
        ch_model_probs = CallableModelWrapper(callable_fn=inference_prob, output_layer='probs')
        # FastGradientMethod
        fgsm_obj = FastGradientMethod(model=ch_model_probs, sess=sess)
        x_advs["fgsm"] = fgsm_obj.generate(x=images_holder, 
            eps=FLAGS.ATTACK_SIZE, clip_min=0.0, clip_max=1.0) # testing now

        # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
        # default: eps_iter=0.05, nb_iter=10
        ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
        x_advs["ifgsm"] = ifgsm_obj.generate(x=images_holder, 
            eps=FLAGS.ATTACK_SIZE, eps_iter=FLAGS.ATTACK_SIZE/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
        
        # MomentumIterativeMethod
        # default: eps_iter=0.06, nb_iter=10
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        x_advs["mim"] = mim_obj.generate(x=images_holder, 
            eps=FLAGS.ATTACK_SIZE, eps_iter=FLAGS.ATTACK_SIZE/10, nb_iter=10, decay_factor=1.0, clip_min=0.0, clip_max=1.0)

        # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
        # default: eps_iter=0.01, nb_iter=40
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        x_advs["madry"] = madry_obj.generate(x=images_holder,
            eps=FLAGS.ATTACK_SIZE, eps_iter=FLAGS.ATTACK_SIZE/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
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
        test_info(sess, model, None, graph_dict, dp_info, FLAGS.TEST_LOG_FILENAME, 
            total_batch=total_test_batch)
        robust_info(sess, model, graph_dict, FLAGS.ROBUST_LOG_FILENAME)


def __compute_S_min_lower_bound_5(M, param_size, sensitivity):
    #import pdb; pdb.set_trace()
    x = np.linalg.solve(M, sensitivity)
    ele_sum = np.matmul(np.transpose(sensitivity), x)
    if ele_sum < 0: return 0
    #print(ele_sum)
    #S_min = np.sqrt(FLAGS.MAX_ITERATIONS*np.linalg.norm(sensitivity, ord=2)/ele_sum)
    S_min = np.sqrt(1e5*np.linalg.norm(sensitivity, ord=2)/ele_sum)
    
    '''
    e = np.ones((param_size, 1))
    x = np.linalg.solve(M, e)
    ele_sum = np.matmul(np.transpose(e), x)
    print(ele_sum)
    S_min = np.sqrt(param_size*FLAGS.MAX_ITERATIONS/ele_sum)
    '''
    '''
    M_inv = np.linalg.inv(M)
    ele_sum = np.sum(M_inv)
    print(ele_sum)
    S_min = np.sqrt(param_size*FLAGS.MAX_ITERATIONS/ele_sum)
    '''
    return S_min


def __compute_S_min_lower_bound_4(M, param_size):
    cond_M = np.linalg.cond(M)
    M_fro = np.linalg.norm(M, ord="fro", axis=(0, 1)) # ()
    S_min = M_fro/cond_M * np.sqrt(param_size)
    return S_min

def __compute_S_min_lower_bound_3(M, param_size):
    #det_M = np.linalg.det(M) # ()
    _, logdet = np.linalg.slogdet(M)
    det_M = np.exp(logdet)
    base = (param_size-1.0)/param_size
    S_min = np.abs(det_M) * np.power(base, (param_size-1.0)/2)
    return S_min

def __compute_S_min_lower_bound_2(M, param_size):
    #det_M = np.linalg.det(M) # ()
    _, logdet = np.linalg.slogdet(M)
    det_M = np.exp(logdet)
    M_fro = np.linalg.norm(M, ord="fro", axis=(0, 1)) # ()
    l_base = (param_size-1.0)/np.square(M_fro)
    l = np.abs(det_M) * np.power(l_base, (param_size-1.0)/2)

    return l

def __compute_S_min_lower_bound_1(M, param_size):
    #det_M = np.linalg.det(M) # ()
    _, logdet = np.linalg.slogdet(M)
    det_M = np.exp(logdet)
    M_fro = np.linalg.norm(M, ord="fro", axis=(0, 1)) # ()
    l_base = (param_size-1.0)/np.square(M_fro)
    l = np.abs(det_M) * np.power(l_base, (param_size-1.0)/2)
    S_base = (param_size-1.0)/(np.square(M_fro)-np.square(l))
    S_min = np.abs(det_M) * np.power(S_base, (param_size-1.0)/2)

    return S_min

def __compute_S_min_from_Jac(Jac, sen):
    #import pdb; pdb.set_trace()
    # Jac [batch, param_size, input_size]
    batch_size = Jac.shape[0]
    param_size = Jac.shape[1]
    batch_S_min = []
    '''
    if param_size <= 1000:
        for b in range(batch_size):
            J = Jac[b, :, :]
            M = np.matmul(J, np.transpose(J))
            s = np.linalg.svd(M, full_matrices=True, compute_uv=False)
            s_sum = np.sum(1.0/s)
            batch_S_min.append(1.0/np.sqrt(s_sum)/np.sqrt(param_size))
    else:
        for b in range(batch_size):
            J = Jac[b, :, :]
            M = np.matmul(J, np.transpose(J))
            S_min = __compute_S_min_lower_bound_5(M, param_size)
            batch_S_min.append(np.sqrt(S_min / param_size)/np.sqrt(param_size))
    '''
    for b in range(batch_size):
        J = Jac[b, :, :]
        sen_ = sen[b]
        M = np.matmul(J, np.transpose(J))
        S_min = __compute_S_min_lower_bound_5(M, param_size, sen_)
        batch_S_min.append(S_min)
    batch_S_min = np.stack(batch_S_min, axis=0)
    return batch_S_min

def compute_S_min_from_Jac(Jac, sens, is_layerwised=True):
    if is_layerwised:
        batch_S_min_layerwised = []
        for (J, sen) in zip(Jac, sens):
            batch_S_min_layerwised.append(__compute_S_min_from_Jac(J, sen))
        batch_S_min_layerwised = np.stack(batch_S_min_layerwised, axis=1)
        return batch_S_min_layerwised
    else:
        return __compute_S_min_from_Jac(Jac, sens)
        

def train():
    """
    """
    import time
    input_sigma = FLAGS.INPUT_SIGMA
    total_dp_sigma = FLAGS.TOTAL_DP_SIGMA
    total_dp_delta = FLAGS.TOTAL_DP_DELTA
    total_dp_epsilon = FLAGS.TOTAL_DP_EPSILON

    tf.reset_default_graph()
    g = tf.get_default_graph()
    # attack_target = 8
    with g.as_default():
        # Placeholder nodes.
        px_holder = [tf.placeholder(tf.float32, [1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS]) for _ in range(FLAGS.BATCH_SIZE)]
        data_holder = tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        noised_data_holder = tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        noise_holder = tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES])
        if FLAGS.IS_MGM_LAYERWISED:
            sgd_sigma_holder = [tf.placeholder(tf.float32, ()) for _ in range(FLAGS.MAX_PARAM_SIZE)]
            trans_sigma_holder = [tf.placeholder(tf.float32, ()) for _ in range(FLAGS.MAX_PARAM_SIZE)]
        else:
            sgd_sigma_holder = tf.placeholder(tf.float32, ())
            trans_sigma_holder = tf.placeholder(tf.float32, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = model_mnist.RDPCNN(noised_data=noised_data_holder, noise=noise_holder, label=label_holder, input_sigma=input_sigma, is_training=is_training)
        priv_accountant = accountant.GaussianMomentsAccountant(data.train_size)
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant,
            [FLAGS.DP_GRAD_CLIPPING_L2NORM, True])

        # model training   
        model_loss = model.loss()
        model_loss_clean = model.loss_clean()
        # training
        #model_op, _, _, model_lr = model.optimization(model_loss)
        model_op, model_lr = model.dp_optimization(model_loss, gaussian_sanitizer, sgd_sigma_holder, trans_sigma_holder, FLAGS.BATCHES_PER_LOT, is_layerwised=FLAGS.IS_MGM_LAYERWISED)
        # analysis
        model_Jac, model_sens = model.compute_Jac_from_input_perturbation(model_loss_clean, FLAGS.DP_GRAD_CLIPPING_L2NORM, is_layerwised=FLAGS.IS_MGM_LAYERWISED)
        model_S_min, model_res = model.compute_S_min_from_input_perturbation(model_loss_clean, is_layerwised=FLAGS.IS_MGM_LAYERWISED)
        model_acc = model.cnn_accuracy


        graph_dict = {}
        graph_dict["px_holder"] = px_holder
        graph_dict["data_holder"] = data_holder
        graph_dict["noised_data_holder"] = noised_data_holder
        graph_dict["noise_holder"] = noise_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["sgd_sigma_holder"] = sgd_sigma_holder
        graph_dict["trans_sigma_holder"] = trans_sigma_holder
        graph_dict["is_training"] = is_training

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.load_model:
            print("CNN loaded.")
            model.tf_load(sess, name=FLAGS.CNN_CKPT_RESTORE_NAME)
        
        if FLAGS.local:
            total_train_lot = 2
            total_valid_lot = 2
        else:
            total_train_lot = int(data.train_size/FLAGS.BATCH_SIZE/FLAGS.BATCHES_PER_LOT)
            total_valid_lot = None        
        
        print("Training...")
        itr_count = 0
        for epoch in range(FLAGS.NUM_EPOCHS):
            start_time = time.time()
            # Compute A norm
            
            min_S_min = float("inf")
            if FLAGS.IS_MGM_LAYERWISED:
                min_S_min_layerwised = [float("inf") for _ in range(FLAGS.MAX_PARAM_SIZE)]
            #'''
            for train_idx in range(total_train_lot):
                for batch_idx in range(FLAGS.BATCHES_PER_LOT):
                    batch_xs, batch_ys, _ = data.next_train_batch(FLAGS.BATCH_SIZE, True)
                    noise = np.random.normal(loc=0.0, scale=input_sigma, size=batch_xs.shape)
                    feed_dict = {
                        noise_holder: noise,
                        noised_data_holder: batch_xs+noise,
                        label_holder: batch_ys,
                        is_training: True
                    }
                    #batch_S_min = sess.run(fetches=model_S_min[0], feed_dict=feed_dict)
                    batch_Jac, batch_sens = sess.run(fetches=[model_Jac, model_sens], feed_dict=feed_dict)
                    batch_S_min = compute_S_min_from_Jac(batch_Jac, batch_sens, FLAGS.IS_MGM_LAYERWISED)
                    #import pdb; pdb.set_trace()
                    if FLAGS.IS_MGM_LAYERWISED: # batch_K_norm is [b, #layer]
                        #import pdb; pdb.set_trace()
                        num_layer = batch_S_min.shape[1]
                        batch_S_min_layerwised = np.amin(batch_S_min, axis=0)
                        min_S_min_layerwised = min_S_min_layerwised[:len(batch_S_min_layerwised)]
                        min_S_min_layerwised = np.minimum(min_S_min_layerwised, batch_S_min_layerwised)
                    else: # scalalr
                        min_S_min = min(min_S_min, min(batch_S_min))
                
                if train_idx % 100 == 9:
                    if FLAGS.IS_MGM_LAYERWISED:
                        print("min S_min layerwised: ", min_S_min_layerwised)
                    else: print("min S_min: ", min_S_min)
            if FLAGS.IS_MGM_LAYERWISED:
                sigma_trans = input_sigma * min_S_min_layerwised
                print("Sigma trans: ", sigma_trans)
                sgd_sigma = np.zeros([FLAGS.MAX_PARAM_SIZE])
                for idx in range(len(sigma_trans)):
                    if sigma_trans[idx] < FLAGS.TOTAL_DP_SIGMA:
                        if FLAGS.TOTAL_DP_SIGMA - sigma_trans[idx] <= FLAGS.INPUT_DP_SIGMA_THRESHOLD:
                            sgd_sigma[idx] = FLAGS.INPUT_DP_SIGMA_THRESHOLD
                        else: sgd_sigma[idx] = FLAGS.TOTAL_DP_SIGMA - sigma_trans[idx]
                print("Sigma grads: ", sgd_sigma)
                #
                #hetero_sgd_sigma = np.sqrt(len(min_S_min_layerwised)/np.sum(np.square(1.0/sgd_sigma[:len(min_S_min_layerwised)])))
                #print("Sigma grads in Heterogeneous form: ", hetero_sgd_sigma)
            else:
                sigma_trans = input_sigma * min_S_min
                print("Sigma trans: ", sigma_trans)
                if sigma_trans >= FLAGS.TOTAL_DP_SIGMA:
                    sgd_sigma = 0.0
                elif FLAGS.TOTAL_DP_SIGMA - sigma_trans <= FLAGS.INPUT_DP_SIGMA_THRESHOLD:
                    sgd_sigma = FLAGS.INPUT_DP_SIGMA_THRESHOLD
                else: sgd_sigma = FLAGS.TOTAL_DP_SIGMA - sigma_trans
                print("Sigma grads: ", sgd_sigma)
            #'''
            #sigma_trans = [34.59252105,0.71371817,16.14990762,0.59402054,0.,0.50355514,30.09081199,0.40404256,21.18426806,0.35788509,0.,0.30048024,0.,0.30312875]
            #sgd_sigma = [0.,0.8,0.,0.8,1.,0.8,0.,0.8,0.,0.8,1.,0.8,1.,0.8,0.,0.,0.,0.]
            #sgd_sigma = [34.59252105,1.0,16.14990762,1.0,1.,1.0,30.09081199,1.0,21.18426806,1.0,1.,1.0,1.,1.0,0.,0.,0.,0.]
            for train_idx in range(total_train_lot):
                terminate = False
                for batch_idx in range(FLAGS.BATCHES_PER_LOT):
                    itr_count += 1
                    batch_xs, batch_ys, _ = data.next_train_batch(FLAGS.BATCH_SIZE, True)
                    noise = np.random.normal(loc=0.0, scale=input_sigma, size=batch_xs.shape)
                    feed_dict = {
                        noise_holder: noise,
                        noised_data_holder: batch_xs+noise,
                        label_holder: batch_ys,
                        is_training: True
                    }
                    if FLAGS.IS_MGM_LAYERWISED:
                        for idx in range(len(sigma_trans)):
                            feed_dict[sgd_sigma_holder[idx]] = sgd_sigma[idx]
                            feed_dict[trans_sigma_holder[idx]] = sigma_trans[idx]
                    else:
                        feed_dict[sgd_sigma_holder] = sgd_sigma
                        feed_dict[trans_sigma_holder] = sigma_trans
                    sess.run(fetches=[model_op], feed_dict=feed_dict)
                    
                    
                    if itr_count > FLAGS.MAX_ITERATIONS:
                        terminate = True
                
                # optimization
                fetches = [model_loss, model_acc, model_lr]
                loss, acc, lr = sess.run(fetches=fetches, feed_dict=feed_dict)
                #import pdb; pdb.set_trace()
                spent_eps_delta, selected_moment_orders = priv_accountant.get_privacy_spent(sess, target_eps=[total_dp_epsilon])
                spent_eps_delta = spent_eps_delta[0]
                selected_moment_orders = selected_moment_orders[0]
                if spent_eps_delta.spent_delta > total_dp_delta or spent_eps_delta.spent_eps > total_dp_epsilon:
                    terminate = True

                # Print info
                if train_idx % FLAGS.EVAL_TRAIN_FREQUENCY == (FLAGS.EVAL_TRAIN_FREQUENCY - 1):
                    print("Epoch: {}".format(epoch))
                    print("Iteration: {}".format(itr_count))
                    print("Sigma used:{}".format(sigma_trans))
                    print("SGD Sigma: {}".format(sgd_sigma))
                    print("Learning rate: {}".format(lr))
                    print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss, acc))
                    print("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}".format(
                        spent_eps_delta.spent_eps, spent_eps_delta.spent_delta, total_dp_sigma, input_sigma))
                    print()
                    #model.tf_save(sess) # save checkpoint

                    with open(FLAGS.TRAIN_LOG_FILENAME, "a+") as file: 
                        file.write("Epoch: {}\n".format(epoch))
                        file.write("Iteration: {}\n".format(itr_count))
                        file.write("Sigma used: {}\n".format(sigma_trans))
                        file.write("SGD Sigma: {}\n".format(sgd_sigma))
                        file.write("Learning rate: {}\n".format(lr))
                        file.write("Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, acc))
                        file.write("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}\n".format(
                            spent_eps_delta.spent_eps, spent_eps_delta.spent_delta, total_dp_sigma, input_sigma))
                        file.write("\n")
                if terminate:
                    break
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
            if epoch % FLAGS.EVAL_VALID_FREQUENCY == (FLAGS.EVAL_VALID_FREQUENCY - 1):
            #if epoch >= 0:
                # validation
                print("\n******************************************************************")
                print("Validation")
                dp_info = {
                    "eps": spent_eps_delta.spent_eps,
                    "delta": spent_eps_delta.spent_delta,
                    "total_sigma": total_dp_sigma,
                    "input_sigma": input_sigma
                }
                valid_dict = test_info(sess, model, None, graph_dict, dp_info, FLAGS.VALID_LOG_FILENAME, total_batch=None, valid=True)
                np.save(FLAGS.DP_INFO_NPY, dp_info, allow_pickle=True)
                ckpt_name='robust_dp_cnn.epoch{}.vloss{:.6f}.vacc{:.6f}.input_sigma{:.4f}.total_sigma{:.4f}.dp_eps{:.6f}.dp_delta{:.6f}.ckpt'.format(
                        epoch,
                        valid_dict["loss"],
                        valid_dict["acc"],
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
        valid_dict = test_info(sess, model, None, graph_dict, dp_info, None, total_batch=None, valid=True)
        np.save(FLAGS.DP_INFO_NPY, dp_info, allow_pickle=True)
                
        ckpt_name='robust_dp_cnn.epoch{}.vloss{:.6f}.vacc{:.6f}.input_sigma{:.4f}.total_sigma{:.4f}.dp_eps{:.6f}.dp_delta{:.6f}.ckpt'.format(
            epoch,
            valid_dict["loss"],
            valid_dict["acc"],
            input_sigma, total_dp_sigma,
            spent_eps_delta.spent_eps,
            spent_eps_delta.spent_delta
        )
        model.tf_save(sess, name=ckpt_name) # extra store



if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
