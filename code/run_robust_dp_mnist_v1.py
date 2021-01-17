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
    sgd_sigma = [np.zeros([FLAGS.BATCH_SIZE]) for _ in range(FLAGS.MAX_PARAM_SIZE)]
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
        for idx in range(len(sgd_sigma)):
            feed_dict[graph_dict["sgd_sigma_holder"][idx]] = sgd_sigma[idx]

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


def compute_K_norm(mats):
    min_K_norm = float("inf")
    for mat in mats:
        u, s_, _ = np.linalg.svd(np.matmul(mat, np.transpose(mat, (0, 2, 1))), full_matrices=False)
        ss = []
        for s in s_:
            ss.append(np.diag(s))
        s = np.stack(ss, 0)
        K = np.matmul(u, np.sqrt(s))

        px_pp_K_norm = np.linalg.norm(K, ord="fro", axis=(1, 2))
        px_pp_I_norm = np.linalg.norm(np.eye(mat.shape[1]), ord="fro", axis=(0, 1))
        # normalize
        K_norm = np.amin(px_pp_K_norm / px_pp_I_norm)
        K_norm = min(min_K_norm, K_norm)
    return K_norm


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
        if FLAGS.IS_K_NORM_LAYERWISED:
            sgd_sigma_holder = [tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE]) for _ in range(FLAGS.MAX_PARAM_SIZE)]
        else:
            sgd_sigma_holder = tf.placeholder(tf.float32, ())
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
        model_op, model_lr = model.dp_optimization(model_loss, gaussian_sanitizer, sgd_sigma_holder, FLAGS.BATCHES_PER_LOT)
        # analysis
        #model_px_pp_jac = model.compute_param_jac_from_input_perturbation(model_loss_clean)
        model_K_norm = model.compute_K_inv_norm_from_input_perturbation_v2(model_loss_clean, is_layerwised=FLAGS.IS_K_NORM_LAYERWISED)
        model_acc = model.cnn_accuracy


        graph_dict = {}
        graph_dict["px_holder"] = px_holder
        graph_dict["data_holder"] = data_holder
        graph_dict["noised_data_holder"] = noised_data_holder
        graph_dict["noise_holder"] = noise_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["sgd_sigma_holder"] = sgd_sigma_holder
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
            #min_A_norm = [float("inf") for _ in range(FLAGS.MAX_PARAM_SIZE)]
            min_K_norm = float("inf")
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
                    
                    #batch_jacs = sess.run(fetches=model_px_pp_jac, feed_dict=feed_dict)
                    #batch_K_norm = compute_K_norm(batch_jacs)
                    #import pdb; pdb.set_trace() # res = sess.run(K_res, feed_dict=feed_dict)
                    batch_K_norm = sess.run(fetches=model_K_norm, feed_dict=feed_dict)
                    if FLAGS.IS_K_NORM_LAYERWISED: # batch_K_norm is [b, #layer]
                        num_layer = batch_K_norm.shape[1]
                        batch_K_norm_layerwised = batch_K_norm
                        batch_K_norm = np.sum(np.square(1.0/batch_K_norm), axis=1)
                        max_idx = np.argmax(batch_K_norm)
                        batch_K_norm =  1.0/np.sqrt(batch_K_norm[max_idx])
                        if batch_K_norm < min_K_norm:
                            min_K_norm = batch_K_norm
                            min_K_norm_layerwised = batch_K_norm_layerwised[max_idx]
                    else: # scalalr
                        min_K_norm = min(min_K_norm, batch_K_norm)
                if train_idx % 10 == 9:
                    print("min K norm: ", min_K_norm)
                    if FLAGS.IS_K_NORM_LAYERWISED:
                        print("min K norm layerwised: ", min_K_norm_layerwised)
            sigma_trans = input_sigma * min_K_norm
            print("Sigma trans: ", sigma_trans)

            if FLAGS.IS_K_NORM_LAYERWISED: # calculate the additive sigma
                sgd_sigma = [np.zeros([FLAGS.BATCH_SIZE]) for _ in range(FLAGS.MAX_PARAM_SIZE)]
                if sigma_trans < FLAGS.TOTAL_DP_SIGMA:
                    sigma_trans_layerwised = input_sigma * min_K_norm_layerwised # [#num_layer]
                    for idx in range(len(sigma_trans_layerwised)):
                        if sigma_trans_layerwised[idx] < FLAGS.TOTAL_DP_SIGMA:
                            if FLAGS.TOTAL_DP_SIGMA - sigma_trans_layerwised[idx] <= FLAGS.INPUT_DP_SIGMA_THRESHOLD:
                                sgd_sigma[idx] = FLAGS.INPUT_DP_SIGMA_THRESHOLD
                            else: sgd_sigma[idx] = FLAGS.TOTAL_DP_SIGMA - sigma_trans_layerwised[idx]
                print("Sigma grads: ", sgd_sigma)
                #
                hetero_sgd_sigma = 1.0/np.sqrt(np.sum(np.square(1.0/sgd_sigma)))
                print("Sigma grads in Heterogeneous form: ", hetero_sgd_sigma)
            else:
                if sigma_trans >= FLAGS.TOTAL_DP_SIGMA:
                    sgd_sigma = 0.0
                elif FLAGS.TOTAL_DP_SIGMA - sigma_trans <= FLAGS.INPUT_DP_SIGMA_THRESHOLD:
                    sgd_sigma = FLAGS.INPUT_DP_SIGMA_THRESHOLD
                else: sgd_sigma = FLAGS.TOTAL_DP_SIGMA - sigma_trans
                print("Sigma grads: ", sgd_sigma)
            
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
                    if FLAGS.IS_K_NORM_LAYERWISED:
                        for idx in range(len(sgd_sigma)):
                            feed_dict[sgd_sigma_holder[idx]] = sgd_sigma[idx]
                    else:
                        feed_dict[sgd_sigma_holder] = sgd_sigma
                    sess.run(fetches=model_op, feed_dict=feed_dict)
                    
                    
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
                    if FLAGS.IS_K_NORM_LAYERWISED:
                        print("Heterogeneous Sigma used: {}".format(hetero_sgd_sigma))
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
                        if FLAGS.IS_K_NORM_LAYERWISED:
                            file.write("Heterogeneous Sigma used: {}\n".format(hetero_sgd_sigma))
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
