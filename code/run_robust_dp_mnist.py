import nn.robust_dp_mnist as model_mnist
import os, math
from PIL import Image
from dependency import *
import utils.model_utils_mnist as  model_utils
from utils.data_utils_mnist import dataset
import robust.utils as robust_utils
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


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test_info(sess, model, test_writer, graph_dict, sgd_dp_info, noise_dp_info, log_file, total_batch=None, valid=False):
    model_loss = model.loss()
    model_acc = model.cnn_accuracy
    
    fetches = [model_loss, model_acc, graph_dict["merged_summary"]]
    if total_batch is None:
        if valid:
            total_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
        else:
            total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    else: total_batch = total_batch

    acc = 0 
    loss = 0
    for idx in range(total_batch):
        if valid:
            batch_xs, batch_ys, _ = data.next_valid_batch(FLAGS.BATCH_SIZE, True)
        else:
            batch_xs, batch_ys, _ = data.next_test_batch(FLAGS.BATCH_SIZE, True)
        feed_dict = {
            graph_dict["images_holder"]: batch_xs,
            graph_dict["label_holder"]: batch_ys
        }

        batch_loss, batch_acc, summary = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_writer.add_summary(summary, idx)
        acc += batch_acc
        loss += batch_loss
        
    acc /= total_batch
    loss /= total_batch

    # Print info
    print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss, acc))
    print("Noise eps: {:.4f}, Noise delta: {:.8f}, Noise sigma: {:.4f}".format(
        noise_dp_info["eps"], noise_dp_info["delta"], noise_dp_info["sigma"]))
    print("SGD eps: {:.4f}, SGD delta: {:.8f}, SGD sigma: {:.4f}".format(
        sgd_dp_info["eps"], sgd_dp_info["delta"], sgd_dp_info["sigma"]))
    print()
    
    with open(log_file, "a+") as file: 
        file.write("Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, acc))
        file.write("Noise eps: {:.4f}, Noise delta: {:.8f}, Noise sigma: {:.4f}\n".format(
            noise_dp_info["eps"], noise_dp_info["delta"], noise_dp_info["sigma"]))
        file.write("SGD eps: {:.4f}, SGD delta: {:.8f}, SGD sigma: {:.4f}\n".format(
            sgd_dp_info["eps"], sgd_dp_info["delta"], sgd_dp_info["sigma"]))
        file.write("---------------------------------------------------\n")
    
    res_dict = {"acc": acc, 
                "loss": loss
                }
    return res_dict


def robust_info(sess, model, graph_dict, log_file):
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
    adv_acc = {}; adv_robust = {}; adv_robust_acc = {}
    test_size = FLAGS.BATCH_SIZE * total_batch
    for key, adv in advs.items():
        for idx in range(test_size):
            pred_labels = np.zeros([FLAGS.NUM_CLASSES])
            for n_draws in range(FLAGS.ROBUST_SAMPLING_N):
                feed_dict = {
                    graph_dict["images_holder"]: [adv[idx]],
                    graph_dict["label_holder"]: [ys[idx]]
                }

                pred = sess.run(fetches=model.cnn_prediction, feed_dict=feed_dict)
                pred_labels[np.argmax(pred[0])] += 1
            try:
                robust_size, robust_eps, robust_delta = robust_utils.gaussian_robustness_size(counts=pred_labels, 
                    eta=0.05, dp_attack_size=FLAGS.ATTACK_SIZE, sigma=FLAGS.ROBUST_SIGMA)
            except:
                import pdb; pdb.set_trace()
                robust_size, robust_eps, robust_delta = robust_utils.gaussian_robustness_size(counts=pred_labels, 
                    eta=0.05, dp_attack_size=FLAGS.ATTACK_SIZE, sigma=FLAGS.ROBUST_SIGMA)
            is_robust[key].append(robust_size >= FLAGS.ATTACK_SIZE)
            is_acc[key].append(np.argmax(ys[idx]) == np.argmax(pred_labels))
        
        adv_acc[key] = np.sum(is_acc[key]) * 1.0 / test_size
        adv_robust[key] = np.sum(is_robust[key]) * 1.0 / test_size
        adv_robust_acc[key] = np.sum([a and b for a,b in zip(is_robust[key], is_acc[key])])*1.0/np.sum(is_robust[key])

        # Print info
        print("{}:".format(key))
        print("accuracy: {}, robustness: {}, robust_accuracy: {}".format(
            adv_acc[key], adv_robust[key], adv_robust_acc[key]))
        print()
        print()
        
        with open(log_file, "a+") as file: 
            file.write("accuracy: {}, robustness: {}, robust_accuracy: {}\n".format(
            adv_acc[key], adv_robust[key], adv_robust_acc[key]))
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
        model = model_mnist.RDPCNN(images_holder, label_holder, FLAGS.ROBUST_SIGMA, is_training)# for adv examples

        model_loss = model.loss()
        model_acc = model.cnn_accuracy
        merged_summary = tf.summary.merge_all()

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
        graph_dict["merged_summary"] = merged_summary
        

    with tf.Session(graph=g) as sess:
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
        test_writer = model_utils.init_writer(FLAGS.TEST_LOG_PATH, g)
        print("\nTest")
        if FLAGS.local:
            total_test_batch = 2
        else:
            total_test_batch = None
        sgd_dp_info = np.load(FLAGS.SGD_DP_INFO_NPY, allow_pickle=True).item()
        noise_dp_info = np.load(FLAGS.NOISE_DP_INFO_NPY, allow_pickle=True).item()
        test_info(sess, model, test_writer, graph_dict, sgd_dp_info, noise_dp_info, FLAGS.TEST_LOG_FILENAME, 
            total_batch=total_test_batch)
        robust_info(sess, model, graph_dict, FLAGS.ROBUST_LOG_FILENAME)
        test_writer.close() 



def train():
    """
    """
    import time
    tf.reset_default_graph()
    g = tf.get_default_graph()
    # attack_target = 8
    with g.as_default():
        # Placeholder nodes.
        images_holder = tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES])
        is_training = tf.placeholder(tf.bool, ())
        # params
        noise_delta = FLAGS.MAX_NOISE_DELTA
        # G-Lipschitz, K-strongly convex, I lower bound
        noise_eps = dp_utils.robust_sigma_to_dp_eps(noise_delta, FLAGS.ROBUST_SIGMA, data.train_size, FLAGS.MAX_ITERATIONS, 
            FLAGS.G_LIP, FLAGS.K_SCONVEX, FLAGS.LOSS_LOWER_BOUND, moment_order=1)
        sgd_delta = FLAGS.MAX_SGD_DELTA
        if sgd_delta != 0 and not FLAGS.NO_DP_SGD:
            sgd_eps = FLAGS.TOTAL_EPS-noise_eps
            sgd_sigma = dp_utils.sgd_eps_delta_to_sigma(sgd_eps, sgd_delta, FLAGS.MAX_ITERATIONS, 
                FLAGS.BATCH_SIZE*FLAGS.BATCHES_PER_LOT/data.train_size, coeff=2)
        else:
            sgd_eps = 0
            sgd_sigma = 0
        # model
        model = model_mnist.RDPCNN(images_holder, label_holder, FLAGS.ROBUST_SIGMA, is_training)


        priv_accountant = accountant.GaussianMomentsAccountant(data.train_size)
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant,
            [FLAGS.DP_GRAD_CLIPPING_L2NORM/ FLAGS.BATCH_SIZE, True])

        # model training   
        model_loss = model.loss()
        if sgd_delta != 0 and not FLAGS.NO_DP_SGD:
            model_op, model_lr = model.dp_optimization(model_loss, gaussian_sanitizer, sgd_sigma)
        else:
            model_op, _, _, model_lr = model.optimization(model_loss)
        model_acc = model.cnn_accuracy
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.load_model:
            print("CNN loaded.")
            model.tf_load(sess, name=FLAGS.CNN_CKPT_RESTORE_NAME)
        # For tensorboard
        train_writer = model_utils.init_writer(FLAGS.TRAIN_LOG_PATH, g)
        valid_writer = model_utils.init_writer(FLAGS.VALID_LOG_PATH, g)
        
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
            for train_idx in range(total_train_lot):
                terminate = False
                for batch_idx in range(FLAGS.BATCHES_PER_LOT):
                    itr_count += 1
                    batch_xs, batch_ys, _ = data.next_train_batch(FLAGS.BATCH_SIZE, True)
                    feed_dict = {
                        images_holder: batch_xs,
                        label_holder: batch_ys,
                        is_training: True
                    }
                    _ = sess.run(fetches=model_op, feed_dict=feed_dict)

                    if itr_count > FLAGS.MAX_ITERATIONS:
                        terminate = True
                
                # optimization
                fetches = [model_loss, model_acc, model_lr, merged_summary]
                loss, acc, lr, summary = sess.run(fetches=fetches, feed_dict=feed_dict)
                #import pdb; pdb.set_trace()
                spent_eps_delta, selected_moment_orders = priv_accountant.get_privacy_spent(sess, target_eps=[sgd_eps])
                #spent_eps_delta, selected_moment_orders = priv_accountant.get_privacy_spent(sess, target_deltas=[sgd_delta])
                spent_eps_delta = spent_eps_delta[0]
                selected_moment_orders = selected_moment_orders[0]
                if sgd_delta != 0 and not FLAGS.NO_DP_SGD:
                    if spent_eps_delta.spent_delta > sgd_delta or spent_eps_delta.spent_eps > sgd_eps:
                        terminate = True

                train_writer.add_summary(summary, train_idx)
                # Print info
                if itr_count % FLAGS.EVAL_TRAIN_FREQUENCY == (FLAGS.EVAL_TRAIN_FREQUENCY - 1):
                    print("Epoch: {}".format(epoch))
                    print("Iteration: {}".format(itr_count))
                    print("Learning rate: {}".format(lr))
                    print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss, acc))
                    print("Noise eps: {:.4f}, Noise delta: {:.8f}, Noise sigma: {:.4f}".format(noise_eps, noise_delta, FLAGS.ROBUST_SIGMA))
                    print("SGD eps: {:.4f}, SGD delta: {:.8f}, SGD sigma: {:.4f}".format(
                        spent_eps_delta.spent_eps, spent_eps_delta.spent_delta, sgd_sigma))
                    print()
                    #model.tf_save(sess) # save checkpoint
                if terminate:
                    break
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
            if epoch % FLAGS.EVAL_VALID_FREQUENCY == (FLAGS.EVAL_VALID_FREQUENCY - 1):
                # validation
                print("\n******************************************************************")
                print("Validation")
                sgd_dp_info = {
                    "eps": spent_eps_delta.spent_eps,
                    "delta": spent_eps_delta.spent_delta,
                    "sigma": sgd_sigma
                }
                noise_dp_info = {
                    "eps": noise_eps,
                    "delta": noise_delta,
                    "sigma":  FLAGS.ROBUST_SIGMA
                }
                valid_dict = test_info(sess, model, valid_writer, graph_dict, sgd_dp_info, noise_dp_info, FLAGS.VALID_LOG_FILENAME, total_batch=None, valid=True)
                
                ckpt_name='robust_dp_cnn.epoch{}.vloss{:.6f}.vacc{:.6f}.noise_eps{:.6f}.noise_delta{:.6f}.sgd_eps{:.6f}.sgd_delta{:.6f}.ckpt'.format(
                        epoch,
                        valid_dict["loss"],
                        valid_dict["acc"],
                        noise_eps, noise_delta,
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
        sgd_dp_info = {
                "eps": spent_eps_delta.spent_eps,
                "delta": spent_eps_delta.spent_delta,
                "sigma": sgd_sigma
        }
        noise_dp_info = {
            "eps": noise_eps,
            "delta": noise_delta,
            "sigma":  FLAGS.ROBUST_SIGMA
        }
        valid_dict = test_info(sess, model, valid_writer, graph_dict, sgd_dp_info, noise_dp_info, FLAGS.VALID_LOG_FILENAME, total_batch=None, valid=True)
            
        ckpt_name='robust_dp_cnn.epoch{}.vloss{:.6f}.vacc{:.6f}.noise_eps{:.6f}.noise_delta{:.6f}.sgd_eps{:.6f}.sgd_delta{:.6f}.ckpt'.format(
                epoch,
                valid_dict["loss"],
                valid_dict["acc"],
                noise_eps, noise_delta,
                spent_eps_delta.spent_eps,
                spent_eps_delta.spent_delta
                )
        model.tf_save(sess, name=ckpt_name) # extra store
        np.save("sgd_dp_info.sgd_eps{:.6f}.sgd_delta{:.6f}.npy".format(
                spent_eps_delta.spent_eps,
                spent_eps_delta.spent_delta
                ), sgd_dp_info)
        np.save("noise_dp_info.noise_eps{:.6f}.noise_delta{:.6f}.npy".format(
                noise_eps, noise_delta
                ), noise_dp_info)

        train_writer.close() 
        valid_writer.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
