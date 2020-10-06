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

    # adv test
    ####################################################################################################
    ch_model_logits = CallableModelWrapper(callable_fn=inference, output_layer='logits')
    ch_model_probs = CallableModelWrapper(callable_fn=inference_prob, output_layer='probs')
    # FastGradientMethod
    fgsm_obj = FastGradientMethod(model=ch_model_probs, sess=sess)
    x_adv_test_fgsm = fgsm_obj.generate(x=graph_dict["images_holder"], 
        eps=FLAGS.ATTACK_SIZE, clip_min=0.0, clip_max=1.0) # testing now

    # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
    # default: eps_iter=0.05, nb_iter=10
    ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
    x_adv_test_ifgsm = ifgsm_obj.generate(x=graph_dict["images_holder"], 
        eps=FLAGS.ATTACK_SIZE, eps_iter=FLAGS.ATTACK_SIZE/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
    
    # MomentumIterativeMethod
    # default: eps_iter=0.06, nb_iter=10
    mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
    x_adv_test_mim = mim_obj.generate(x=graph_dict["images_holder"], 
        eps=FLAGS.ATTACK_SIZE, eps_iter=FLAGS.ATTACK_SIZE/10, nb_iter=10, decay_factor=1.0, clip_min=0.0, clip_max=1.0)

    # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
    # default: eps_iter=0.01, nb_iter=40
    madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
    x_adv_test_madry = madry_obj.generate(x=graph_dict["images_holder"],
        eps=FLAGS.ATTACK_SIZE, eps_iter=FLAGS.ATTACK_SIZE/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
    ####################################################################################################

    
    fetches = [model_loss, model_acc, graph_dict["merged_summary"]]
    if total_batch is None:
        if valid:
            total_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
        else:
            total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    else: total_batch = total_batch

    model_acc = 0 
    model_loss = 0
    for idx in range(total_batch):
        if valid:
            batch_xs, batch_ys, _ = data.next_valid_batch(FLAGS.BATCH_SIZE, True)
        else:
            batch_xs, batch_ys, _ = data.next_test_batch(FLAGS.BATCH_SIZE, True)
        feed_dict = {
            graph_dict["images_holder"]: batch_xs,
            graph_dict["label_holder"]: batch_ys,
            graph_dict["robust_sigma_holder"]: FLAGS.ROBUST_SIGMA
        }

        batch_loss, batch_acc, summary = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_writer.add_summary(summary, idx)
        acc += batch_acc
        loss += batch_loss
        
    model_acc /= total_batch
    model_loss /= total_batch

    # Print info
    print("Loss: {:.4f}, Accuracy: {:.4f}".format(model_loss, model_acc))
    print("Noise eps: {:.4f}, Noise delta: {:.8f}, Noise sigma: {:.4f}".format(
        noise_dp_info["eps"], noise_dp_info["delta"], noise_dp_info["sigma"]))
    print("SGD eps: {:.4f}, SGD delta: {:.8f}, SGD sigma: {:.4f}".format(
        sgd_dp_info["eps"], sgd_dp_info["delta"], sgd_dp_info["sigma"]))
    print()
    
    with open(log_file, "a+") as file: 
        file.write("Loss: {:.4f}, Accuracy: {:.4f}\n".format(model_loss, model_acc))
        file.write("Noise eps: {:.4f}, Noise delta: {:.8f}, Noise sigma: {:.4f}\n".format(
            noise_dp_info["eps"], noise_dp_info["delta"], noise_dp_info["sigma"]))
        file.write("SGD eps: {:.4f}, SGD delta: {:.8f}, SGD sigma: {:.4f}\n".format(
            sgd_dp_info["eps"], sgd_dp_info["delta"], sgd_dp_info["sigma"]))
        file.write("---------------------------------------------------\n")
    
    # Test, calculate max robustness size
    if valid == False:
        # generate adversarial examples
        total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
        ys = []; xs = []; adv_fgsm = []; adv_ifgsm = []; adv_mim = []; adv_madry = []
        for idx in range(total_batch):
            batch_xs, batch_ys, _ = data.next_test_batch(FLAGS.BATCH_SIZE, True)
            ys.append(batch_ys)
            xs.append(batch_xs)
            adv_fgsm.append(sess.run(x_adv_test_fgsm, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob: 1.0}))
            adv_ifgsm.append(sess.run(x_adv_test_ifgsm, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob: 1.0}))
            adv_mim.append(sess.run(x_adv_test_mim, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob: 1.0}))
            adv_madry.append(sess.run(x_adv_test_madry, feed_dict = {x:batch_xs, y_:batch_ys, keep_prob: 1.0}))
        ys = np.concatenate(ys, axis=1)
        advs = {}
        advs["clean"] = np.concatenate(xs, axis=1)
        advs["fgsm"] = np.concatenate(adv_fgsm, axis=1)
        advs["ifgsm"] = np.concatenate(adv_ifgsm, axis=1)
        advs["mim"] = np.concatenate(adv_mim, axis=1)
        advs["madry"] = np.concatenate(adv_madry, axis=1)

        is_acc = {
            "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
        }
        is_robust = {
            "clean": [], "fgsm": [], "ifgsm": [], "mim": [], "madry": []
        }
        adv_acc = {}; adv_robust = {}; adv_robust_acc = {}
        test_size = FLAGS.BATCH_SIZE * total_batch
        for key, adv in advs:
            for idx in range(test_size):
                for n_draws in range(0, FLAGS.ROBUST_SAMPLING_N):
                    feed_dict = {
                        graph_dict["images_holder"]: [adv[idx]],
                        graph_dict["label_holder"]: [ys[idx]],
                        graph_dict["robust_sigma_holder"]: FLAGS.ROBUST_SIGMA
                    }

                    pred = sess.run(fetches=model.cnn_prediction, feed_dict=feed_dict)
                    robust_size = robust_utils.robustnessGGaussian.gaussian_robustness_size(counts=pred[0], 
                        eta=0.05, dp_attack_size=FLAGS.ATTACK_SIZE, sigma=FLAGS.ROBUST_SIGMA)
                    is_robust[key].append(robust_size >= FLAGS.ATTACK_SIZE)
                    is_acc[key].append(np.argmax(batch_ys[0]) == np.argmax(pred[0]))

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


    res_dict = {"model_acc": model_acc, 
                "model_loss": model_loss,
                "adv_acc": adv_acc,
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
        robust_sigma_holder = tf.placeholder(tf.float32, ())
        dp_noise_eps_holder = tf.placeholder(tf.float32, ())
        dp_noise_delta_holder = tf.placeholder(tf.float32, ())
        dp_sgd_sigma_holder = tf.placeholder(tf.float32, ())
        is_training = tf.placeholder(tf.bool, ())

        # model
        model = model_mnist.RDPCNN(images_holder, label_holder, robust_sigma_holder, is_training)
        
        model_loss = model.loss()
        model_acc = model.cnn_accuracy
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["robust_sigma_holder"] = robust_sigma_holder
        graph_dict["dp_noise_eps_holder"] = dp_noise_eps_holder
        graph_dict["dp_noise_delta_holder"] = dp_noise_delta_holder
        graph_dict["dp_sgd_sigma_holder"] = dp_sgd_sigma_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary
        

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # load model
        model.tf_load(sess, name=FLAGS.CNN_CKPT_RESTORE_NAME)

        # tensorboard writer
        test_writer = model_utils.init_writer(FLAGS.TEST_LOG_PATH, g)
        print("\nTest")
        if FLAGS.local:
            total_test_batch = 2
        else:
            total_test_batch = None
        sgd_dp_info = np.load(FLAGS.SGD_DP_INFO_NPY)
        noise_dp_info = np.load(FLAGS.NOISE_DP_INFO_NPY)
        test_info(sess, model, test_writer, graph_dict, sgd_dp_info, noise_dp_info, FLAGS.TEST_LOG_FILENAME, 
            total_batch=total_test_batch)
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
        images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        robust_sigma_holder = tf.placeholder(tf.float32, ())
        dp_noise_eps_holder = tf.placeholder(tf.float32, ())
        dp_noise_delta_holder = tf.placeholder(tf.float32, ())
        dp_sgd_sigma_holder = tf.placeholder(tf.float32, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = model_mnist.RDPCNN(images_holder, label_holder, robust_sigma_holder, is_training)

        # dp setup
        priv_accountant = accountant.GaussianMomentsAccountant(dataset.train_size)
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant,
            [FLAGS.DP_GRAD_CLIPPING_L2NORM/ FLAGS.BATCH_SIZE, True])

        # model training   
        model_loss = model.loss()
        model_op, model_lr = model.optimization(model_loss, gaussian_sanitizer, dp_sgd_sigma_holder)
        model_acc = model.cnn_accuracy
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["robust_sigma_holder"] = robust_sigma_holder
        graph_dict["dp_noise_eps_holder"] = dp_noise_eps_holder
        graph_dict["dp_noise_delta_holder"] = dp_noise_delta_holder
        graph_dict["dp_sgd_sigma_holder"] = dp_sgd_sigma_holder
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
        noise_delta = FLAGS.MAX_NOISE_DELTA
        # G-Lipschitz, K-strongly convex, I lower bound
        noise_eps = dp_utils.robust_sigma_to_dp_eps(noise_delta, FLAGS.ROBUST_SIGMA, data.train_size, FLAGS.MAX_ITERATIONS, 
            FLAGS.G_LIP, FLAGS.K_SCONVEX, FLAGS.LOSS_LOWER_BOUND, moment_order=1)
        sgd_delta = FLAGS.MAX_SGD_DELTA
        sgd_eps = FLAGS.TOTAL_EPS-noise_eps
        sgd_sigma = sgd_eps_delta_to_sigma(sgd_eps, sgd_delta, FLAGS.MAX_ITERATIONS, 
            FLAGS.BATCH_SIZE*FLAGS.BATCHES_PER_LOT/data.train_size, coeff=2)
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
                        robust_sigma_holder: FLAGS.ROBUST_SIGMA,
                        dp_noise_eps_holder: noise_eps,
                        dp_noise_delta_holder: noise_delta,
                        dp_sgd_sigma_holder: sgd_sigma,
                        is_training: True
                    }
                    _ = sess.run(fetches=model_op, feed_dict=feed_dict)

                    if itr_count > FLAGS.MAX_ITERATIONS:
                        terminate = True
                
                # optimization
                fetches = [model_loss, model_acc, model_lr, merged_summary]
                loss, acc, lr, summary = sess.run(fetches=fetches, feed_dict=feed_dict)
                spent_eps_delta = priv_accountant.get_privacy_spent(
                    sess, target_eps=[sgd_eps])[0]
                if spent_eps_delta.spent_delta > sgd_delta or spent_eps_delta.spent_eps > sgd_eps:
                    terminate = True

                train_writer.add_summary(summary, train_idx)
                # Print info
                if train_idx % FLAGS.EVAL_FREQUENCY == (FLAGS.EVAL_FREQUENCY - 1):
                    print("Epoch: {}".format(epoch))
                    print("Learning rate: {}".format(lr))
                    print("Loss: {:.4f}, Accuracy: {:.4f}".format(model_loss, model_acc))
                    print("Noise eps: {:.4f}, Noise delta: {:.8f}, Noise sigma: {:.4f}".format(noise_eps, noise_delta, FLAGS.ROBUST_SIGMA))
                    print("SGD eps: {:.4f}, SGD delta: {:.8f}, SGD sigma: {:.4f}".format(
                        spent_eps_delta.spent_eps, spent_eps_delta.spent_delta, sgd_sigma))
                    print()
                    #model.tf_save(sess) # save checkpoint
                if terminate:
                    break
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
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
            valid_dict = test_info(sess, model, valid_writer, graph_dict, sgd_dp_info, noise_dp_info, FLAGS.VALID_LOG_FILENAME, total_batch=total_valid_batch, valid=True)
            
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
        valid_dict = test_info(sess, model, valid_writer, graph_dict, sgd_dp_info, noise_dp_info, FLAGS.VALID_LOG_FILENAME, total_batch=total_valid_batch, valid=True)
            
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
