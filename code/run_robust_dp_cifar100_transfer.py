import nn.robust_dp_cifar100_transfer as model_cifar100
import os, math
from PIL import Image
from dependency import *
from tensornets.preprocess import keras_resnet_preprocess
import utils.model_utils_cifar100 as  model_utils
from utils.data_utils_cifar100 import dataset
import robust.additiveNoise as additiveNoise
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
    model_loss = model.loss()
    model_loss_reg = model.loss_reg()
    model_acc = model.cnn_accuracy
    input_sigma = dp_info["input_sigma"]

    full_data = False
    
    batch_size = FLAGS.BATCH_SIZE
    if total_batch is None:
        if is_valid:
            total_batch = int(data.valid_size/batch_size)
        else:
            total_batch = int(data.test_size/batch_size)
        full_data = True
    else: total_batch = total_batch

    acc = 0 
    loss = 0
    reg = 0
    for b_idx in range(total_batch):
        if is_valid:
            x_ = data.x_valid
            y_ = data.y_valid
        else:
            x_ = data.x_test
            y_ = data.y_test
        #batch_xs = keras_resnet_preprocess(x_[b_idx*batch_size:(b_idx+1)*batch_size])
        batch_xs = x_[b_idx*batch_size:(b_idx+1)*batch_size]
        batch_ys = y_[b_idx*batch_size:(b_idx+1)*batch_size]

        feed_dict = {
            graph_dict["data_holder"]: batch_xs,
            graph_dict["is_training"]: False
        }
        batch_pretrain = sess.run(fetches=model.pre_trained_cnn, feed_dict=feed_dict)
        #batch_xs = np.tile(batch_xs, [1,1,1,3])
        noise = np.random.normal(loc=0.0, scale=input_sigma, size=batch_pretrain.shape)
        feed_dict = {
            graph_dict["noise_holder"]: noise,
            graph_dict["noised_pretrain_holder"]: batch_pretrain+noise,
            graph_dict["label_holder"]: batch_ys,
            graph_dict["is_training"]: False
        }
        fetches = [model_loss, model_loss_reg, model_acc]
        feed_dict[graph_dict["sgd_sigma_holder"]] = 0.0
        feed_dict[graph_dict["trans_sigma_holder"]] = 0.0

        batch_loss, batch_reg, batch_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        acc += batch_acc
        reg += batch_reg
        loss += batch_loss
        
    acc /= total_batch
    reg /= total_batch
    loss /= total_batch

    # Print info
    print("Loss: {:.4f}, Reg loss: {:.4f}, Accuracy: {:.4f}".format(loss, reg, acc))
    print("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}".format(
        dp_info["eps"], dp_info["delta"], dp_info["total_sigma"], dp_info["input_sigma"]))
    
    with open(log_file, "a+") as file: 
        if full_data:
            file.write("FULL DATA\n")
        file.write("Loss: {:.4f}, Reg loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, reg, acc))
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
        total_batch = int(test_size/FLAGS.BATCH_SIZE)
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
        model = model_cifar100.RDPCNN(images_holder, label_holder, FLAGS.INPUT_SIGMA, is_training) # for adv examples

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
        test_info(sess, model, True, graph_dict, dp_info, FLAGS.TEST_LOG_FILENAME, 
            total_batch=total_test_batch)
        robust_info(sess, model, graph_dict, FLAGS.ROBUST_LOG_FILENAME)



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
        noised_pretrain_holder = tf.placeholder(tf.float32, [batch_size, 100])
        noise_holder = tf.placeholder(tf.float32, [batch_size, 100])
        label_holder = tf.placeholder(tf.float32, [batch_size, FLAGS.NUM_CLASSES])
        sgd_sigma_holder = tf.placeholder(tf.float32, ())
        trans_sigma_holder = tf.placeholder(tf.float32, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = model_cifar100.RDPCNN(data=data_holder, label=label_holder, input_sigma=input_sigma, is_training=is_training, noised_pretrain=noised_pretrain_holder, noise=noise_holder)
        priv_accountant = accountant.GaussianMomentsAccountant(data.train_size)
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant,
            [FLAGS.DP_GRAD_CLIPPING_L2NORM, True])

        # model training   
        model_loss = model.loss()
        model_loss_clean = model.loss_clean()
        model_loss_reg = model.loss_reg()
        # training
        #model_op, _, _, model_lr = model.optimization(model_loss)
        model_op, model_lr = model.dp_optimization([model_loss, model_loss_reg], gaussian_sanitizer, sgd_sigma_holder, trans_sigma_holder, FLAGS.BATCHES_PER_LOT, is_layerwised=FLAGS.IS_MGM_LAYERWISED)
        # analysis
        model_M, model_sens = model.compute_M_from_input_perturbation([model_loss_clean, model_loss_reg], FLAGS.DP_GRAD_CLIPPING_L2NORM, is_layerwised=FLAGS.IS_MGM_LAYERWISED)
        model_acc = model.cnn_accuracy


        graph_dict = {}
        graph_dict["data_holder"] = data_holder
        graph_dict["noised_pretrain_holder"] = noised_pretrain_holder
        graph_dict["noise_holder"] = noise_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["sgd_sigma_holder"] = sgd_sigma_holder
        graph_dict["trans_sigma_holder"] = trans_sigma_holder
        graph_dict["is_training"] = is_training

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        
        if FLAGS.load_pretrained:
            model.tf_load_pretrained(sess, name=FLAGS.PRETRAINED_CNN_CKPT_RESTORE_NAME)
        
        if FLAGS.load_model:
            model.tf_load(sess, name=FLAGS.CNN_CKPT_RESTORE_NAME)
        
        if FLAGS.local:
            total_train_lot = 2
            total_valid_lot = 2
        else:
            total_train_lot = int(data.train_size/batch_size/FLAGS.BATCHES_PER_LOT)
            total_valid_lot = None        
        
        print("Training...")
        itr_count = 0
        itr_start_time = time.time()
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
                lot_feeds = []
                lot_M = []
                for _ in range(FLAGS.BATCHES_PER_LOT):
                    #batch_xs = keras_resnet_preprocess(data.x_train[b_idx*batch_size:(b_idx+1)*batch_size])
                    batch_xs = data.x_train[b_idx*batch_size:(b_idx+1)*batch_size]
                    batch_ys = data.y_train[b_idx*batch_size:(b_idx+1)*batch_size]

                    feed_dict = {
                        data_holder: batch_xs,
                        is_training: True
                    }
                    batch_pretrain = sess.run(fetches=model.pre_trained_cnn, feed_dict=feed_dict)
                    #batch_xs = np.tile(batch_xs, [1,1,1,3])
                    noise = np.random.normal(loc=0.0, scale=input_sigma, size=batch_pretrain.shape)
                    feed_dict = {
                        noise_holder: noise,
                        noised_pretrain_holder: batch_pretrain+noise,
                        label_holder: batch_ys,
                        is_training: True
                    }
                    #import pdb; pdb.set_trace()
                    #batch_S_min = sess.run(fetches=model_S_min[0], feed_dict=feed_dict)
                    batch_M = sess.run(fetches=model_M, feed_dict=feed_dict)
                    #batch_S_min = compute_S_min_from_M(batch_M, FLAGS.IS_MGM_LAYERWISED)/FLAGS.DP_GRAD_CLIPPING_L2NORM
                    lot_feeds.append(feed_dict)
                    lot_M.append(batch_M)

                    b_idx += 1
                
                lot_M = sum(lot_M) / (FLAGS.BATCHES_PER_LOT**2)
                lot_S_min = compute_S_min_from_M(lot_M, FLAGS.IS_MGM_LAYERWISED)/FLAGS.DP_GRAD_CLIPPING_L2NORM
                #import pdb; pdb.set_trace()
                min_S_min = lot_S_min
                sigma_trans = input_sigma * min_S_min
                
                if sigma_trans >= FLAGS.TOTAL_DP_SIGMA:
                    sgd_sigma = 0.0
                else: 
                    sgd_sigma = FLAGS.TOTAL_DP_SIGMA - sigma_trans
                    sigma_trans = FLAGS.TOTAL_DP_SIGMA
                for feed_dict in lot_feeds:
                    # DP-SGD
                    feed_dict[sgd_sigma_holder] = sgd_sigma
                    feed_dict[trans_sigma_holder] = sigma_trans
                    sess.run(fetches=[model_op], feed_dict=feed_dict)

                itr_count += 1
                if itr_count > FLAGS.MAX_ITERATIONS:
                    terminate = True
               
                # for input transofrmation
                if train_idx % 1 == 0:
                    print("min S_min: ", min_S_min)
                    print("Sigma trans: ", sigma_trans)
                    print("Sigma grads: ", sgd_sigma)
                
                # optimization
                fetches = [model_loss, model_loss_reg, model_acc, model_lr]
                loss, reg, acc, lr = sess.run(fetches=fetches, feed_dict=feed_dict)
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
                    print("Loss: {:.4f}, Reg loss: {:.4f}, Accuracy: {:.4f}".format(loss, reg, acc))
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
                        file.write("Loss: {:.4f}, Reg loss: {:.4f},  Accuracy: {:.4f}\n".format(loss, reg, acc))
                        file.write("Total dp eps: {:.4f}, total dp delta: {:.8f}, total dp sigma: {:.4f}, input sigma: {:.4f}\n".format(
                            spent_eps_delta.spent_eps, spent_eps_delta.spent_delta, total_dp_sigma, input_sigma))
                        file.write("\n")
                
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
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-ep_start_time))
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
        valid_dict = test_info(sess, model, False, graph_dict, dp_info, FLAGS.TEST_LOG_FILENAME, total_batch=None)
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
