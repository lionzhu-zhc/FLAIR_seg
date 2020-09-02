import tf.nets.unetpp as Unetpp
import tf.nets.loss as Loss
import tf.utils as utils
import os
import tensorflow as tf
import math
import numpy as np

Server = 0
#---------------------paths--------------------------------------------------
if Server == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = 'D:\datasets\diyiyiyuan\DWIFLAIR/flair_npy2d_all/'
    trainPath = path + 'train/'
    testPath = path + 'test/'
    TotalNum = len(os.listdir(trainPath + 'img/'))
    out_path = path + 'exps/tf_exp1/'
    npy_path = out_path + 'npys/'
    ckpt_path = out_path + 'ckpt/'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

elif Server == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = '/opt/zhc/dwi_flair/flair_npy2d_all/'
    out_path = path + 'exps/exp4/'
    npy_path = out_path + 'npys/'
    ckpt_path = out_path + 'ckpt/'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

#-Img paras------------------------------------------------------------------------------------------------------------
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNEL = 1
CLASSNUM = 2

#-training paras--------------------------------------------------------------------------------------------------------
EPOCH = 300
TRAIN_BATCHSIZE = 32
LEARNING_RATE = 1e-3

ITER_PER_EPOCH = TotalNum // TRAIN_BATCHSIZE
DECAY_INTERVAL = ITER_PER_EPOCH * EPOCH // 15
MAX_ITERATION = ITER_PER_EPOCH * EPOCH
# MAX_ITERATION = 10
SAVE_CKPT_INTERVAL = ITER_PER_EPOCH * EPOCH // 2

ValidFlag = False
TestFlag = True

def early_training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)

def late_training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)

def FCNX_run():
    with tf.name_scope('inputs'):
        annotation = tf.placeholder(tf.int32, shape=[TRAIN_BATCHSIZE, None, None, 1],
                                    name='annotation')   # shape BHWC
        image = tf.placeholder(tf.float32, shape=[TRAIN_BATCHSIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL],
                               name='image')   # shape BHWC

    bn_flag = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    logits, pred_annot = Unetpp.unetpp(tensor_in=image, BN_FLAG=bn_flag, CLASSNUM=CLASSNUM)

    with tf.variable_scope('loss'):
        class_weight = tf.constant([0.1, 2])
        loss_reduce = Loss.cross_entropy_loss(pred= logits, ground_truth= annotation, class_weight = class_weight)
        # loss_reduce = LossPy.iou_loss(pred = logits, target= annotation, class_num= CLASSNUM)

    with tf.variable_scope('valid_IOU'):
        iou = tf.placeholder(tf.float32)
        tf.summary.scalar('IOU', iou)

    with tf.variable_scope('trainOP'):
        LRate = tf.placeholder(tf.float32)
        trainable_vars = tf.trainable_variables()
        train_op = early_training(LRate, loss_reduce, trainable_vars)
        tf.summary.scalar('lr', LRate)

    with tf.variable_scope('fcnx') as scope:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config = config)
        print('Begin training:.......')

        merge_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(out_path + '/log/train', sess.graph)
        valid_writer = tf.summary.FileWriter(out_path + '/log/valid')
        sess.run(tf.global_variables_initializer())
        scope.reuse_variables()
        saver = tf.train.Saver()

        global LEARNING_RATE
        meanIOU = 0.001
        for itr in range(MAX_ITERATION):
            vol_batch, seg_batch = utils.get_data_train_2d(trainPath, batchsize=TRAIN_BATCHSIZE)

            # -changed learning rate ------------------------------------------------------------------------------------
            if (itr + 1) % DECAY_INTERVAL == 0:
                LEARNING_RATE = LEARNING_RATE * 0.90
                print('learning_rate:', LEARNING_RATE)

            # -validation with IOU each 10 epoch------------------------------------------------------------------------
            if (itr + 1) % (ITER_PER_EPOCH * 10) == 0 and ValidFlag:
                test_dirs = os.listdir(testPath + '/vol/')
                one_pred_or_label = one_label_and_pred = 0
                test_num = len(test_dirs)
                if test_num < TRAIN_BATCHSIZE:
                    for i in range(TRAIN_BATCHSIZE - test_num):
                        test_dirs.append(test_dirs[i])
                    test_num = TRAIN_BATCHSIZE
                test_times = math.ceil(test_num / TRAIN_BATCHSIZE)
                for i in range(test_times):
                    if i != (test_times - 1):
                        tDir = test_dirs[i * TRAIN_BATCHSIZE: (i + 1) * TRAIN_BATCHSIZE]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    if i == (test_times - 1):
                        tDir = test_dirs[(test_num - TRAIN_BATCHSIZE): test_num]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    test_feed = {image: vol_batch, annotation: seg_batch, bn_flag: False, keep_prob: 1}
                    test_pred_annotation = sess.run(pred_annot, feed_dict=test_feed)
                    for j in range(TRAIN_BATCHSIZE):
                        label_batch = np.squeeze(seg_batch[j, ...]).astype(np.uint8)
                        pred_batch = np.squeeze(test_pred_annotation[j, ...]).astype(np.uint8)
                        label_bool = (label_batch == 1)
                        pred_bool = (pred_batch == 1)
                        union = np.logical_or(label_bool, pred_bool)
                        intersection = np.logical_and(label_bool, pred_bool)
                        one_pred_or_label = one_pred_or_label + np.count_nonzero(union)
                        one_label_and_pred = one_label_and_pred + np.count_nonzero(intersection)
                meanIOU = one_label_and_pred / (one_pred_or_label + 1e-4)
                print('valid meanIOU', meanIOU)

            # -training training training training-----------------------------------------------------------------------
            feed = {LRate: LEARNING_RATE, iou: meanIOU, image: vol_batch, annotation: seg_batch,
                    bn_flag: True, keep_prob: 1}
            sess.run(train_op, feed_dict=feed)
            train_loss_print, summary_str = sess.run([loss_reduce, merge_op], feed_dict=feed)
            train_writer.add_summary(summary_str, itr)
            print(itr, '|', MAX_ITERATION)
            print('loss:', train_loss_print)

            if (itr + 1) % SAVE_CKPT_INTERVAL == 0:
                saver.save(sess, out_path + 'ckpt/modle', global_step= (itr+1))

            #-Test Test Test Test---------------------------------------------------------------------------------------
            if itr == (MAX_ITERATION - 1) and TestFlag:
                print('End training:.......')
                test_dirs = os.listdir(testPath + '/img/')
                test_num = len(test_dirs)

                if test_num < TRAIN_BATCHSIZE:
                    for i in range(TRAIN_BATCHSIZE - test_num):
                        test_dirs.append(test_dirs[i])
                    test_num = TRAIN_BATCHSIZE
                test_times = int(math.ceil(test_num / TRAIN_BATCHSIZE))
                for i in range(test_times):
                    if i != (test_times - 1):
                        tDir = test_dirs[i * TRAIN_BATCHSIZE : (i+1) * TRAIN_BATCHSIZE]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    if i == (test_times - 1):
                        tDir = test_dirs[(test_num-TRAIN_BATCHSIZE) : test_num]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    test_feed = {image: vol_batch, annotation: seg_batch, bn_flag: False, keep_prob:1}
                    test_logits, test_pred_annotation = sess.run([logits, pred_annot], feed_dict=test_feed)
                    test_score = test_logits[..., 1]
                    for j in range(TRAIN_BATCHSIZE):
                        label_batch = np.squeeze(seg_batch[j,...])
                        pred_batch = np.squeeze(test_pred_annotation[j,...])
                        score_batch = np.squeeze(test_score[j, ...])
                        namePre = tDir[j]
                        namePre = namePre[:-4]
                        print("test_itr:", namePre)
                        # utils.save_imgs_zd(resultPath, namePre, label_tosave, pred_tosave)
                        utils.save_imgs(out_path, namePre, label_batch, pred_batch)
                        utils.save_npys(out_path, namePre, label_batch, pred_batch)

        train_writer.close()
        valid_writer.close()
        
if __name__ == '__main__':
    print("Begin...")
    FCNX_run()
    print("Finished!")