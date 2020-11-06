import numpy as np
import random
import tensorflow as tf
import os
import scipy.misc as smc
import cv2

#-----------------------for 2D data read and save------------------------------------------
def randomFlipud(img, msk, u=0.5):
    if random.random() < u:
        img = np.flipud(img)
        msk = np.flipud(msk)
    return img.astype(np.float32), msk.astype(np.int32)

def randomFliplr(img, msk, u=0.5):
    if random.random() < u:
        img = np.fliplr(img)
        msk = np.fliplr(msk)
    return img.astype(np.float32), msk.astype(np.int32)

def randomRotate90(img, msk, u=0.5):
    if random.random() < u:
        img = np.rot90(img)
        msk = np.rot90(msk)
    return img.astype(np.float32), msk.astype(np.int32)

def randomCrop(img, msk, u=0.5):
    if random.random() < u:
        h,w = img.shape
        img2 = cv2.resize(img, (int(h * 1.2), int(w * 1.2)))
        img = img2[(int(h*1.2-h)//2) : (int(h*1.2-h)//2 + h), (int(w*1.2-w)//2) : (int(w*1.2-w)//2 + w)]
        msk2 = cv2.resize(msk.astype(np.float32), (int(h * 1.2), int(w * 1.2)))
        msk = msk2[(int(h * 1.2 - h) // 2): (int(h * 1.2 - h) // 2 + h),
               (int(w * 1.2 - w) // 2): (int(w * 1.2 - w) // 2 + w)]
    return img.astype(np.float32), msk.astype(np.int32)

def get_data_train_2d(trainPath, batchsize):
    vol_batch = []
    seg_batch = []
    for i in range(1, batchsize + 1):
        if i == 1:
            vol_batch, seg_batch = get_batch_train_2d(trainPath)
        else:
            vol_batch_tmp, seg_batch_tmp = get_batch_train_2d(trainPath)
            vol_batch = np.concatenate((vol_batch, vol_batch_tmp), axis=0)
            seg_batch = np.concatenate((seg_batch, seg_batch_tmp), axis=0)
    return vol_batch, seg_batch     #NHWC, NHWC

def get_batch_train_2d(trainPath):
    dirs_train = os.listdir(trainPath + 'img/')
    samples = random.choice(dirs_train)
    #print(samples)
    vol_batch = np.load(trainPath + 'img/' + samples)   # 128x128
    seg_batch = np.load(trainPath + 'seg/' + samples)   # 128x128
    # --- data augmentation-------------
    vol_batch, seg_batch = randomFlipud(vol_batch, seg_batch)
    vol_batch, seg_batch = randomFliplr(vol_batch, seg_batch)
    vol_batch, seg_batch = randomRotate90(vol_batch, seg_batch)
    vol_batch, seg_batch = randomCrop(vol_batch, seg_batch)

    vol_batch = np.expand_dims(vol_batch, axis=0)
    seg_batch = np.expand_dims(seg_batch, axis=0)
    vol_batch = np.expand_dims(vol_batch, axis=3)
    seg_batch = np.expand_dims(seg_batch, axis=3)
    vol_batch.astype(np.float32)
    seg_batch.astype(np.int32)
    return vol_batch, seg_batch     # 1x128x128x1, 1x128x128x1


def get_data_test_2d(testPath, tDir, batchsize):
    vol_batch = []
    seg_batch = []
    for i in range(1, batchsize+1):
        if i == 1:
            vol_batch, seg_batch = get_batch_test_2d(testPath, tDir, i-1)
        else:
            vol_batch_tmp, seg_batch_tmp = get_batch_test_2d(testPath, tDir, i-1)
            vol_batch = np.concatenate((vol_batch, vol_batch_tmp), axis = 0)
            seg_batch = np.concatenate((seg_batch, seg_batch_tmp), axis = 0)
    return vol_batch, seg_batch  # NHWC, NHWC

def get_batch_test_2d(testPath, tDir, ind):
    vol_batch = np.load(testPath + 'img/' + tDir[ind])
    seg_batch = np.load(testPath + 'seg/' + tDir[ind])
    vol_batch = np.expand_dims(vol_batch, axis = 0)
    seg_batch = np.expand_dims(seg_batch, axis = 0)
    vol_batch = np.expand_dims(vol_batch, axis=3)
    seg_batch = np.expand_dims(seg_batch, axis = 3)
    vol_batch.astype(np.float32)
    seg_batch.astype(np.int32)
    return vol_batch, seg_batch

def save_imgs(result_path, name_pre, label_batch, pred_batch, img_depth =1):
    # red is mask, blue is pred, green is pred*mask
    IMAGE_HEIGHT = label_batch.shape[-2]
    IMAGE_WIDTH = label_batch.shape[-1]
    str_split = name_pre.split('_')
    casePath = result_path + 'imgs/' +  str_split[0] + '/'
    # casePath = resultPath + 'imgs/' + str_split[1]  + '/'
    if not (os.path.exists(casePath)):
        os.makedirs(casePath)
    for dept in range(img_depth):
        label_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        label_slice = label_batch
        pred_slice = pred_batch

        label_cord = np.where(label_slice == 0)
        label_img_mat[0, label_cord[0], label_cord[1]] = 128
        label_img_mat[1, label_cord[0], label_cord[1]] = 128
        label_img_mat[2, label_cord[0], label_cord[1]] = 128

        label_cord = np.where(label_slice == 1)
        # blue, false negative, loujian
        label_img_mat[0, label_cord[0], label_cord[1]] = 50
        label_img_mat[1, label_cord[0], label_cord[1]] = 50
        label_img_mat[2, label_cord[0], label_cord[1]] = 250

        pred_cord = np.where(pred_slice == 1)
        # green, false positive, wujian
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 10
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 210
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 10

        pred_label = pred_slice * label_slice
        pred_cord = np.where(pred_label == 1)
        label_img_mat[0, pred_cord[0], pred_cord[1]] = 210
        label_img_mat[1, pred_cord[0], pred_cord[1]] = 10
        label_img_mat[2, pred_cord[0], pred_cord[1]] = 10

        label_img_mat = np.transpose(label_img_mat, [1, 2, 0])

        smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(
            casePath + str_split[1] + '-seg.png')
        # cv2.imwrite(casePath + str_split[1] + '-seg.png', label_img_mat)


def save_npys(res_path, name_pre, label_batch, pred_batch, score_batch= None):
    str_split = name_pre.split('_')
    casePath = res_path + 'npys/' + str_split[0] + '/'
    if not (os.path.exists(casePath)):
        os.makedirs(casePath)
    np.save(casePath + str_split[1] + '-mask.npy', label_batch)
    np.save(casePath + str_split[1] + '-pred.npy', pred_batch)
    if score_batch is not None:
        np.save(casePath + str_split[1] + '-score.npy', score_batch)

def labels_to_onehot(lables, class_num = 1):
    '''
    :param lables:  shape [batchsize, depth, height, width], 4D, no channel axis
    :param class_num:
    :return:
    '''

    if isinstance(class_num, tf.Tensor):
        class_num_tf = tf.to_int32(class_num)
    else:
        class_num_tf = tf.constant(class_num, tf.int32)
    in_shape = tf.shape(lables)
    out_shape = tf.concat([in_shape, tf.reshape(class_num_tf, (1,))], 0) # add a extra axis for classNum, 5D

    if class_num == 1:
        return tf.reshape(lables, out_shape)
    else:
        lables = tf.reshape(lables, (-1,)) # squeeze labels to one row x N cols vector [0,0,0,1,......]
        dense_shape = tf.stack([tf.shape(lables)[0], class_num_tf], 0)   # denshape [N cols , classNum]

        lables = tf.to_int64(lables)
        ids = tf.range(tf.to_int64(dense_shape[0]), dtype= tf.int64)  # ids is a 1xN vector as[0,1,2,3...., N-1]
        ids = tf.stack([ids, lables], axis= 1)  #ids is N x clsNum mat
        one_hot = tf.SparseTensor(indices= ids, values= tf.ones_like(lables, dtype= tf.float32), dense_shape = tf.to_int64(dense_shape))
        one_hot = tf.sparse_reshape(one_hot, out_shape)
        return tf.cast(one_hot, tf.float32)


def to_onehot(lables, class_num = 1):

    one_hot = tf.one_hot(indices= lables, depth= class_num,
                        on_value= 1.0, off_value= 0.0, axis= -1, dtype= tf.float32)  #one_hot shape [batch, d, h, w, channel] 5D
    return one_hot