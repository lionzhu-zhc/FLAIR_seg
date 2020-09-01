# unet++

import tensorflow as tf

def standard_unit(x, filters,BN_FLAG, kernel_size=3, padding="same", strides=1,):
    conv = tf.layers.conv2d(x, filters= filters, kernel_size= kernel_size, strides= 1,
                            padding= 'same', activation='relu', name='conv1')
    conv = tf.layers.batch_normalization(conv, momentum=0.9, training=BN_FLAG, name='BN1')
    conv = tf.layers.conv2d(conv, filters=filters, kernel_size=kernel_size, strides=1,
                            padding='same', activation='relu', name='conv2')
    c = tf.layers.batch_normalization(conv, momentum=0.9, training=BN_FLAG, name='BN2')
    p = tf.nn.max_pool(c, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')
    return c, p

def upsampling_unit(x, skip, filters, BN_FLAG):
    us = tf.layers.conv2d_transpose(x, filters= filters, kernel_size=2, strides=(2,2), padding='same', name='deconv')
    us = tf.layers.batch_normalization(us, momentum=0.9, training=BN_FLAG, name='BN')
    concat = tf.concat([us, skip], axis=-1)
    c, _ = standard_unit(concat, filters)
    return c

def bottleneck(x, filters, BN_FLAG, kernel_size=(3,3), padding ='same', strides=1):
    c = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding= padding, activation='relu', name='conv1')
    c = tf.layers.batch_normalization(c, momentum=0.9, training= BN_FLAG, name='BN1')
    c = tf.layers.conv2d(c, filters, kernel_size, strides=strides, padding=padding, activation='relu', name='conv2')
    c = tf.layers.batch_normalization(c, momentum=0.9, training=BN_FLAG, name='BN2')
    return c

def unetpp(tensor_in, BN_FLAG, CLASSNUM):
    f = [16, 32, 64, 128, 256]
    img_input = tensor_in

    with tf.variable_scope('c11'):
        c11, p1 = standard_unit(img_input, f[0], BN_FLAG)
    with tf.variable_scope('c21'):
        c21, p2 = standard_unit(p1, f[1], BN_FLAG)
    with tf.variable_scope('c31'):
        c31, p3 = standard_unit(p2, f[2], BN_FLAG)
    with tf.variable_scope('c41'):
        c41, p4 = standard_unit(p3, f[3], BN_FLAG)
    with tf.variable_scope('c51'):
        c51, _ = standard_unit(p4, f[4], BN_FLAG)

    with tf.variable_scope('u12'):
        u12 = tf.layers.conv2d_transpose(c21, filters=f[0], kernel_size=(2,2), strides=(2,2), padding='same', name='convu12')
        u12 = tf.layers.batch_normalization(u12, momentum=0.9, training=BN_FLAG, name='BNu12')
        concat = tf.concat([u12,c11], axis=-1)
        c12, _ = standard_unit(concat, f[0], BN_FLAG)

    with tf.variable_scope('u22'):
        u22 = tf.layers.conv2d_transpose(c31, filters=f[1], kernel_size=(2,2), strides=(2,2), padding='same', name='convu22')
        u22 = tf.layers.batch_normalization(u22, momentum=0.9, training=BN_FLAG, name='BNu22')
        concat = tf.concat([u22, c21], axis=-1)
        c22, _=standard_unit(concat, f[1], BN_FLAG)

    with tf.variable_scope('u32'):
        u32 = tf.layers.conv2d_transpose(c41, filters=f[2], kernel_size=(2, 2), strides=(2, 2), padding='same', name='convu32')
        u32 = tf.layers.batch_normalization(u32, momentum=0.9, training=BN_FLAG, name='BNu32')
        concat = tf.concat([u32, c31], axis=-1)
        c32, _ = standard_unit(concat, f[2], BN_FLAG)

    with tf.variable_scope('u42'):
        u42 = tf.layers.conv2d_transpose(c51, filters=f[3], kernel_size=(2, 2), strides=(2, 2), padding='same', name='convu42')
        u42 = tf.layers.batch_normalization(u42, momentum=0.9, training=BN_FLAG, name='BNu42')
        concat = tf.concat([u42, c41], axis=-1)
        c42, _ = standard_unit(concat, f[3], BN_FLAG)

    with tf.variable_scope('u13'):
        u13 = tf.layers.conv2d_transpose(c22, filters=f[0], kernel_size=(2, 2), strides=(2, 2), padding='same', name='convu13')
        u13 = tf.layers.batch_normalization(u13, momentum=0.9, training=BN_FLAG, name='BNu13')
        concat = tf.concat([u13, c11, c12], axis=-1)
        c13, _ = standard_unit(concat, f[0], BN_FLAG)

    with tf.variable_scope('u23'):
        u23 = tf.layers.conv2d_transpose(c32, filters=f[1], kernel_size=(2, 2), strides=(2, 2), padding='same', name='convu23')
        u23 = tf.layers.batch_normalization(u23, momentum=0.9, training=BN_FLAG, name='BNu23')
        concat = tf.concat([u23, c21, c22], axis=-1)
        c23, _ = standard_unit(concat, f[1], BN_FLAG)

    with tf.variable_scope('u33'):
        u33 = tf.layers.conv2d_transpose(c42, filters=f[1], kernel_size=(2, 2), strides=(2, 2), padding='same', name='convu33')
        u33 = tf.layers.batch_normalization(u33, momentum=0.9, training=BN_FLAG, name='BNu33')
        concat = tf.concat([u33, c31, c32], axis=-1)
        c33, _ = standard_unit(concat, f[2], BN_FLAG)

    with tf.variable_scope('u14'):
        u14 = tf.layers.conv2d_transpose(c23, filters=f[0], kernel_size=(2,2), strides=(2,2), padding='same', name='convu14')
        u14 = tf.layers.batch_normalization(u14, momentum=0.9, training=BN_FLAG, name='BNu14')
        concat = tf.concat([u14, c11, c12, c13], axis=-1)
        c14,_ = standard_unit(concat, f[0], BN_FLAG)

    with tf.variable_scope('u24'):
        u24 = tf.layers.conv2d_transpose(c33, filters=f[1], kernel_size=(2, 2), strides=(2, 2), padding='same', name='convu24')
        u24 = tf.layers.batch_normalization(u24, momentum=0.9, training=BN_FLAG, name='BNu24')
        concat = tf.concat([u24, c21, c22, c23], axis=-1)
        c24, _ = standard_unit(concat, f[1], BN_FLAG)

    with tf.variable_scope('u15'):
        u15 = tf.layers.conv2d_transpose(c24, filters=f[0], kernel_size=(2, 2), strides=(2, 2), padding='same', name='convu15')
        u15 = tf.layers.batch_normalization(u15, momentum=0.9, training=BN_FLAG, name='BNu15')
        concat = tf.concat([u15, c11,c12,c13,c14], axis=-1)
        c15, _ = standard_unit(concat, f[0], BN_FLAG)

    with tf.variable_scope('output'):
        with tf.variable_scope('output1'):
            output1 = tf.layers.conv2d(c12, filters=CLASSNUM, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                                       padding='same', name='conv1')
        with tf.variable_scope('output2'):
            output2 = tf.layers.conv2d(c13, filters=CLASSNUM, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                                       padding='same', name='conv1')
        with tf.variable_scope('output3'):
            output3 = tf.layers.conv2d(c14, filters=CLASSNUM, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                                       padding='same', name='conv1')
        with tf.variable_scope('output4'):
            output4 = tf.layers.conv2d(c15, filters=CLASSNUM, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same',
                                       name='conv1')

        annot_pred = tf.argmax(output4, axis=3, name='pred')
    return output4, tf.expand_dims(annot_pred, axis= 3)
