import os
import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import utils as ut


class ConvBlock:
    def __init__(self, n_filters=64, filter_size=(3, 3), strides=(1, 1), padding='same', activation='elu', use_bn=True):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

    def __call__(self, x):
        x = tf.keras.layers.Conv2D(self.n_filters, self.filter_size, strides=self.strides, padding=self.padding)(x)
        if self.use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation(self.activation)(x)


class ConvBlockTranspose:
    def __init__(self, n_filters=64, filter_size=(3, 3), strides=(2, 2), padding='same', activation='elu', use_bn=True):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

    def __call__(self, x):
        x = tf.keras.layers.Conv2DTranspose(self.n_filters, self.filter_size, strides=self.strides,
                                            padding=self.padding)(x)
        if self.use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation(self.activation)(x)


class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, add_mae=False):
        super(SSIMLoss, self).__init__(name='ssim')
        self.add_mae = add_mae
        self.tf_mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss_ssim = 2 - tf.image.ssim(y_true, y_pred, 1., filter_size=9)
        loss_ssim = tf.math.reduce_mean(loss_ssim)
        if self.add_mae:
            loss_ssim = loss_ssim + 5 * self.tf_mae(y_true, y_pred)

        return tf.math.reduce_mean(loss_ssim)


class BCELoss(tf.keras.losses.Loss):
    def __init__(self):
        self.tf_bce = tf.keras.losses.BinaryCrossentropy()

    def dice_coef(self, y_true, y_pred, smooth=1):
        intersection = tf.math.abs(y_true * y_pred)
        intersection = tf.math.reduce_sum(intersection)
        total_area = tf.math.square(y_true) + tf.math.square(y_pred)
        total_area = tf.math.reduce_sum(total_area)
        return 1 - (2. * intersection + smooth) / (total_area + smooth + 1e-8)

    def __call__(self, y_true, y_pred):
        return self.tf_bce(y_true, y_pred) + self.dice_coef(y_true, y_pred)


def decay_schedule(epoch, learning_rate):
    if epoch > 1 and epoch % 9 == 0 and learning_rate <= 1e-5:
        learning_rate /= 3
    return learning_rate


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean), name='get_epsilon')
    deviation = tf.math.exp(.5 * z_log_var, name='apply_epsilon')
    offset = tf.math.multiply(deviation, epsilon, name='mult_eps')
    # offset = 0
    return z_mean + offset


class KLLoss:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, z, mean, logvar):
        loss = 1 + 2 * logvar - tf.math.square(mean) - tf.math.exp(2 * logvar)
        return - self.factor * .5 * tf.math.reduce_mean(loss)


class Encoder:
    def __init__(self):
        pass

    def __call__(self, x_input):
        x = ConvBlock(n_filters=32)(x_input)
        x = ConvBlock(n_filters=32)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=64)(x)
        x = ConvBlock(n_filters=64)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=128)(x)
        x = ConvBlock(n_filters=128)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=256)(x)
        x = ConvBlock(n_filters=256)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=512)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=512)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        return tf.keras.layers.Flatten(name='out_encoder')(x)


class Decoder:
    def __init__(self):
        pass

    def __call__(self, z_latent, out_name, out_nc):
        x = tf.keras.layers.Reshape((2, 2, 512))(z_latent)
        x = ConvBlockTranspose(padding='same')(x)
        x = ConvBlock(512)(x)
        x = ConvBlockTranspose(padding='valid')(x)
        x = ConvBlock(512)(x)
        x = ConvBlockTranspose(padding='same')(x)
        x = ConvBlock(256)(x)
        x = ConvBlock(256)(x)
        x = ConvBlockTranspose(padding='same')(x)
        x = ConvBlock(128)(x)
        x = ConvBlock(128)(x)
        x = ConvBlockTranspose(padding='same')(x)
        x = ConvBlock(64)(x)
        x = ConvBlock(64)(x)
        x = ConvBlockTranspose(padding='same')(x)
        x = ConvBlock(32)(x)
        x = ConvBlock(32)(x)
        return tf.keras.layers.Conv2D(out_nc, (3, 3), strides=(1, 1), activation='sigmoid',
                                      padding='same', name=out_name)(x)


class VAE2train:
    def __init__(self, factor_kl=1e-3):
        self.factor_kl = factor_kl
        self._build_model()

    def _build_model(self, print_summary=False):
        tf.keras.backend.clear_session()
        in_image = tf.keras.layers.Input(shape=(144, 144, 3), name='in_image')
        in_mask = tf.keras.layers.Input(shape=(144, 144, 1), name='in_mask')

        out_encoder = Encoder()(in_image)

        x = tf.keras.layers.Dense(1024)(out_encoder)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('elu', name='out_latent_1')(x)

        z_mean = tf.keras.layers.Dense(2048, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(2048, name='z_logvar')(x)
        z_latent = tf.keras.layers.Lambda(sampling, output_shape=(2048,), name='z_sampling')([z_mean, z_log_var])

        outs = {}
        for (out_name, out_nc) in [('out_image_pre', 3), ('out_mask', 1)]:
            outs[out_name] = Decoder()(z_latent, out_name, out_nc)

        # Tidy the image to use only the face regions of the estimated output and join with the original background
        x = tf.keras.layers.Multiply()([outs['out_image_pre'], in_mask])
        x_bg = tf.keras.layers.Multiply()([in_image, 1. - in_mask])
        out_image = tf.keras.layers.Add(name='out_image')([x, x_bg])

        self.model = tf.keras.models.Model([in_image, in_mask], [out_image, outs['out_mask'], outs['out_image_pre']])

        if print_summary:
            self.model.summary()

        loss_KL = KLLoss(factor=self.factor_kl)(z_latent, z_mean, z_log_var)
        self.model.add_loss(loss_KL)
        self.model.add_metric(loss_KL, name='kl_div', aggregation='mean')

        loss_face_ssim = SSIMLoss()(in_image, out_image)
        self.model.add_loss(loss_face_ssim)
        self.model.add_metric(loss_face_ssim, name='ssim_face', aggregation='mean')

        loss_face_mae = 4 * tf.keras.losses.MeanAbsoluteError()(in_image, out_image)
        self.model.add_loss(loss_face_mae)
        self.model.add_metric(loss_face_mae, name='mae_face', aggregation='mean')

        loss_mask_bce = BCELoss()(in_mask, outs['out_mask'])
        self.model.add_loss(loss_mask_bce)
        self.model.add_metric(loss_mask_bce, name='bce_mask', aggregation='mean')

    def train(self, gen_train, gen_val, traindir=None):
        if traindir is None:
            self.traindir = os.path.join('traindir', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.traindir = traindir

        lr_scheduler = LearningRateScheduler(decay_schedule)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, restore_best_weights=True)

        self.logdir = os.path.join(self.traindir, 'logs')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir, exist_ok=True)

        self.ckptdir = os.path.join(self.traindir, 'checkpoints')
        if not os.path.exists(self.ckptdir):
            os.makedirs(self.ckptdir)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(self.logdir, histogram_freq=1)

        checkpoint_fname = ('weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}-{val_ssim_face:.2f}-{val_mae_face:.2f}'
                            ' -{val_bce_mask:.2f}-{val_kl_div:.2f}.h5')
        checkpoint = ModelCheckpoint(os.path.join(self.ckptdir, checkpoint_fname), save_weights_only=True,
                                     save_format="tf", monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1e-3))

        self.model.fit(gen_train.generator(),
                   validation_data=gen_val.generator(),
                   #verbose=1,
                   #steps_per_epoch=4000,
                   #validation_steps=700,
                   #epochs=2000,
                   #callbacks=[lr_scheduler, checkpoint, earlystop, tensorboard_callback],
                   callbacks=[checkpoint],
                   steps_per_epoch=10,
                   validation_steps=2,
                   epochs=2,
                   )


if __name__ == '__main__':
    celeba_dir = os.environ['celeba']
    list_eval_partition = pd.read_csv(os.path.join(celeba_dir, 'list_eval_partition.txt'), sep='\s+', header=None)
    list_eval_partition.columns = ['bname', 'set_id']
    list_eval_partition['path'] = list_eval_partition['bname'].apply(lambda x: os.path.join(celeba_dir, 'imgs', x))

    fpaths_train = list_eval_partition.query('set_id == 0')['path'].values.tolist()
    fpaths_val = list_eval_partition.query('set_id == 1')['path'].values.tolist()
    fpaths_test = list_eval_partition.query('set_id == 2')['path'].values.tolist()

    np.random.seed(5)
    for fset in [fpaths_train, fpaths_val, fpaths_test]:
        np.random.shuffle(fset)

    gen_train = ut.InputGen(impaths=fpaths_train, loadsize_factor=2)
    gen_val = ut.InputGen(impaths=fpaths_val, loadsize_factor=2)
    vae = VAE2train()
    vae.train(gen_train, gen_val)
