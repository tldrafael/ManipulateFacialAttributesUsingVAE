import os
import sys
import datetime
from glob import iglob
from skimage import io, transform
import tensorflow as tf
import numpy as np
import utils as ut
import training as tr


class VAE2predict:
    def __init__(self, use_sampling=False):
        self.use_sampling = use_sampling
        self._build_model()

    def _build_model(self):
        in_image = tf.keras.layers.Input(shape=(144, 144, 3), name='in_image')

        out_encoder = tr.Encoder()(in_image)

        x = tf.keras.layers.Dense(1024)(out_encoder)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('elu', name='out_latent_1')(x)

        z_mean = tf.keras.layers.Dense(2048, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(2048, name='z_logvar')(x)

        if self.use_sampling:
            z_latent = tf.keras.layers.Lambda(tr.sampling, output_shape=(2048,), name='z_sampling')([z_mean, z_log_var])
        else:
            z_latent = tf.keras.layers.Lambda(lambda x: x[0], output_shape=(2048,), name='z_sampling')([z_mean, z_log_var])

        outs = {}
        for (out_name, out_nc) in [('out_image_pre', 3), ('out_mask', 1)]:
            outs[out_name] = tr.Decoder()(z_latent, out_name, out_nc)

        # Tidy the image to use only the face regions of the estimated output and join with the original background
        x = tf.keras.layers.Multiply()([outs['out_image_pre'], outs['out_mask']])
        x_bg = tf.keras.layers.Multiply()([in_image, 1. - outs['out_mask']])
        out_image = tf.keras.layers.Add(name='out_image')([x, x_bg])

        self.model = tf.keras.models.Model(in_image, [out_image, outs['out_mask'], outs['out_image_pre']])

    def load_weights(self, modelpath=None, ckpt_dir=None):
        if ckpt_dir is None and modelpath is None:
            raise('Not possible to load the model')
            sys.exit()

        if ckpt_dir is not None:
            fpaths_weights = list(iglob(os.path.join(ckpt_dir, 'w*.h5')))
            fpaths_weights.sort()
            self.modelpath = fpaths_weights[-1]
        else:
            self.modelpath = modelpath

        self.model.load_weights(self.modelpath)

    def predict(self, X):
        if len(X.shape) == 3:
            X = X[None]

        return self.model.predict(X)

    def predict_path(self, paths):
        if isinstance(paths, str):
            paths = [paths]

        X = np.stack([ut.load_img(p) for p in paths])
        return self.predict(X)


def save_predictions(preds, org_dim=(144, 144)):
    now_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    now_timestamp = '.'
    dir_save = os.path.join('cache', now_timestamp)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    n_samples = preds[0].shape[0]
    for i in range(n_samples):
        for j, name in enumerate(['reconst', 'mask']):
            jpath = os.path.join(dir_save, '{}_{}.png'.format(i, name))
            Xsave = preds[j][i]
            if j == 1:
                # Round the mask pixels
                Xsave = Xsave.round()
            Xsave = transform.resize(Xsave, org_dim)
            io.imsave(jpath, Xsave)


if __name__ == '__main__':
    impath = sys.argv[1]
    im = io.imread(impath) / 255
    org_dim = im.shape[:-1]
    im = transform.resize(im, (144, 144))

    vae = VAE2predict()
    vae.load_weights(modelpath='traindir/trained_113steps/checkpoints/weights.best.predict.h5')
    X_pred = vae.predict(im)
    save_predictions(X_pred, org_dim=org_dim)
