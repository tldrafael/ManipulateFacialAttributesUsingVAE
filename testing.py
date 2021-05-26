import os
import sys
from glob import iglob
from skimage import io
import tensorflow as tf
import numpy as np
import utils as ut
import training as tr


class VAE2predict:
    def __init__(self):
        self._build_model()

    def _build_model(self, print_summary=False):
        in_image = tf.keras.layers.Input(shape=(144, 144, 3), name='in_image')

        out_encoder = tr.Encoder()(in_image)

        x = tf.keras.layers.Dense(1024)(out_encoder)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('elu', name='out_latent_1')(x)

        z_mean = tf.keras.layers.Dense(2048, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(2048, name='z_logvar')(x)
        z_latent = tf.keras.layers.Lambda(tr.sampling, output_shape=(2048,), name='z_sampling')([z_mean, z_log_var])

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


if __name__ == '__main__':
    impath = sys.argv[1]

    vae = VAE2predict()
    vae.load_weights(modelpath='traindir/trained_113steps/checkpoints/weights.best.predict.h5')
    X_pred = vae.predict_path(im)
