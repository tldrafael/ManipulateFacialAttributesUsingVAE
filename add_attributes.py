import argparse
import os
import sys
import datetime
from glob import iglob
from skimage import io, transform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import training as tr


class VAE2AddAttr:
    def __init__(self, use_sampling=False):
        self.use_sampling = use_sampling
        self._build_model()

    def _build_model(self):
        in_image = tf.keras.layers.Input(shape=(144, 144, 3), name='in_image')
        in_attribute = tf.keras.layers.Input(shape=(2048,), name='in_attr')

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

        z_addattr = tf.keras.layers.Add(name='in_decoder_pos')([z_latent, in_attribute])

        outs = {}
        for (out_name, out_nc) in [('out_image_pre', 3), ('out_mask', 1)]:
            if out_name == 'out_image_pre':
                outs[out_name] = tr.Decoder()(z_addattr, out_name, out_nc)
            else:
                outs[out_name] = tr.Decoder()(z_latent, out_name, out_nc)

        # Tidy the image to use only the face regions of the estimated output and join with the original background
        x = tf.keras.layers.Multiply()([outs['out_image_pre'], outs['out_mask']])
        x_bg = tf.keras.layers.Multiply()([in_image, 1. - outs['out_mask']])
        out_image = tf.keras.layers.Add(name='out_image')([x, x_bg])

        self.model = tf.keras.models.Model([in_image, in_attribute],
                                           [out_image, outs['out_mask'], outs['out_image_pre']])

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
        return self.model.predict(X)

    def add_attribute(self, X, X_attr, attr_factor=1.):
        return self.predict([X, X_attr * attr_factor])

    def predict_spectrum(self, X, X_attr, boundaries=(-2, 2), nsamples=5):
        attr_factors = np.linspace(boundaries[0], boundaries[1], nsamples)
        return [self.add_attribute(X, X_attr, f)[0] for f in attr_factors], attr_factors


def save_spetrum(im, preds, attr_factors, org_dim=(144, 144)):
    now_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    now_timestamp = '.'
    dir_save = os.path.join('cache', now_timestamp)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    ncol = len(preds) + 1
    fig, axs = plt.subplots(1, ncol, figsize=(3 * ncol, 6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    [a.set_axis_off() for a in axs]
    axs[0].imshow(im)
    for a, p, f in zip(axs[1:], preds, attr_factors):
        a.imshow(p[0])
        a.set_title(f)

    fig.savefig(os.path.join(dir_save, 'spectrum.jpg'), bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, help='attribute', dest='attr')
    parser.add_argument('-f', type=str, help='filepath', dest='impath')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    im = io.imread(args.impath) / 255
    org_dim = im.shape[:-1]
    im = transform.resize(im, (144, 144))

    vae = VAE2AddAttr()
    vae.load_weights(modelpath='traindir/trained_113steps/checkpoints/weights.best.predict.h5')

    vec_attr = np.load(ut.vec_attrs_path[args.attr])
    X_pred, attr_factor = vae.predict_spectrum(im[None], vec_attr[None])
    save_spetrum(im, X_pred, attr_factor, org_dim)
