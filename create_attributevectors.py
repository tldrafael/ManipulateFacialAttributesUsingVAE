import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import utils as ut
import training as tr


def mount_path(bname, im_dir=os.environ['celeba']):
    return os.path.join(im_dir, 'imgs', bname)


def load_celeba():
    fpath = os.path.join(os.environ['celeba'], 'list_attr_celeba.txt')
    df_celeba = pd.read_csv(fpath, sep='\s+', skiprows=[0])
    df_celeba.reset_index(inplace=True)
    df_celeba = df_celeba.rename(columns={'index': 'bname'})
    return df_celeba


def load_train_bnames():
    partitions_path = os.path.join(os.environ['celeba'], 'list_eval_partition.txt')
    list_eval_partition = pd.read_csv(partitions_path, sep='\s+', header=None)
    list_eval_partition.columns = ['bname', 'set_id']
    list_eval_partition['path'] = list_eval_partition['bname'].apply(mount_path)
    fpaths_train = list_eval_partition.query('set_id == 0')['path'].values.tolist()
    bnames_train = [f.split('/')[-1] for f in fpaths_train]
    return bnames_train


def create_attribute_vector(modelpath, df_celeba, attr, bnames_train):
    """
    As the training and predicting models are different their architecture, you have to adapt the trained weight file
    to the predict model.
    """
    trainedVAE = tr.VAE2train()
    trainedVAE.model.load_weights(modelpath)

    model_latent = tf.keras.models.Model(trainedVAE.model.inputs[0],
                                         trainedVAE.model.get_layer('z_sampling').output)

    if df_celeba.query('bname in @bnames_train').groupby(attr).size().min() < 10000:
        arg_replace = True
    else:
        arg_replace = False

    bnames_attr = df_celeba.query('bname in @bnames_train and {} == 1'.format(attr)).\
                            sample(10000, replace=arg_replace, axis=0)['bname'].values
    bnames_attr_not = df_celeba.query('bname in @bnames_train and {} == -1'.format(attr)).\
                                sample(10000, replace=arg_replace, axis=0)['bname'].values

    fpaths_attr = [mount_path(b) for b in bnames_attr]
    fpaths_attr_not = [mount_path(b) for b in bnames_attr_not]

    gen_attr = ut.InputGen(impaths=fpaths_attr, shuffle=False, loadsize_factor=10, bs=100, mode_predict=True)
    gen_attr_not = ut.InputGen(impaths=fpaths_attr_not, shuffle=False, loadsize_factor=10, bs=100, mode_predict=True)

    preds_attr = model_latent.predict(gen_attr.generator(), steps=100)
    preds_attr_not = model_latent.predict(gen_attr_not.generator(), steps=100)
    return preds_attr.mean(axis=0) - preds_attr_not.mean(axis=0)


if __name__ == '__main__':
    attr = sys.argv[1]
    modelpath = 'traindir/trained_113steps/checkpoints/weights.best.h5'
    df_celeba = load_celeba()
    # Use only the training samples to estimate the attribute vectors
    bnames_train = load_train_bnames()

    vec_attr = create_attribute_vector(modelpath, df_celeba, attr, bnames_train)

    outfile = os.path.join('cache', '{}.npy'.format(attr))
    np.save(outfile, vec_attr)