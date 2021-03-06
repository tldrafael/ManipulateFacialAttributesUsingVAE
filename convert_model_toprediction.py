import sys
import tensorflow as tf
import training as tr
import testing as tst


def get_predict_modelpath(modelpath):
    return modelpath.replace('.h5', '.predict.h5')


def save_weights_to_predictmodel(modelpath):
    """
    As the training and predicting models are different their architecture, you have to adapt the trained weight file
    to the predict model.
    """
    trainedVAE = tr.VAE2train()
    trainedVAE.model.load_weights(modelpath)

    tf.keras.backend.clear_session()
    predictVAE = tst.VAE2predict()
    for l in predictVAE.model.layers:
        try:
            trained_weights = trainedVAE.model.get_layer(l.name).get_weights()
            l.set_weights(trained_weights)
        except Exception as e:
            print('Not loading weights to {} - ERROR: {}'.format(l.name, e))

    predict_modelpath = get_predict_modelpath(modelpath)
    predictVAE.model.save_weights(predict_modelpath)


if __name__ == '__main__':
    save_weights_to_predictmodel(sys.argv[1])
