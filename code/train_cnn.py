import argparse, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import logging


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Model hyperparameters to be fed from outside this script
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--img-shape', type=str, default="200x200x3")

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])


    args, _ = parser.parse_known_args()
    
    # Input Params
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    img_shape = tuple([int(num) for num in args.img_shape.split("x")])

    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    test_dir = args.test

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info('.... .... .... Loading Data .... .... ....')

    X_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    X_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    X_test  = np.load(os.path.join(test_dir, 'test.npz'))['image']
    y_test  = np.load(os.path.join(test_dir, 'test.npz'))['label']

    ## reshape to (batch size, width, height, channels) for Tensorflow
    #X_train = X_train.reshape(X_train.shape[0], img_shape[0], img_shape[1], img_shape[2])
    #X_val = X_val.reshape(X_val.shape[0], img_shape[0], img_shape[1], img_shape[2])
    #X_test = X_test.reshape(X_test.shape[0], img_shape[0], img_shape[1], img_shape[2])

    K.set_image_data_format('channels_last')

    logger.info('.... .... .... Normalizing Data .... .... ....')
    
    # Normalizing the data
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_val /= 255
    X_test /= 255

    print(X_train.shape)

    ## load and edit a pre-trained VGG16 model for transfer learning:

    logger.info('.... .... .... Building Model .... .... ....')

    num_classes = 1            ## binary classification

    vgg_model=VGG16(weights="imagenet", input_shape=img_shape, include_top=False)

    model= Sequential()

    for layer in vgg_model.layers:
        model.add(layer)
    for layer in model.layers:
        layer.trainable=False

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(256,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))

    print(model.summary())

    optimizer = Adam(learning_rate=lr)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=[accuracy_score,
                  precision_score,
                  recall_score,
                  f1_score]
                 )

    logger.info('.... .... .... Setting GPUs .... .... ....')
    # convert the model to multi gpu model if gpu_count > 1
    if gpu_count > 1:

        model = multi_gpu_model(model, gpus=gpu_count)
        logger.info('.... .... Setting multi gpu model with ' + str(gpu_count) + ' gpus')



    # train model:
    logger.info('.... .... .... Training Model .... .... ....')
    model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_val, y_val), 
                        epochs=epochs)


    # Model metrics:
    logger.info('.... .... .... Evaluating Model Metrics .... .... ....')
    metrics = model.evaluate(X_test, y_test, verbose=0)

    print('Validation loss    :', metrics[0])
    print('Validation accuracy:', metrics[1])


    # save trained CNN Keras model to "model_dir" (path specificied earlier)
    logger.info('.... .... .... Saving Model .... .... ....')
    sess = K.get_session()

    tensorflow.saved_model.simple_save(
            sess,
            os.path.join(model_dir, 'model/1'),
            inputs={'inputs': model.input},
            outputs={t.name: t for t in model.outputs})

    logger.info('.... .... .... THE END! .... .... ....')

