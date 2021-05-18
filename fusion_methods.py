from keras.layers import Input, BatchNormalization, Dense, Dropout, Lambda, Reshape, Conv2D, Flatten, Conv2DTranspose, Add, Concatenate, Multiply
from keras.models import Model, load_model
from keras.layers.core import Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K

import spatial_stream_weights.utils as ut
import matplotlib.pyplot as plt
import cv2
import joblib
import numpy as np


def generate_dataset(features, grid_size, shuffle=True, audio_weight=1, video_weight=1, context=0):
    small_constant = 1e-6

    if context == 0:
        if shuffle:
            np.random.shuffle(features)
        video_features = np.array([feat[0]*video_weight for feat in features])
        video_features[video_features == 0] = small_constant
        video_features = np.nan_to_num(video_features, nan=small_constant)

        audio_features = np.array([feat[1]*audio_weight for feat in features])
        audio_features[audio_features == 0] = small_constant
        video_features = np.nan_to_num(video_features, nan=small_constant)

        index_target = np.array([feat[2] for feat in features])

        position_target = []
        for feat in features:
            target = np.zeros(grid_size)
            for point in feat[3]:
                if not np.all(point == 0):
                    target[(int(point[0]), int(point[1]))] += 1
            position_target.append(np.expand_dims(target, 2))
        position_target = np.array(position_target)
        position_target[position_target == 0] = small_constant
        # position_target = np.array([feat[3].flatten() for feat in features])
    else:
        raise NotImplementedError("Adding Context Information is not yet implemented")

        # Sorting Features by sequence no.
        sequenced_features = []
        part_list = []
        current_sequence = features[0][4]
        for feat in features:
            if feat[4] == current_sequence:
                part_list.append(feat)
            else:
                current_sequence = feat[4]
                sequenced_features.append(part_list.copy())
                part_list = []

        # Splitting sequences in even chunks
        splitted_features = []
        for sequence in sequenced_features:
            splitted_features.extend(chunks(sequence, context))

        if shuffle:
            np.random.shuffle(splitted_features)
        video_features = []
        audio_features = []
        index_target = []
        position_target = []
        i = 0
        for chunk in splitted_features:
            # Extracting video features
            video_features.append(np.array([frame[0] * video_weight for frame in chunk]))

            # Extracting audio features
            audio_features.append(np.array([frame[1] * audio_weight for frame in chunk]))

            # Extracting speaker number
            index_target.append(chunk[context][2])

            # Extracting target
            target = np.zeros(grid_size)
            for point in chunk[context][3]:
                if not np.all(point == 0):
                    target[(int(point[0]), int(point[1]))] += 1
            position_target.append(target.flatten())

            i += 1
            print("\rcalculated: {0:.2f}%".format(100 * i / len(splitted_features)), end='')

        # Converting Lists to arrays
        video_features = np.array(video_features)
        video_features[video_features == 0] = small_constant

        audio_features = np.array(audio_features)
        audio_features[audio_features == 0] = small_constant

        position_target = np.array(position_target)
        position_target[position_target == 0] = small_constant

    return [audio_features, video_features], [index_target, position_target, np.zeros(len(index_target))]


def chunks(l, context):
    return [[l[i-context+j] for j in range(context+1)] for i in range(context, len(l))]


def dummy_loss(y_pred, y_true):
    return K.zeros(1)


def scalar_log_func(x):
    lamb = Reshape((1, 1))(x[2])
    return Add()([lamb*K.log(x[0]), (1-lamb)*K.log(x[1])])


def matrix_log_func(x):
    lamb_a = x[2]
    lamb_v = x[3]
    return Add()([lamb_a*K.log(x[0]), lamb_v*K.log(x[1])])


def naive_log_func(x):
    return Add()([K.log(x[0]), K.log(x[1])])


def get_model(grid_size, lr, fusion_type="matrix"):
    # Defining Model
    audio_in = Input(shape=grid_size)
    video_in = Input(shape=grid_size)

    if fusion_type == "scalar":
        aud_flat = Flatten()(audio_in)
        vid_flat = Flatten()(video_in)

        aud_dense = Dense(64)(aud_flat)
        aud_dense = BatchNormalization()(aud_dense)
        aud_dense = Activation('relu')(aud_dense)
        aud_dense = Dense(32)(aud_dense)
        aud_dense = BatchNormalization()(aud_dense)
        aud_dense = Activation('relu')(aud_dense)

        vid_dense = Dense(64)(vid_flat)
        vid_dense = BatchNormalization()(vid_dense)
        vid_dense = Activation('relu')(vid_dense)
        vid_dense = Dense(32)(vid_dense)
        vid_dense = BatchNormalization()(vid_dense)
        vid_dense = Activation('relu')(vid_dense)

        aud_vid_comb = Concatenate()([aud_dense, vid_dense])
        lamb = Dense(16, activation='relu')(aud_vid_comb)
        lamb = Dense(1, activation='sigmoid')(lamb)

        inputs = Lambda(scalar_log_func)([audio_in, video_in, lamb])
        inputs = Reshape((grid_size[0], grid_size[1], 1))(inputs)

    if fusion_type == "matrix":
        audio_conv = Conv2D(64, (3, 3), padding='same')(audio_in)
        audio_conv = Conv2D(32, (3, 3), padding='same')(audio_conv)
        lamb_a = Conv2D(1, (1, 1), padding='same')(audio_conv)

        video_conv = Conv2D(64, (3, 3), padding='same')(video_in)
        video_conv = Conv2D(32, (3, 3), padding='same')(video_conv)
        lamb_v = Conv2D(1, (1, 1), padding='same')(video_conv)

        inputs = Lambda(matrix_log_func)([audio_in, video_in, lamb_a, lamb_v])
    else:
        inputs = Lambda(naive_log_func)([audio_in, video_in])

    cnn1 = Conv2D(128, (3, 3), padding='same')(inputs)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = Activation('relu')(cnn1)

    cnn2 = Conv2D(128, (3, 3), padding='same')(cnn1)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = Activation('relu')(cnn2)

    cnn3 = Conv2D(128, (3, 3), padding='same')(cnn2)
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = Activation('relu')(cnn3)

    cnn4 = Conv2D(128, (3, 3), padding='same')(cnn3)
    cnn4 = BatchNormalization()(cnn4)
    cnn4 = Activation('relu')(cnn4)

    cnn5 = Conv2D(128, (3, 3), padding='same')(cnn4)
    cnn5 = BatchNormalization()(cnn5)
    cnn5 = Activation('relu')(cnn5)

    dnn1 = Conv2DTranspose(128, (3, 3), padding='same')(cnn5)

    dnn2 = Add()([dnn1, Conv2DTranspose(128, (3, 3), padding='same')(cnn4)])
    dnn2 = Activation('relu')(BatchNormalization()(dnn2))

    dnn3 = Add()([dnn2, Conv2DTranspose(128, (3, 3), padding='same')(cnn3)])
    dnn3 = Activation('relu')(BatchNormalization()(dnn3))

    dnn4 = Add()([dnn3, Conv2DTranspose(128, (3, 3), padding='same')(cnn2)])
    dnn4 = Activation('relu')(BatchNormalization()(dnn4))

    dnn5 = Add()([dnn4, Conv2DTranspose(128, (3, 3), padding='same')(cnn1)])
    dnn5 = Activation('relu')(BatchNormalization()(dnn5))

    out = Conv2DTranspose(1, (1, 1), padding='same', name="doa", activation='relu')(dnn5)

    # Branching
    input_neurons = Flatten()(out)
    speakers = Dense(32)(input_neurons)
    speakers = Dropout(rate=0.4)(speakers)
    speakers = Activation('relu')(speakers)
    speakers = Dense(4)(speakers)
    speakers = Activation('sigmoid', name="speaker")(speakers)

    doa = out

    # Building Model
    model = Model(inputs=[audio_in, video_in], outputs=[speakers, doa])
    model.compile(optimizer=Adam(lr=lr), loss=['binary_crossentropy', 'mean_squared_error'])
    model.summary()
    return model


if __name__ == "__main__":
    # Parameters
    np.random.seed(1995)
    grid_size = (20, 24)
    batch_size = 32
    epochs = 4
    learning_rate = 0.001
    model_path = "model/fusion.h5"
    train = True

    # Loading Features
    train_features = joblib.load("AVDIAR-FEATURES_TRAIN/train")
    test_features = joblib.load("AVDIAR-FEATURES_TEST/test")
    val_features = joblib.load("AVDIAR-FEATURES_VAL/val")

    # Converting Features
    train_set = generate_dataset(train_features, grid_size)
    test_set = generate_dataset(test_features, grid_size, shuffle=False)
    val_set = generate_dataset(val_features, grid_size)

    if train:
        # Getting Model
        model = get_model(grid_size, learning_rate)
        callbacks = [EarlyStopping('val_loss')]

        # Training
        model.fit(train_set[0], train_set[1], validation_data=val_set, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        model.save(model_path)
    else:
        model = load_model(model_path, custom_objects={"Add": Add, "dummy_loss": dummy_loss})

    # Evaluating Model
    score = model.evaluate(test_set[0], test_set[1])
    print("Combined Loss: {}, speakers: {}, doa: {}".format(score[0], score[1], score[2]))

    cap = cv2.VideoCapture("AVDIAR_TEST/Seq29-3P-S1M0/Video/Seq29-3P-S1M0_CAM1.mp4")
    if not cap.isOpened():
        raise IOError("Error opening video stream or file: ", "AVDIAR_TEST/Seq29-3P-S1M0/Video/Seq29-3P-S1M0_CAM1.mp4")

    test_result = []
    for (a, v, t, d) in zip(val_set[0][0], val_set[0][1], val_set[1][0], val_set[1][1]):
        inputs = [np.expand_dims(a, 0), np.expand_dims(v, 0)]
        prediction = model.predict(inputs)
        test_result.append((prediction, t, d))
    print(ut.calculate_metrics(test_result, grid_size))
