from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD

import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import os, shutil, math, cv2
import matplotlib.pyplot as plt


def conv(filter, kernel_size, strides=1):
    return L.Conv2D(filter, kernel_size, strides=strides, padding='same', use_bias=False,  kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))


def residual_block_a(filters, strides):
    def f(x):
        x = L.BatchNormalization()(x)
        b = L.Activation('relu')(x)

        x = conv(filters // 4, 1, strides)(b)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)

        x = conv(filters // 4, 3)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)

        x = conv(filters, 1)(x)

        sc = conv(filters, 1, strides)(b)

        return L.Add()([x, sc])
    return f


def residual_block_b(filters):
    def f(x):
        sc = x

        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)

        x = conv(filters // 4, 1)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)

        x = conv(filters // 4, 3)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)

        x = conv(filters, 1)(x)

        return L.Add()([x, sc])
    return f


def residual_block(filters, strides, unit_size):
    def f(x):
        x = residual_block_a(filters, strides)(x)
        for i in range(unit_size - 1):
            x = residual_block_b(filters)(x)
        return x
    return f


def plot_graph(history, log_dir, model_name):
    lr_lst = []
    for i_lr in range(len(LR_LIST)):
        lr_lst.append(abs(math.log10(LR_LIST[i_lr])) * 0.1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(TITLE)
    ax1.set_title('Accuracy')
    ax1.plot(history.history['accuracy'], label='Training Acc')
    ax1.plot(history.history['val_accuracy'], label='Test Acc')
    ax1.plot(lr_lst[0:EPOCHS])
    ax2.set_title('Loss')
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Test Loss')
    ax2.plot(lr_lst[0:EPOCHS])
    ax2.set_ylim([-0.05, 1.0])
    ax1.legend()
    ax2.legend()

    amax_acc = np.argmax(history.history['val_accuracy'])
    valacc = history.history['val_accuracy']
    amin_loss = np.argmin(history.history['val_loss'])
    valloss = history.history['val_loss']
    ax1.axvline(x=amax_acc, color='black', linewidth=0.7, linestyle='--')
    ax1.plot(amax_acc, valacc[amax_acc], 'o-', color='orange')
    ax1.text(amax_acc, valacc[amax_acc] + 0.05, '{0:d},  {1:.4f}'.format(amax_acc, valacc[amax_acc]), ha='center')
    ax2.axvline(x=np.argmin(history.history['val_loss']), color='black', linewidth=0.7, linestyle='--')
    ax2.plot(amin_loss, valloss[amin_loss], 'o-', color='orange')
    ax2.text(amin_loss, valloss[amin_loss] - 0.1, '{0:d},  {1:.4f}'.format(amin_loss, valloss[amin_loss]), ha='center')

    major_xticks = [i for i in range(0, EPOCHS + 1, 10)]
    minor_xticks = [i for i in range(0, EPOCHS + 1, 5)]
    acc_major_yticks = [i for i in np.arange(0.3, 1.1, 0.1)]
    acc_minor_yticks = [i for i in np.arange(0.3, 1.1, 0.05)]
    loss_major_yticks = [i for i in np.arange(0.0, 1.1, 0.1)]
    loss_minor_yticks = [i for i in np.arange(0.0, 1.1, 0.05)]

    ax1.set_xticks(major_xticks)
    ax1.set_xticks(minor_xticks, minor=True)
    ax1.set_yticks(acc_major_yticks)
    ax1.set_yticks(acc_minor_yticks, minor=True)
    ax2.set_xticks(major_xticks)
    ax2.set_xticks(minor_xticks, minor=True)
    ax2.set_yticks(loss_major_yticks)
    ax2.set_yticks(loss_minor_yticks, minor=True)

    ax1.grid(axis='x', which='major')
    ax1.grid(axis='x', which='minor')
    ax1.grid(axis='y', which='major')
    ax1.grid(axis='y', which='minor', linestyle=":")
    ax2.grid(axis='x', which='major')
    ax2.grid(axis='x', which='minor')
    ax2.grid(axis='y', which='major')
    ax2.grid(axis='y', which='minor', linestyle=":")
    plt.show()
    fig.savefig(log_dir + model_name + '_fig_accloss.png')


TITLE = 'CIFAR10_ResNet'
SAVEPATH = './result/'
LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EPOCHS = 150
LR_LIST = []
for i_lr in range(80):
    LR_LIST.append(1e-1)
for i_lr in range(40):
    LR_LIST.append(1e-2)
for i_lr in range(40):
    LR_LIST.append(1e-3)
for i_lr in range(50):
    LR_LIST.append(5e-4)
for i_lr in range(50):
    LR_LIST.append(1e-4)
if __name__ == '__main__':
    #### load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print("##############################################################################")
    print('X_train shape = %s, y_train shape = %s' % (X_train.shape, y_train.shape))
    print('X_test shape = %s, y_test shape = %s' % (X_test.shape, y_test.shape))
    print("##############################################################################")

    train_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
    test_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    train_gen.fit(X_train)
    test_gen.fit(X_test)


    #### make model
    input = L.Input(shape=(32, 32, 3))

    x = conv(16, 3)(input)
    x = residual_block(64, 1, 18)(x)
    x = residual_block(128, 2, 18)(x)
    x = residual_block(256, 2, 18)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.GlobalAveragePooling2D()(x)

    output = L.Dense(10, activation='softmax', kernel_regularizer=l2(0.0001))(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    #### save model
    model_name = 'test_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    log_dir = SAVEPATH + model_name + '/'
    os.makedirs(log_dir)
    shutil.copy2(__file__, log_dir)
    plot_model(model, to_file=log_dir + model_name + "_fig_pm.png", show_shapes=True)

    model_cp_path = os.path.join(log_dir, (model_name + '_checkpoint.h5'))
    model_csv_path = os.path.join(log_dir, (model_name + '_csv.csv'))

    TModelCheckpoint = ModelCheckpoint(model_cp_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    TCSVLogger = CSVLogger(model_csv_path)
    TLearningRateScheduler = LearningRateScheduler(lambda epoch: float(LR_LIST[epoch]))

    #### train model
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['accuracy'])

    history = model.fit(train_gen.flow(X_train, y_train, batch_size=128), steps_per_epoch=X_train.shape[0] // 128,
                        validation_data=test_gen.flow(X_test, y_test, batch_size=128), validation_steps=X_test.shape[0] // 128,
                        callbacks=[TModelCheckpoint, TCSVLogger, TLearningRateScheduler],
                        epochs=EPOCHS, batch_size=128, verbose=1, workers=8)

    #### plot
    plot_graph(history, log_dir, model_name)

    cm_model = M.load_model(model_cp_path)
    scores = cm_model.evaluate_generator(test_gen.flow(X_test, y_test, batch_size=128))
    print("accuracy: ", np.round(scores[1], decimals=4))
