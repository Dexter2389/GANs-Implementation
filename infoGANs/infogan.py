'''Trains infoGAN on Cifar10 using Keras

[1] Chen, Xi, et al. "Infogan: Interpretable representation learning by
information maximizing generative adversarial nets." 
Advances in Neural Information Processing Systems. 2016.
'''

import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

class InfoGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.num_classes = 10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 134

        optimizer = Adam(0.0002, 0.5)
        losses = ["binary_crossentropy", self.mutual_info_loss]

        #Building Discriminator and Recognition Network
        self.discriminator = self.discriminator_model()
        self.recognition = self.recognition_model()

        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the recognition network Q
        self.recognition.compile(loss=[self.mutual_info_loss], optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.generator_model()

        # The generator takes noise and the target label as input and generates the corresponding digit of that label
        gen_input = Input(shape=(self.latent_dim, ))
        img = self.generator(gen_input)

        self.discriminator.trainable = True

        # The discriminator takes generated image as input and determines validity
        valid = self.discriminator(img)
        # The recognition network produces the label
        target_label = self.recognition(img)

        # The combined model (stacked generator and discriminator)
        self.combined = Model(gen_input, [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)

    def generator_model(self):

        model = Sequential()

        model.add(Dense(1024, input_dim=self.latent_dim, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(448*4*4, activation="relu"))
        model.add(Reshape((4,4,448)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=(4,4), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=(4,4), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=(4,4), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(self.channels, kernel_size=(4,4), padding='same'))
        model.add(Activation("tanh"))

        gen_input = Input(shape=(self.latent_dim, ))
        img = model(gen_input)

        model.summary()

        return Model(gen_input, img)

    def discriminator_recognition_net(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=(4,4), strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(4,4), strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.25))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=(4,4), strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.25))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())

        return model
    
    def discriminator_model(self):
        model = self.discriminator_recognition_net()
        model.add(Dense(self.channels, activation="sigmoid"))

        dis_input = Input(shape=self.img_shape)
        validity = model(dis_input)

        model.summary()

        return Model(dis_input, validity)
    
    def recognition_model(self):
        model = self.discriminator_recognition_net()
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))

        reco_input = Input(shape=self.img_shape)
        label = model(reco_input)

        model.summary()

        return Model(reco_input, label)

    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy
    
    def sample_generator_input(self, batch_size, noise_variable=124):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, noise_variable))
        #sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical((np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)), num_classes=self.num_classes)
        return sampled_noise, sampled_labels

    def train(self, epochs, batch_size=128, sample_interval=50):

        #Load the Dataset
        (X_train, y_train), (_, _) = cifar10.load_data()
        print(X_train.shape)
        #Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        print(X_train.shape)
        y_train = y_train.reshape(-1, 1)

        #Adversarial ground truths
        valid = np.ones((batch_size, self.channels))
        fake = np.zeros((batch_size, self.channels))

        for epoch in range(epochs):
            
            #------------------Train Discriminator---------------

            #Selecting a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            #Sample noise and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
            #Generate a half batch of new images
            gen_imgs = self.generator.predict(gen_input)

            #------------------Train on real and generated data---------------
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            # Avg. loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator and Recognition Network
            g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(c)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5
            for j in range(r):
                axs[j,i].imshow(gen_imgs[j,:,:,0], cmap="brg")
                axs[j,i].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save(self, model, model_name):
        model_path = "saved_model/%s.json" % model_name
        weights_path = "saved_model/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    def save_model(self):
        self.save(self.generator, "generator")
        self.save(self.discriminator, "discriminator")

if __name__ == "__main__":
    infogan = InfoGAN()
    infogan.train(epochs=50000, batch_size=128, sample_interval=50)