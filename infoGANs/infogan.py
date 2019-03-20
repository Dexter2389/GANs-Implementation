'''Trains infoGAN on Cifar10 using Keras

[1] Chen, Xi, et al. "Infogan: Interpretable representation learning by
information maximizing generative adversarial nets." 
Advances in Neural Information Processing Systems. 2016.
'''

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import datasets
from tensorflow.keras import layers, models, optimizers, utils

def generator_model(layer_filters=[256, 128, 64], latent_dim=134, activation="tanh", channels=3, kernel_size=(4,4), padding="same"):
        
    gen_input = layers.Input(shape=(latent_dim, ))

    x = layers.Dense(1024, activation="relu")(gen_input)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Dense(448*4*4, activation="relu")(x)
    x = layers.Reshape((4,4,448))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    for filters in layer_filters:
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters, kernel_size=kernel_size, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
    
    conv_last = layers.Conv2D(channels, kernel_size=kernel_size, padding="same")(x)
    activation = layers.Activation(activation)(conv_last)
    
    model = models.Model(gen_input, activation)
    model.summary()
    
    return model
    
def discriminator_recognition_net(layer_filters=[64, 128, 256], img_shape=(32, 32, 3), kernel_size=(4,4), strides=2):
    
    dis_input = layers.Input(shape=img_shape)
    
    x = layers.Conv2D(layer_filters[0], kernel_size=kernel_size, strides=strides, padding="same")(dis_input)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.25)(x)
    
    for filters in layer_filters[1:(len(layer_filters)-1)]:
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
        x = layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2D(layer_filters[(len(layer_filters)-1)], kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Flatten()(x)
    
    return models.Model(dis_input, x)

def discriminator_model(img_shape=(32, 32, 3), activation="sigmoid", channels=3):
    dis_input = layers.Input(shape=img_shape)
    
    x = discriminator_recognition_net()(dis_input)
    final = layers.Dense(channels, activation=activation)(x)
    
    model = models.Model(dis_input, final)
    model.summary()
    
    return model

def recognition_model(img_shape=(32, 32, 3), activation="softmax", num_classes=10):
    reco_input = layers.Input(shape=img_shape)
    
    x = discriminator_recognition_net()(reco_input)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(num_classes, activation=activation)(x)
    
    model = models.Model(reco_input, x)
    model.summary()
    
    return model

def mutual_info_loss(c, c_given_x):
    """The mutual information metric we aim to minimize"""
    eps = 1e-8
    conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
    entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))
    
    return conditional_entropy + entropy

def sample_generator_input(batch_size, noise_variable=124, num_classes=10):
    # Generator inputs
    sampled_noise = np.random.normal(0, 1, (batch_size, noise_variable))
    #sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
    sampled_labels = utils.to_categorical((np.random.randint(0, num_classes, batch_size).reshape(-1, 1)), num_classes=num_classes)
    
    return sampled_noise, sampled_labels

def infoGAN():
        img_rows = 32
        img_cols = 32
        channels = 3
        num_classes = 10
        img_shape = (img_rows, img_cols, channels)
        latent_dim = 134

        optimizer = optimizers.Adam(0.0002, 0.5)
        losses = ["binary_crossentropy", mutual_info_loss]

        #Building Discriminator and Recognition Network
        discriminator = discriminator_model()
        recognition = recognition_model()

        discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the recognition network Q
        recognition.compile(loss=[mutual_info_loss], optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        generator = generator_model()

        # The generator takes noise and the target label as input and generates the corresponding digit of that label
        gen_input = layers.Input(shape=(latent_dim, ))
        img = generator(gen_input)

        discriminator.trainable = True

        # The discriminator takes generated image as input and determines validity
        valid = discriminator(img)
        # The recognition network produces the label
        target_label = recognition(img)

        # The combined model (stacked generator and discriminator)
        combined = models.Model(gen_input, [valid, target_label])
        combined.compile(loss=losses, optimizer=optimizer)

        return discriminator, recognition, generator, combined

def train(discriminator, generator, combined, epochs, batch_size=128, sample_interval=50, channels=3):

    #Load the Dataset
    (X_train, y_train), (_, _) = datasets.cifar10.load_data()
    print(X_train.shape)
    
    #Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    print(X_train.shape)
    y_train = y_train.reshape(-1, 1)
    
    #Adversarial ground truths
    valid = np.ones((batch_size, channels))
    fake = np.zeros((batch_size, channels))
    for epoch in range(epochs):
        
        #------------------Train Discriminator---------------
    
        #Selecting a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
    
        #Sample noise and categorical labels
        sampled_noise, sampled_labels = sample_generator_input(batch_size)
        gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
    
        #Generate a half batch of new images
        gen_imgs = generator.predict(gen_input)
    
        #------------------Train on real and generated data---------------
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    
        # Avg. loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
        # Train Generator and Discriminator Network
        g_loss = combined.train_on_batch(gen_input, [valid, sampled_labels])
    
        # Plot the progress
        print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))
    
        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(epoch, generator)

def sample_images(epoch, generator, num_classes=10):

    r, c = 10, 10
    fig, axs = plt.subplots(r, c)
    
    for i in range(c):
        sampled_noise, _ = sample_generator_input(c)
        label = utils.to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=num_classes)
        gen_input = np.concatenate((sampled_noise, label), axis=1)
        gen_imgs = generator.predict(gen_input)
        gen_imgs = 0.5 * gen_imgs + 0.5
        for j in range(r):
            axs[j,i].imshow(gen_imgs[j,:,:,0], cmap="brg")
            axs[j,i].axis('off')
    
    fig.savefig("infoGANs/images/%d.png" % epoch)
    plt.close()

def save(model, model_name):
    
    model_path = "infoGANs/saved_model/%s.json" % model_name
    weights_path = "infoGANs/saved_model/%s_weights.hdf5" % model_name
    options = {"file_arch": model_path,
                "file_weight": weights_path}
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])

def save_model(generator, discriminator):
    save(generator, "generator")
    save(discriminator, "discriminator")



if __name__ == "__main__":
    discriminator, recognition, generator, combined = infoGAN()
    train(discriminator, generator, combined, epochs=50000)
