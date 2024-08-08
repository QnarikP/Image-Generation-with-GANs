import numpy as np
import os
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Set up paths for saving models and images
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def build_generator():
    model = Sequential()
    
    model.add(Dense(256 * 7 * 7, activation="relu", input_dim=100))
    model.add(Reshape((7, 7, 256)))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(1, kernel_size=3, padding='same'))
    model.add(Activation('tanh'))

    model.summary()

    return model

def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model

def train(generator, discriminator, combined, epochs, batch_size, save_interval):
    # Load and preprocess the dataset
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator

        # Select a random half of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_images = generator.predict(noise)

        # Train the discriminator (real classified as real, fake classified as fake)
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator

        noise = np.random.normal(0, 1, (batch_size, 100))

        # Train the generator (wants discriminator to mistake images as real)
        g_loss = combined.train_on_batch(noise, real_labels)
        
        if epoch % 100 == 0:
            # Print the progress
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Save generated images at save_interval
        if epoch % save_interval == 0:
            save_images(epoch, generator)

def save_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    count = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig(os.path.join(output_dir, f"mnist_{epoch}.png"))
    plt.close()

if __name__ == '__main__':
    optimizer = Adam(learning_rate=0.0001, beta_1=0.5)  # Adjusted learning rate

    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    # Build the generator
    generator = build_generator()

    # The generator takes noise as input and generates images
    noise = np.random.normal(0, 1, (32, 100))
    gen_images = generator(noise)

    # For the combined model, we will only train the generator
    discriminator.trainable = False

    # The combined model (stacked generator and discriminator)
    combined = Sequential([generator, discriminator])
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Start training
    train(generator, discriminator, combined, epochs=10000, batch_size=32, save_interval=200)
