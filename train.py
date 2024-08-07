from tensorflow.keras.datasets import mnist
from generator import build_generator
from discriminator import build_discriminator
from gan import compile_models
from utils import save_images

def load_data():
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.0  # Normalize to [-1, 1]
    X_train = np.expand_dims(X_train, axis=-1)
    return X_train

def train(generator, discriminator, combined, epochs=10000, batch_size=32, save_interval=200):
    X_train = load_data()
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)

        # Print progress
        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
            save_images(generator, epoch)

if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()
    combined = compile_models(generator, discriminator)
    train(generator, discriminator, combined, epochs=10000, batch_size=32, save_interval=200)
