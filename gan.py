from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

def compile_models(generator, discriminator):
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.trainable = False

    z = Input(shape=(100,))
    img = generator(z)
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return combined
