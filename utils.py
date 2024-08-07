import numpy as np
import matplotlib.pyplot as plt

def save_images(generator, epoch, save_dir='images'):
    noise = np.random.normal(0, 1, (25, 100))
    gen_imgs = generator.predict(noise)
    
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(5, 5)
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig(f"{save_dir}/mnist_{epoch}.png")
    plt.close()
