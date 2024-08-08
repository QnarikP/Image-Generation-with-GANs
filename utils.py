import matplotlib.pyplot as plt
import numpy as np
import os

def save_images(generator, epoch, save_dir='images'):
    print(f"\033[94mSaving images for epoch {epoch}...\033[0m")
    os.makedirs(save_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (25, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{save_dir}/mnist_{epoch}.png")
    plt.close()
    print(f"\033[94mImages saved at {save_dir}/mnist_{epoch}.png\033[0m")
