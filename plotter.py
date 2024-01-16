import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def plot_and_save_loss_per_epoch_1(list, title, folder):
    plt.figure(figsize=(8, 6))

    plt.plot(range(len(list)), list, c='b', label='loss', alpha=0.3)

    y_ma = np.convolve(list, np.ones(10)/10, mode='valid')

    plt.plot(range(len(y_ma)), y_ma, c='b', label='loss', alpha=1)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'Results/{folder}/{title}.png')

def plot_imgs(images, num_cols, num_rows, idxs):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, int(10 / num_cols * num_rows)))
    norm = Normalize(vmin=-1, vmax=1)
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if i == 0:
                axes[i, j].set_title(f'{idxs[j]}')
            axes[i, j].imshow(images[index], cmap='gray', norm=norm)  # cmap은 흑백 이미지일 경우 사용
            axes[i, j].axis('off')  # 축 숨김
    plt.show()