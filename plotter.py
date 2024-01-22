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

def plot_npys(root_dir, envs, names, title, x_title, y_title, ma=1):
    datas = {}
    lens = []
    for i, n in enumerate(names):
        datas[n] = np.load(f'{root_dir}/{envs[i]}_{n}_Log.npy')
        lens.append(len(datas[n]))
    minimum_len = np.min(lens)

    plt.figure(figsize=(8, 6))

    for k in datas.keys():
        plot = np.convolve(datas[k], np.ones(ma)/ma, mode='valid')
        y_axis = [np.NaN] * (ma - 1) + list(plot)
        plt.plot(list(range(minimum_len)), y_axis, label=k, alpha=1)
    
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(loc='lower right')
    plt.savefig(f'{root_dir}/{title}.png')

if __name__ == '__main__':
    plot_npys('./Results/agents', ['Sprites-v1'], ['cnn'], 'Sprites-v1', 'step', 'reward', 2000)

