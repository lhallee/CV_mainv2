import numpy as np
import matplotlib.pyplot as plt

def preview_crops(imgs, GTs, num_class=2):
    rows = num_class
    columns = 1
    imgs = np.transpose(np.array(imgs), axes=(0, 2, 3, 1))
    GTs = np.transpose(np.array(GTs), axes=(0, 2, 3, 1))
    for i in range(len(imgs)):
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title('Img')
        if num_class == 2:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(GTs[i][:,:,1], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title('GT')
            plt.show()
        else:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(GTs[i][:, :, 1], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title('GT')
            fig.add_subplot(rows, columns, 3)
            plt.imshow(GTs[i][:, :, 2], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title('GT')
            plt.show()

def checker(path, imgs, GTs, batch, num_class=2):
    rows = num_class
    columns = 1
    imgs = np.transpose(np.array(imgs.detach().cpu().numpy()), axes=(0, 2, 3, 1))
    GTs = np.transpose(np.array(GTs.detach().cpu().numpy()), axes=(0, 2, 3, 1))
    i = np.random.randint(0, len(imgs))
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(imgs[i][:,:,1])
    plt.axis('off')
    plt.title('Img')
    if num_class == 2:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GTs[i][:, :, 1], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
    else:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GTs[i][:, :, 1], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        fig.add_subplot(rows, columns, 3)
        plt.imshow(GTs[i][:, :, 2], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
    plt.imsave(path + str(epoch) + '_check_img.png', np.hstack(((imgs[0][:,:,1]), (GTs[0][:,:,1]))))

def test_saver(path, imgs, GTs, batch):
    imgs = np.transpose(np.array(imgs.detach().cpu().numpy()), axes=(0, 2, 3, 1))
    GTs = np.transpose(np.array(GTs.detach().cpu().numpy()), axes=(0, 2, 3, 1))
    for i in range(len(imgs)):
        plt.imsave(path + str(epoch) + '_check_img.png', np.hstack(((imgs[0][:,:,1]), (GTs[0][:,:,1]))))