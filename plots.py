import numpy as np
import matplotlib.pyplot as plt

def preview_crops(imgs, GTs, num_class=2):
    rows = 1
    columns = num_class
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
            plt.imshow(GTs[i][:,:,0], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title('GT')
            plt.show()
        else:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(GTs[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title('GT')
            fig.add_subplot(rows, columns, 3)
            plt.imshow(GTs[i][:, :, 1], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title('GT')
            plt.show()

def checker(path, feed_img, imgs, GTs, epoch, batch, num_class=2):
    rows = 1
    columns = num_class
    imgs = np.transpose(np.array(imgs), axes=(0, 2, 3, 1))
    GTs = np.transpose(np.array(GTs), axes=(0, 2, 3, 1))
    feed_img = np.transpose(np.array(feed_img), axes=(0, 2, 3, 1))
    i = np.random.randint(0, len(imgs))
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(imgs[i][:,:,0])
    plt.axis('off')
    plt.title('Img')
    if num_class == 2:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GTs[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
    else:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GTs[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        fig.add_subplot(rows, columns, 3)
        plt.imshow(GTs[i][:, :, 1], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
    save = np.hstack((np.mean(feed_img[0], -1).reshape(len(imgs[0]), len(imgs[0])), imgs[0][:,:,0], GTs[0][:,:,0]))
    plt.imsave(path + str(batch) + '_' + str(epoch) + '_check_img.png', save)

def test_saver(path, imgs, GTs, batch):
    imgs = np.transpose(np.array(imgs.detach().cpu().numpy()), axes=(0, 2, 3, 1))
    GTs = np.transpose(np.array(GTs.detach().cpu().numpy()), axes=(0, 2, 3, 1))
    for i in range(len(imgs)):
        plt.imsave(path + str(batch) + '_check_img.png', np.hstack(((imgs[0][:,:,1]), (GTs[0][:,:,1]))))