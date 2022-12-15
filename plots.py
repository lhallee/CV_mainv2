import numpy as np
import matplotlib.pyplot as plt

def preview_crops(imgs, GTs, num_class=2):
    #Displays crops from dataloaders
    rows = 1
    columns = num_class
    #Back to normal image format
    imgs = np.transpose(np.array(imgs), axes=(0, 2, 3, 1))
    GTs = np.transpose(np.array(GTs), axes=(0, 2, 3, 1))
    for i in range(len(imgs)):
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title('Img')
        if num_class == 1:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(GTs[i][:,:,0], cmap='gray')
            plt.axis('off')
            plt.title('GT')
            plt.show()
        else:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(GTs[i][:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title('GT')
            fig.add_subplot(rows, columns, 3)
            plt.imshow(GTs[i][:, :, 1], cmap='gray')
            plt.axis('off')
            plt.title('GT')
            plt.show()

def preview_crops_eval(imgs):
    #Displays crops from dataloaders
    #Back to normal image format
    imgs = np.transpose(np.array(imgs), axes=(0, 2, 3, 1))
    for i in range(len(imgs)):
        plt.imshow(imgs[i])
        plt.show()

def checker(path, feed_img, SR, GT, epoch, batch, num_class=2):
    #For checking training progress mid training
    rows = 1
    columns = num_class
    SR = np.transpose(np.array(SR), axes=(0, 2, 3, 1))
    GT = np.transpose(np.array(GT), axes=(0, 2, 3, 1))
    feed_img = np.transpose(np.array(feed_img), axes=(0, 2, 3, 1))
    i = np.random.randint(0, len(SR))
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(SR[i][:,:,0])
    plt.axis('off')
    plt.title('Img')
    if num_class == 1:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GT[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
    else:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GT[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        fig.add_subplot(rows, columns, 3)
        plt.imshow(GT[i][:, :, 1], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
    save = np.hstack((np.mean(feed_img[0], -1).reshape(len(SR[0]), len(SR[0])), SR[0][:,:,0], GT[0][:,:,0]))
    plt.imsave(path + str(batch) + '_' + str(epoch) + '_check_img.png', save) #save to results path

def test_saver(path, feed_img, SR, GT, batch):
    #For saving test image to result path
    #Back to normal image format
    SR = np.transpose(np.array(SR), axes=(0, 2, 3, 1))
    GT = np.transpose(np.array(GT), axes=(0, 2, 3, 1))
    feed_img = np.transpose(np.array(feed_img), axes=(0, 2, 3, 1))
    for i in range(len(SR)):
        save = np.hstack((np.mean(feed_img[i], -1).reshape(len(SR[0]), len(SR[0])), SR[i][:, :, 0], GT[i][:, :, 0]))
        plt.imsave(path + str(batch) + '_' + '_test_img.png', save)

def eval_saver(path, SR, im_num, eval_type):
    #For saving evaluation results
    levels = np.linspace(0.0, 1.0, 21)
    plt.contourf(SR, levels=levels, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.show()
    plt.imsave(path + 'eval' + eval_type + str(im_num) + '_img.png', SR)