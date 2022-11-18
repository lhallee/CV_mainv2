import argparse
import os
from data_processing import file_to_dataloader
from run_model import Solver
from plots import preview_crops
from torch.backends import cudnn


def main(config):
    config.output_ch = config.num_class
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.img_path):
        os.makedirs(config.img_path)
    if not os.path.exists(config.GT_path):
        os.makedirs(config.GT_path)

    print(config)

    train_loader, valid_loader, test_loader = file_to_dataloader(img_path=config.img_path,
                                                                 GT_path=config.GT_path,
                                                                 dim=config.image_size,
                                                                 num_class=config.num_class,
                                                                 train_per=config.train_per,
                                                                 batch_size=config.batch_size
                                                                 )
    print(len(train_loader), len(valid_loader), len(test_loader))
    vis_imgs, vis_GTs = train_loader.dataset[:3]
    preview_crops(vis_imgs, vis_GTs, config.num_class)
    solver = Solver(config, train_loader, valid_loader, test_loader)


    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'test':
        solver.test()

def run_from_main():
    parser = argparse.ArgumentParser()
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--scheduler', type=str, default=None, help='None, or exp anneal \'exp\'')

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='R2AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--img_path', type=str, default='./img/')
    parser.add_argument('--GT_path', type=str, default='./GT/')
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--train_per', type=float, default=0.7, help='Percentage of training data in dataloaders')
    config = parser.parse_args()
    main(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--scheduler', type=str, default=None, help='None, or exp anneal \'exp\'')

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='R2AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--img_path', type=str, default='./img/')
    parser.add_argument('--GT_path', type=str, default='./GT/')
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--train_per', type=float, default=0.7, help='Percentage of training data in dataloaders')

    config = parser.parse_args()
    main(config)
