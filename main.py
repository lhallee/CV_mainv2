import argparse
import os
from data_processing import Imageset_processing
from run_model import Solver
from plots import preview_crops, preview_crops_eval
from mock_data import to_dataloader_mock
from torch.backends import cudnn
from evaluation import eval_solver


def main(config):
    config.output_ch = config.num_class - 1
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
    data_setup = Imageset_processing(config)
    if config.mode == 'eval':
        eval_loader, num_col, num_row = data_setup.eval_dataloader()
        vis_imgs = eval_loader.dataset[:10]
        preview_crops_eval(vis_imgs)
        solver = eval_solver(config, eval_loader, num_col, num_row)
        solver.eval()
    #Can choose between real data in a path or generated data of squares of various sizes
    elif config.data_type == 'Real':
        train_loader, valid_loader, test_loader = data_setup.to_dataloader()
        print(len(train_loader), len(valid_loader), len(test_loader))
        vis_imgs, vis_GTs = train_loader.dataset[:10]
        preview_crops(vis_imgs, vis_GTs, config.num_class)
        solver = Solver(config, train_loader, valid_loader, test_loader)
    elif config.data_type == 'Mock':
        train_loader, valid_loader, test_loader = to_dataloader_mock(dim=config.image_size,
                                                                     train_per=config.train_per,
                                                                     batch_size=config.batch_size
                                                                     )
        print(len(train_loader), len(valid_loader), len(test_loader))
        vis_imgs, vis_GTs = train_loader.dataset[:10]
        preview_crops(vis_imgs, vis_GTs, config.num_class)
        solver = Solver(config, train_loader, valid_loader, test_loader)

    #Train utilizes random weights to train until stopping criteria of the number of epochs
    #then calls the test function
    if config.mode == 'train':
        solver.train()
        solver.test()
    #Uses pretrained weights from model_path
    elif config.mode == 'test':
        solver.test()


def run_from_main():
    parser = argparse.ArgumentParser()
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--scheduler', type=str, default='cosine', help='None, exp, cosine')
    parser.add_argument('--loss', type=str, default='DiceBCE', help='BCE, DiceBCE, IOU, CE')

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='R2AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--img_path', type=str, default='./img/')
    parser.add_argument('--GT_path', type=str, default='./GT/')
    parser.add_argument('--eval_img_path', type=str, default='./eval_img/')
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--train_per', type=float, default=0.7, help='Percentage of training data in dataloaders')
    parser.add_argument('--data_type', type=str, default='Real', help='Real or Mock data')
    parser.add_argument('--progress', type=bool, default=True, help='Save images over time or not')
    config = parser.parse_args(args=[])
    main(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--scheduler', type=str, default='cosine', help='None, exp, cosine')
    parser.add_argument('--loss', type=str, default='DiceBCE', help='BCE, DiceBCE, IOU, CE')

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train, test, or eval')
    parser.add_argument('--model_type', type=str, default='R2AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--img_path', type=str, default='./img/')
    parser.add_argument('--GT_path', type=str, default='./GT/')
    parser.add_argument('--eval_img_path', type=str, default='./eval_img/') #full LN for evaluation
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--train_per', type=float, default=0.7, help='Percentage of training data in dataloaders')
    parser.add_argument('--data_type', type=str, default='Real', help='Real or Mock data')
    parser.add_argument('--progress', type=bool, default=False, help='Save images over time or not')
    parser.add_argument('--eval_type', type=str, default='Windowed', help='Type of evaluation. Windowed or Scaled')

    config = parser.parse_args()
    main(config)
