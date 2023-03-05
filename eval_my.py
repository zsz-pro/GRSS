import argparse
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import json
from dataset import RSDataset,WHUOPTSARDataset,GRSSDataset
import sync_transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from models.deeplabv3_version_1.deeplabv3 import DeepLabV3 as model1
from models.deeplabv3_version_2.deeplabv3 import DeepLabV3 as model2
from models.deeplabv3_version_3.deeplabv3 import DeepLabV3 as deeplabv3
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from palette import colorize_mask
from PIL import Image
from collections import OrderedDict
import cv2
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
# from models.SOLC.solc import SOLC
# from models.SOLCV2.solcv2 import SOLCV2
# from models.SOLCV3.solcv3 import SOLCV3_res50
# from models.SOLCV4.solcv4 import SOLCV4
# from models.SOLCV5.solcv5 import SOLCV5
from models.SOLCV7.solcv7 import SOLCV7
from models.MCANet.mcanet import MCANet
from torch.optim.lr_scheduler import StepLR
def parse_args():
    parser = argparse.ArgumentParser(description="Remote Sensing Segmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset-name', type=str, default='eight')
    
    # -===================！！！！！！！
    parser.add_argument('--train-data-root', type=str, default='/data/sy/whu-opt-sar-dataset-256/train')
    parser.add_argument('--val-data-root', type=str, default='/data/sy/whu-opt-sar-dataset-256/val')
    parser.add_argument('--save_root', type=str, default='/data/sy/experiments-whu-opt-sar-dataset-256')
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='weight-decay (default:1e-4)')
    
    parser.add_argument('--train-batch-size', type=int, default=8, metavar='N', help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=1, metavar='N', help='batch size for testing (default:16)')
    # output_save_path 
    parser.add_argument('--experiment-start-time', type=str, default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    
    # learning_rate 
    parser.add_argument('--base-lr', type=float, default=1e-3, metavar='M', help='')
    
    parser.add_argument('--total-epochs', type=int, default=40, metavar='N', help='number of epochs to train (default: 120)')
    
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # -===================！！！！！！！
    parser.add_argument('--model', type=str, default='solcv7', help='model name')
    
    # -===================！！！！！！！
    parser.add_argument('--save-pseudo-data-path', type=str, default='pseudo-data')
    # augmentation
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')
    # model
    
    parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--n-blocks', type=str, default='3, 4, 23, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='') # len=16
    parser.add_argument('--multi-grids', type=str, default='1, 1, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='6, 12, 18', help='')
    parser.add_argument('--deeplabv3-no-global-pooling', action='store_true', default=False)
    parser.add_argument('--deeplabv3-use-deformable-conv', action='store_true', default=False)
    parser.add_argument('--no-syncbn', action='store_true', default=False, help='using Synchronized Cross-GPU BatchNorm')
    # criterion
    parser.add_argument('--class-loss-weight', type=list, default=
    [0.0, 0.016682825992096393, 0.12286476797975535, 0.09874940237721894, 0.04047604729817842, 0.015269075073618998, 0.6013717852280317, 0.3362534066400197]) # 2022-06-07...
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')
    
    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    
    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='Adam')
    
    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=2, help='numbers of GPUs')
    
    parser.add_argument('--num_workers', type=int, default=8)
    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False)

    parser.add_argument('--best-kappa', type=float, default=0)


    parser.add_argument('--resume-path', type=str, default=None)
    
    parser.add_argument('--resume_model', type=bool, default=False)
    parser.add_argument('--resume_model_path', type=str, default='')
    parser.add_argument('--resume_start_epoch', type=int, default=0)
    parser.add_argument('--resume_total_epochs', type=int, default=500)

    args = parser.parse_args()
    # args.save_root = "/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/"
    # directory = args.save_root + "/%s/%s/" % ( args.model, args.experiment_start_time)
    # args.directory = directory
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # config_file = os.path.join(directory, 'config.json')
    # with open(config_file, 'w') as file:
    #     json.dump(vars(args), file, indent=4)

    if args.use_cuda:
        print('Numbers of GPUs:', len(args.gpu_ids))
    else:
        print("Using CPU")
    return args

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def evaluate(output, target):
    output[output == 0] = 0.00001
    output[output < 0] = 999
    target[target <= 0] = 0.00001
    valid_mask = ((target>0) + (output>0)) > 0

    output = output[valid_mask]
    target = target[valid_mask]
    # abs_diff = np.abs(output - target)

    # mse = np.mean(abs_diff ** 2)
    # rmse = np.sqrt((abs_diff ** 2).mean())
    # mae = np.mean(abs_diff)

    maxRatio = torch.maximum(output / target, target / output)
    delta1 = ((maxRatio < 1.25)+0).type(torch.float).mean()
    # delta2 = (maxRatio < 1.25 ** 2).mean()
    # delta3 = (maxRatio < 1.25 ** 3).mean()
    return delta1

class Trainer(object):
    def __init__(self, args):
        self.args = args
        resize_scale_range = [float(scale) for scale in args.resize_scale_range.split(',')]
        sync_transform = sync_transforms.ComposeWHU([
            sync_transforms.RandomFlipWHU(args.flip_ratio)
        ])
        self.resore_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        self.visualize = transforms.Compose([transforms.ToTensor()]) # /255.
        
        dataset_name = args.dataset_name
        class_name = []
        if dataset_name == 'fifteen': 
            from class_names import fifteen_classes 
            class_name = fifteen_classes()
        if dataset_name == 'eight': 
            from class_names import eight_classes
            class_name = eight_classes()
        if dataset_name == 'five': 
            from class_names import five_classes
            class_name = five_classes()
        if dataset_name == 'seven': 
            from class_names import seven_classes
            class_name = seven_classes()
        if dataset_name == 'two': 
            from class_names import two_classes
            class_name = two_classes()            
        self.train_dataset = WHUOPTSARDataset(class_name, root=args.train_data_root, mode='train', sync_transforms=sync_transform) # random flip
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.train_batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       drop_last=True)
        print('class names {}.'.format(self.train_dataset.class_names))
        print('Number samples {}.'.format(len(self.train_dataset)))
        if not args.no_val:
            val_data_set = GRSSDataset(class_name, root=args.val_data_root, mode='val', sync_transforms=None)
            self.val_loader = DataLoader(dataset=val_data_set,
                                         batch_size=args.val_batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False,
                                         drop_last=True)
        self.num_classes = len(self.train_dataset.class_names)
        print("类别数：", self.num_classes) # 16
        print(self.train_dataset.class_names)
        self.class_loss_weight = torch.Tensor(args.class_loss_weight)
         # -===================！！！！！！！  ignore 0
        # self.criterion = nn.CrossEntropyLoss(weight=self.class_loss_weight, reduction='mean', ignore_index=0).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        n_blocks = args.n_blocks
        n_blocks = [int(b) for b in n_blocks.split(',')]
        atrous_rates = args.deeplabv3_atrous_rates
        atrous_rates = [int(s) for s in atrous_rates.split(',')]
        multi_grids = args.multi_grids
        multi_grids = [int(g) for g in multi_grids.split(',')]

        if args.model == 'deeplabv3_version_1':
            model = model1(num_classes=self.num_classes)# dilate_rate=[6,12,18]
            # resume
            if args.resume_path:
                state_dict = torch.load('')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
        if args.model == 'deeplabv3_version_3':
            model = deeplabv3(num_classes=self.num_classes,
                           n_blocks=n_blocks,
                           atrous_rates=atrous_rates,
                           multi_grids=multi_grids,
                           output_stride=args.output_stride)          
        if args.model == 'solcv7':
            from models.SOLCV7.solcv7 import SOLCV7
            
            model = SOLCV7(num_classes=self.num_classes)
            print('======> model SOLC Version seven =============== ')    
            # from models.SOLCV7.solcv7 import SOLCV7
            
        if args.model == 'mcanet':
            from models.MCANet.mcanet import MCANet
            
            model = MCANet(num_classes=self.num_classes)
            print('======> model MCANet (Paper) =============== ')    
            # from models.SOLCV7.solcv7 import SOLCV7
        if args.model == 'ournetv1':
            from models.OurNet.ourv1 import OURV1
            
            model = OURV1(num_classes=self.num_classes)
            print('======> model OURV1 Version seven =============== ')    
     
            
        # print(model)

        if args.resume_model:
            print('resume model', args.resume_model)
            state_dict = torch.load(args.resume_model_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print('=========> resume model success', args.resume_model_path)

        if args.use_cuda:
            model = model.cuda()
            # self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            # -===================！！！！！！！  
            self.model = nn.DataParallel(model, device_ids=args.gpu_ids)

        # SGD不work，Adadelta出奇的好？
        if args.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(),
                                                  lr=args.base_lr,
                                                  weight_decay=args.weight_decay)
        if args.optimizer_name == 'Adam':
        # -===================！！！！！！！  ignore 0
            self.optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), 
                                              lr=args.base_lr, weight_decay=args.weight_decay)
        if args.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=model.parameters(),
                                             lr=args.base_lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        self.max_iter = args.total_epochs * len(self.train_loader)
        self.save_pseudo_data_path = args.save_root + '/' + args.save_pseudo_data_path
        # self.mixup_transform = sync_transforms.Mixup()
         

    def validating(self, epoch):
        self.model.eval()# 把module设成预测模式，对Dropout和BatchNorm有影响
        with torch.no_grad():
            conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
            tbar = tqdm(self.val_loader)
            for index, data in enumerate(tbar):
                # assert data[0].size()[2:] == data[1].size()[1:]
                imgs_sar = Variable(data[0])
                imgs_opt = Variable(data[1])
                # masks = Variable(data[2])
                # dsms = Variable(data[3])
                name = data[2]
                if self.args.use_cuda:
                    imgs_sar = imgs_sar.cuda()
                    imgs_opt = imgs_opt.cuda()
                    # masks = (masks/255).cuda()
                    

                outputs,height = self.model(imgs_sar, imgs_opt)
                        
                bulidings = outputs[:,1,:,:]
                # torch.max(tensor, dim)：指定维度上最大的数，返回tensor和下标
                # _, preds = torch.max(outputs, 1)
                func = nn.Sigmoid()
                prob = func(bulidings)
                preds = prob>0.9

                # _, preds = torch.max(outputs, 1)
                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                height = height.data.cpu().numpy().squeeze().astype(np.float32)*500000
                cv2.imwrite("/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/ournetv1/height3/"+str(name[0]),height)
                # cv2.imwrite("/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/ournetv1/pred_mask/"+str(name[0]),preds*255)
                
                # masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
                # score = prob.data.cpu().numpy()
                # if preds.ndim ==2:
                #     preds=preds.reshape(1,512,512)
                #     masks=masks.reshape(1,512,512)
                # val_visual = []
                
                # img_pil = self.resore_transform(data[1][0])
                # img_pil = Image.fromarray(np.uint8(np.array(img_pil)[:, :, :3]))
                # img_pil.convert('RGB')
                # print('convert success')
                
                # for i in range(score.shape[0]):
                #     num_score = np.sum(score[i] > 0.9)
                #     if num_score > 0:
                #         img_pil = self.resore_transform(data[1][i])
                #         preds_pil = Image.fromarray(preds[i].astype(np.uint8)).convert('L')
                #         pred_vis_pil = colorize_mask(preds[i])
                #         gt_vis_pil = colorize_mask(data[2][i].numpy())
                #         img_pil = Image.fromarray(np.uint8(np.array(img_pil)[:, :, :3]))
                #         val_visual.extend([self.visualize(img_pil.convert('RGB')),
                #                         self.visualize(gt_vis_pil.convert('RGB')),
                #                         self.visualize(pred_vis_pil.convert('RGB'))])
                # if val_visual:
                #     val_visual = torch.stack(val_visual, 0)
                #     val_visual = torchvision.utils.make_grid(tensor=val_visual,
                #                                             nrow=3,
                #                                             padding=5,
                #                                             normalize=False,
                #                                             range=None,
                #                                             scale_each=False,
                #                                             pad_value=0)
                #     writer.add_image(tag='pres&GTs', img_tensor=val_visual, global_step=None, walltime=None)
                    
            #     conf_mat += metric.confusion_matrix(pred=preds.flatten(),
            #                                         label=masks.flatten(),
            #                                         num_classes=self.num_classes)
                                                    
            # val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = metric.evaluate(conf_mat)

            #  # table = PrettyTable(["序号", "名称", "acc", "IoU"])
            # for i in range(self.num_classes):
            #     # table.add_row([i, self.train_dataset.class_names[i], val_acc_per_class[i], val_IoU[i]])
            #     print('====> class id ', i, self.train_dataset.class_names[i], val_acc_per_class[i], val_IoU[i])
            # # print(table)
            # delta1 = evaluate(height,dsms)
            # score = (delta1+val_acc)/2            
            # print("val_mean_IoU (Iou):", val_mean_IoU)
            # print("delta1:", delta1)
            # print("score:",score)
            
            # print("val_acc (OA):", val_acc)            
            # print("kappa (Kappa):", val_kappa)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    args.train_data_root = "/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/train/"
    args.val_data_root = "/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/val_temp/"
    args.save_root = "/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/"
    args.dataset_name = "two"
    # args.class_loss_weight =[0.06, 0.116682825992096393]#像素占得越多，权重越小;2是building
    args.class_loss_weight =[0.1, 0.1]#像素占得越多，权重越小;2是building
    args.resume_model = True
    args.resume_model_path ="/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/datasets/GRSS/ournetv1/both_all/epoch_99_delta1_0.00544_train_Iou_0.75184.pth"
    args.val_batch_size = 1
    args.eval = True
    args.model = 'ournetv1'
    # args.model = "deeplabv3_version_3"
    # writer = SummaryWriter(args.directory)
    trainer = Trainer(args)
    

    trainer.validating(epoch=0)

    
    # if args.resume_model:
    #     print("=====> Continue Train:")
    #     args.start_epoch = args.resume_start_epoch
    #     args.total_epochs = args.resume_total_epochs
    # scheduler = StepLR(trainer.optimizer, step_size=args.step_size, gamma=args.gamma)
    # for epoch in range(args.start_epoch, args.total_epochs):
    #     trainer.training(epoch)
    #     scheduler.step()
    #     if not trainer.args.no_val:
    #         trainer.validating(epoch)
            
    #python train.py --model solcv7 --train-batch-size 8 --val-batch-size 8
    #python train.py --model solcv7 --eval True --train-batch-size 8 --val-batch-size 8
    #print('resume model', args.resume_model)
    # state_dict = torch.load(args.resume_model_path)