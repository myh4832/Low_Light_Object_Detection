import os.path as osp
import torch
import torch.utils.data as data
import basicsr.data.util as util
import torch.nn.functional as F
import cv2
import numpy as np
from basicsr.utils.img_util import low_light_transform


class Dataset_COCO(data.Dataset):
    def __init__(self, opt):
        super(Dataset_COCO, self).__init__()
        self.opt = opt
        # self.cache_data = opt['cache_data']
        # self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_gt']
        self.gamma_range = opt['gamma_range']
        self.gaussian_range = opt['gaussian_range']
        # self.io_backend_opt = opt['io_backend']
        # self.data_type = opt['io_backend']
        self.path_GT = util.glob_file_list(self.GT_root)
        
        
    def __getitem__(self, index):
        # Load gt and lq images. Dimension order: CHW; channel order: RGB;
        # image range: [0, 1], float32.
        gt_path = self.path_GT[index]
        
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt, self.opt['train_size'])
        img_lq = low_light_transform(img_gt, gamma_range=self.gamma_range, gaussian_range=self.gaussian_range)
        
        img_gt = torch.from_numpy(np.ascontiguousarray(img_gt.transpose(2, 0, 1))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(img_lq.transpose(2, 0, 1))).float()
        img_gt = img_gt / 255.
        img_lq = img_lq / 255.

        # augmentation for training
        if self.opt['phase'] == 'train':
            img_LQ_l = [img_lq]
            img_LQ_l.append(img_gt)
            rlt = util.augment_torch(
                img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_lq = rlt[0]
            img_gt = rlt[1]
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.path_GT)