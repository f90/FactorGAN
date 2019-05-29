import os.path
import random
import torchvision.transforms as transforms
import torch
from datasets.image2image.base_dataset import BaseDataset
from datasets.image2image.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        #assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        #assert(self.opt.loadSize >= self.opt.fineSize)
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)

        if self.opt.direction == 'BtoA':
            input_nc = self.B_nc
            output_nc = self.A_nc
        else:
            input_nc = self.A_nc
            output_nc = self.B_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        #return {'A': A, 'B': B,
        #        'A_paths': AB_path, 'B_paths': AB_path}
        #print(time.time() - t)
        return torch.cat([A,B])

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
