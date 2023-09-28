import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision

from torch.nn.utils import spectral_norm

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

""" ESRGAN backbone 생성자 """
class Generator(nn.Module):
    def __init__(self, scale_factor, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(Generator, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = scale_factor

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
    
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    

""" PatchGAN 식별자 """
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_type='batch'):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers

        '''
        'batch'
        'instance'
        'spectral'
        'batchspectral'
        'none'
        '''
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
            use_sp_norm = False
            use_bias = False
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
            use_sp_norm = False
            use_bias = True
        elif norm_type == 'spectral':
            norm_layer = None
            use_sp_norm = True
            use_bias = True
        elif norm_type == 'batchspectral':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
            use_sp_norm = True
            use_bias = False
        elif norm_type == 'none':
            norm_layer = None
            use_sp_norm = False
            use_bias = True
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [
            self.use_spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_sp_norm),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                self.use_spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_sp_norm),
                norm_layer(ndf * nf_mult) if norm_layer else None,
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            self.use_spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_sp_norm),
            norm_layer(ndf * nf_mult) if norm_layer else None,
            nn.LeakyReLU(0.2)
        ]
        sequence += [self.use_spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), use_sp_norm)]
        
        sequence_new = []
        for n in range(len(sequence)):
            if sequence[n] is not None:
                sequence_new.append(sequence[n])

        self.model = nn.Sequential(*sequence_new)

    def use_spectral_norm(self, module, mode=False):
        if mode:
            return spectral_norm(module)
        return module

    def forward(self, input):
        return self.model(input)
    
    """ ESRGAN backbone 생성자 """
class Generator(nn.Module):
    def __init__(self, scale_factor, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(Generator, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = scale_factor

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
    
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
import os
import random
import torchvision.transforms as transforms
from PIL import Image

class Dataset(object):
    def __init__(self, label_dir, input_dir, image_size, scale):
        self.label_dir = [os.path.join(label_dir, x) for x in os.listdir(label_dir) if self.check_image_file(x)]
        self.input_dir = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if self.check_image_file(x)]

        self.label_dir.sort()
        self.input_dir.sort()

        self.image_size = image_size
        self.to_Tensor = transforms.ToTensor()
        self.resize = transforms.Resize((image_size // scale, image_size // scale), interpolation=Image.BICUBIC)
        self.rotates = [0, 90, 180, 270]
    
    def check_image_file(self, filename: str):
        return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])
    
    """ 데이터 어그멘테이션 """
    def data_augmentation(self, hr, lr):
        """ 이미지 가로 세로 값 """
        width, height = hr.size
        """ 랜덤 이미지 포지션 """
        x = random.randint(0, width - self.image_size)
        y = random.randint(0, height - self.image_size)
        
        """ 크롭 이미지 """
        hr = hr.crop((x, y, x+self.image_size, y+self.image_size))
        lr = lr.crop((x, y, x+self.image_size, y+self.image_size))

        """ 리사이즈 이미지 """
        # hr = hr.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        # lr = lr.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        """ 이미지 회전 """
        deg = random.choice(self.rotates)
        hr = hr.rotate(deg)
        lr = lr.rotate(deg)
        """ 이미지 좌우반전 """
        flip_left_right = random.choice([True, False])
        if flip_left_right:
            hr = hr.transpose(method=Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(method=Image.FLIP_LEFT_RIGHT)
        """ 이미지 상하반전 """
        flip_up_down = random.choice([True, False])
        if flip_up_down:
            hr = hr.transpose(method=Image.FLIP_TOP_BOTTOM)
            lr = lr.transpose(method=Image.FLIP_TOP_BOTTOM)
        return hr, self.resize(lr)

    # lr & hr 이미지를 읽고 크롭하여 lr & hr 이미지를 반환하는 함수
    def __getitem__(self, idx):
        """ 데이터 이름 정보 프린트 """
        #print(f'hr : {self.label_dir[idx]}, lr : {self.input_dir[idx]}')
        hr = Image.open(self.label_dir[idx]).convert("RGB") # HR 이미지 생성
        lr = Image.open(self.input_dir[idx]).convert("RGB") # LR 이미지 생성
        """ 사이즈 프린트 """
        #print(f'hr size : {hr.size}, lr size : {lr.size}')
        hr, lr = self.data_augmentation(hr, lr) # 이미지 어그멘테이션 적용
        return self.to_Tensor(lr), self.to_Tensor(hr) # 데이터 셋 반환

    def __len__(self):
        return len(self.label_dir)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    
class VGGLoss(torch.nn.Module):
    def __init__(self, feature_layer: int = 35) -> None:
        """ Constructing characteristic loss function of VGG network. For VGG19 35th layer. """
        super(VGGLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children())[:feature_layer]).eval()
        # Freeze parameters. Don't train.
        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vgg_loss = torch.nn.functional.l1_loss(self.features(source), self.features(target))

        return vgg_loss
    
#


from PIL import Image
import torchvision.transforms as transforms

###

import os.path
import logging
import torch

from utils import utils_logger
from utils import utils_image as util
# from utils import utils_model
from models.network_rrdbnet import RRDBNet as net


def main():

    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    testsets = 'testsets'       # fixed, set path of testsets
    testset_Ls = ['RealSRSet']  # ['RealSRSet','DPED']

    model_names = ['RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED','RealSR_JPEG']
    model_names = ['BSRGAN']    # 'BSRGANx2' for scale factor 2



    save_results = True
    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in model_names:
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join('model_zoo', model_name+'.pth')          # set model path
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # torch.cuda.set_device(0)      # set GPU ID
        logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

#            model_old = torch.load(model_path)
#            state_dict = model.state_dict()
#            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
#                state_dict[key2] = param
#            model.load_state_dict(state_dict, strict=True)

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        torch.cuda.empty_cache()

        for testset_L in testset_Ls:

            L_path = os.path.join(testsets, testset_L)
            #E_path = os.path.join(testsets, testset_L+'_'+model_name)
            E_path = os.path.join(testsets, testset_L+'_results_x'+str(sf))
            util.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
            idx = 0

            for img in util.get_image_paths(L_path):

                # --------------------------------
                # (1) img_L
                # --------------------------------
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

                img_L = util.imread_uint(img, n_channels=3)
                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)

                # --------------------------------
                # (2) inference
                # --------------------------------
                img_E = model(img_L)

                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(img_E)
                if save_results:
                    util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))


if __name__ == '__main__':

    main()
