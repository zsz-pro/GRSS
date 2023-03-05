import torch
import torch.nn.functional as F
from torch import nn
import sys
#打印出当前的默认路径
print (sys.path)
#将/Path/to/your/dictionary 改为你程序的路径

sys.path.append('/mnt/e67bb84d-54c9-4502-b46a-154b7875b215/zsz/SOLC/')
# from aspp import _ASPP
# from models.utils import initialize_weights
from models.OurNet.aspp import BasicRFB
from functools import reduce

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if downsample:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class SAGate(nn.Module):
    def __init__(self, channels, out_ch, reduction=16):
        super(SAGate, self).__init__()
        self.channels = channels

        self.fusion1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.gate = nn.Sequential(
            nn.Conv2d(channels , channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

        self.softmax = nn.Softmax(dim=1)

        self.fusion2 = nn.Conv2d(channels * 2, out_ch, kernel_size=1)

    def forward(self, sar, opt):
        b, c, h, w = sar.size()
        output = [sar, opt]

        fea_U = self.fusion1(torch.cat([sar, opt], dim=1))
        fea_s = self.avg_pool(fea_U) + self.max_pool(fea_U)
        attention_vector = self.gate(fea_s)
        attention_vector = attention_vector.reshape(b, 2, self.channels, -1)
        attention_vector = self.softmax(attention_vector)
        attention_vector = list(attention_vector.chunk(2, dim=1))
        attention_vector = list(map(lambda x: x.reshape(b, self.channels, 1, 1), attention_vector))
        V = list(map(lambda x, y: x * y, output, attention_vector))
        # concat + conv
        V = reduce(lambda x, y: self.fusion2(torch.cat([x, y], dim=1)), V)

        return V

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out



class OURV2(nn.Module):
    def __init__(self, num_classes, atrous_rates=[6,12,18]):
        super(OURV2, self).__init__()

        self.sar_en0 = _EncoderBlock(1, 64, downsample=False) # 256->128, 1->64
        self.sar_en1 = _EncoderBlock(64, 128) # 256->128, 1->64
        self.sar_en2 = _EncoderBlock(128, 256)  # 128->64, 64->256
        self.sar_en3 = _EncoderBlock(256, 512)  # 64->32, 256->512
        self.sar_en4 = _EncoderBlock(512, 1024)  # 32->32 *** , 512->1024
        self.sar_en5 = _EncoderBlock(1024, 2048, downsample=False)  # 32->32 *** , 1024->2048

        self.opt_en0 = _EncoderBlock(3, 64, downsample=False) # 256->128, 4->64
        self.opt_en1 = _EncoderBlock(64, 128) # 256->128, 4->64
        self.opt_en2 = _EncoderBlock(128, 256)  # 128->64, 64->256
        self.opt_en3 = _EncoderBlock(256, 512)  # 64->32, 256->512
        self.opt_en4 = _EncoderBlock(512, 1024)  # 32->32 *** , 512->1024
        self.opt_en5 = _EncoderBlock(1024, 2048, downsample=False)  # 32->32 *** , 1024->2048

        self.aspp = BasicRFB(256 * 2, 256)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

        self.low_level_down = SAGate(256, 48)

        self.sar_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.opt_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        out_channels=[2**(i+6) for i in range(5)]#[64, 128, 256, 512, 1024]
        w = 32
        self.h_de1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.h_de2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.h_de3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.h_de4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],1,3,1,1),
            nn.Sigmoid(),)


        initialize_weights(self)

    def forward(self, sar, opt):
        sar_en0 = self.sar_en0(sar)
        sar_en1 = self.sar_en1(sar_en0)
        sar_en2 = self.sar_en2(sar_en1)
        sar_en3 = self.sar_en3(sar_en2)
        sar_en4 = self.sar_en4(sar_en3)
        sar_en5 = self.sar_en5(sar_en4)

        opt_en0 = self.opt_en0(opt)#512-512
        opt_en1 = self.opt_en1(opt_en0)#512-256
        opt_en2 = self.opt_en2(opt_en1)#256-128
        opt_en3 = self.opt_en3(opt_en2)#128-64
        opt_en4 = self.opt_en4(opt_en3)#64-32
        opt_en5 = self.opt_en5(opt_en4)

        low_level_features = self.low_level_down(sar_en2, opt_en2)
        #c,2048->256
        high_level_features_kp = torch.cat([self.sar_high_level_down(sar_en5), self.opt_high_level_down(opt_en5)], 1)

        high_level_features = self.aspp(high_level_features_kp)

        high_level_features = F.upsample(high_level_features, sar_en2.size()[2:], mode='bilinear')

        low_high = torch.cat([low_level_features, high_level_features], 1)

        sar_opt_decoder = self.decoder1(low_high)
        mask = F.upsample(sar_opt_decoder, sar.size()[2:], mode='bilinear')#(b,cl,h,w)
        
        h_de1 = self.h_de1(high_level_features_kp,sar_en3)
        h_de2 = self.h_de2(h_de1,sar_en2)
        h_de3 = self.h_de3(h_de2,sar_en1)#b,256
        h_de4 = self.h_de4(h_de3,sar_en0)#b,128
        height = self.o(h_de4)
        return mask, height


if __name__ == "__main__":
    model = OURV2(num_classes=2)
    model.cuda(3)
    model.train()
    size = 512
    batchsize = 2
    sar = torch.randn(batchsize, 1, size, size).cuda(3)
    opt = torch.randn(batchsize, 3, size, size).cuda(3)
    mask,height = model(sar,opt)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", mask.shape,height.shape)
