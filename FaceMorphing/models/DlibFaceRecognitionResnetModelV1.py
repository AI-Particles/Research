from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

import dlib_routines_port as dp


class DlibFaceRecognitionResnetModelV1(nn.Module):
    def __init__(self, no_grad_mode=True):
        super(DlibFaceRecognitionResnetModelV1, self).__init__()

        self.no_grad_mode = no_grad_mode

        self.input_rgb_image_sized131 = dp.InputRgbImage()
        self.con130 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con129 = torch.nn.BatchNorm2d(num_features=32)
        self.relu128 = nn.ReLU()
        self.max_pool127 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.con125 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con124 = torch.nn.BatchNorm2d(num_features=32)
        self.relu123 = nn.ReLU()
        self.con122 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con121 = torch.nn.BatchNorm2d(num_features=32)
        self.add_prev120 = dp.AddPrev()
        self.relu119 = nn.ReLU()
        self.con117 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con116 = torch.nn.BatchNorm2d(num_features=32)
        self.relu115 = nn.ReLU()
        self.con114 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con113 = torch.nn.BatchNorm2d(num_features=32)
        self.add_prev112 = dp.AddPrev()
        self.relu111 = nn.ReLU()
        self.con109 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con108 = torch.nn.BatchNorm2d(num_features=32)
        self.relu107 = nn.ReLU()
        self.con106 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con105 = torch.nn.BatchNorm2d(num_features=32)
        self.add_prev104 = dp.AddPrev()
        self.relu103 = nn.ReLU()
        self.con101 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con100 = torch.nn.BatchNorm2d(num_features=64)
        self.relu99 = nn.ReLU()
        self.con98 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con97 = torch.nn.BatchNorm2d(num_features=64)
        self.avg_pool94 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.add_prev93 = dp.AddPrev()
        self.relu92 = nn.ReLU()
        self.con90 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con89 = torch.nn.BatchNorm2d(num_features=64)
        self.relu88 = nn.ReLU()
        self.con87 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con86 = torch.nn.BatchNorm2d(num_features=64)
        self.add_prev85 = dp.AddPrev()
        self.relu84 = nn.ReLU()
        self.con82 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con81 = torch.nn.BatchNorm2d(num_features=64)
        self.relu80 = nn.ReLU()
        self.con79 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con78 = torch.nn.BatchNorm2d(num_features=64)
        self.add_prev77 = dp.AddPrev()
        self.relu76 = nn.ReLU()
        self.con74 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con73 = torch.nn.BatchNorm2d(num_features=64)
        self.relu72 = nn.ReLU()
        self.con71 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con70 = torch.nn.BatchNorm2d(num_features=64)
        self.add_prev69 = dp.AddPrev()
        self.relu68 = nn.ReLU()
        self.con66 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con65 = torch.nn.BatchNorm2d(num_features=128)
        self.relu64 = nn.ReLU()
        self.con63 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con62 = torch.nn.BatchNorm2d(num_features=128)
        self.avg_pool59 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.add_prev58 = dp.AddPrev()
        self.relu57 = nn.ReLU()
        self.con55 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con54 = torch.nn.BatchNorm2d(num_features=128)
        self.relu53 = nn.ReLU()
        self.con52 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con51 = torch.nn.BatchNorm2d(num_features=128)
        self.add_prev50 = dp.AddPrev()
        self.relu49 = nn.ReLU()
        self.con47 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con46 = torch.nn.BatchNorm2d(num_features=128)
        self.relu45 = nn.ReLU()
        self.con44 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con43 = torch.nn.BatchNorm2d(num_features=128)
        self.add_prev42 = dp.AddPrev()
        self.relu41 = nn.ReLU()
        self.con39 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con38 = torch.nn.BatchNorm2d(num_features=256)
        self.relu37 = nn.ReLU()
        self.con36 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con35 = torch.nn.BatchNorm2d(num_features=256)
        self.avg_pool32 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.add_prev31 = dp.AddPrev()
        self.relu30 = nn.ReLU()
        self.con28 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con27 = torch.nn.BatchNorm2d(num_features=256)
        self.relu26 = nn.ReLU()
        self.con25 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con24 = torch.nn.BatchNorm2d(num_features=256)
        self.add_prev23 = dp.AddPrev()
        self.relu22 = nn.ReLU()
        self.con20 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con19 = torch.nn.BatchNorm2d(num_features=256)
        self.relu18 = nn.ReLU()
        self.con17 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con16 = torch.nn.BatchNorm2d(num_features=256)
        self.add_prev15 = dp.AddPrev()
        self.relu14 = nn.ReLU()
        self.con12 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con11 = torch.nn.BatchNorm2d(num_features=256)
        self.relu10 = nn.ReLU()
        self.con9 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con8 = torch.nn.BatchNorm2d(num_features=256)
        self.avg_pool5 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.add_prev4 = dp.AddPrev()
        self.relu3 = nn.ReLU()
        self.avg_pool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.pre_fc_no_bias1 = torch.nn.Flatten()
        self.fc_no_bias1 = torch.nn.Linear(in_features=256, out_features=128, bias=False)

    def forward(self, input):
        with torch.no_grad() if self.no_grad_mode else nullcontext():
            input_rgb_image_sized131_out = self.input_rgb_image_sized131(input)
            con130_out = self.con130(input_rgb_image_sized131_out)
            affine_con129_out = self.affine_con129(con130_out)
            relu128_out = self.relu128(affine_con129_out)
            max_pool127_out = self.max_pool127(relu128_out)
            con125_out = self.con125(max_pool127_out)
            affine_con124_out = self.affine_con124(con125_out)
            relu123_out = self.relu123(affine_con124_out)
            con122_out = self.con122(relu123_out)
            affine_con121_out = self.affine_con121(con122_out)
            add_prev120_out = self.add_prev120(affine_con121_out, max_pool127_out)
            relu119_out = self.relu119(add_prev120_out)
            con117_out = self.con117(relu119_out)
            affine_con116_out = self.affine_con116(con117_out)
            relu115_out = self.relu115(affine_con116_out)
            con114_out = self.con114(relu115_out)
            affine_con113_out = self.affine_con113(con114_out)
            add_prev112_out = self.add_prev112(affine_con113_out, relu119_out)
            relu111_out = self.relu111(add_prev112_out)
            con109_out = self.con109(relu111_out)
            affine_con108_out = self.affine_con108(con109_out)
            relu107_out = self.relu107(affine_con108_out)
            con106_out = self.con106(relu107_out)
            affine_con105_out = self.affine_con105(con106_out)
            add_prev104_out = self.add_prev104(affine_con105_out, relu111_out)
            relu103_out = self.relu103(add_prev104_out)
            con101_out = self.con101(relu103_out)
            affine_con100_out = self.affine_con100(con101_out)
            relu99_out = self.relu99(affine_con100_out)
            con98_out = self.con98(relu99_out)
            affine_con97_out = self.affine_con97(con98_out)
            avg_pool94_out = self.avg_pool94(relu103_out)
            padded_avg_pool94_out = F.pad(input=avg_pool94_out, pad=[0, 0, 0, 0, 0, 32, 0, 0], mode='constant', value=0)
            add_prev93_out = self.add_prev93(padded_avg_pool94_out, affine_con97_out)
            relu92_out = self.relu92(add_prev93_out)
            con90_out = self.con90(relu92_out)
            affine_con89_out = self.affine_con89(con90_out)
            relu88_out = self.relu88(affine_con89_out)
            con87_out = self.con87(relu88_out)
            affine_con86_out = self.affine_con86(con87_out)
            add_prev85_out = self.add_prev85(affine_con86_out, relu92_out)
            relu84_out = self.relu84(add_prev85_out)
            con82_out = self.con82(relu84_out)
            affine_con81_out = self.affine_con81(con82_out)
            relu80_out = self.relu80(affine_con81_out)
            con79_out = self.con79(relu80_out)
            affine_con78_out = self.affine_con78(con79_out)
            add_prev77_out = self.add_prev77(affine_con78_out, relu84_out)
            relu76_out = self.relu76(add_prev77_out)
            con74_out = self.con74(relu76_out)
            affine_con73_out = self.affine_con73(con74_out)
            relu72_out = self.relu72(affine_con73_out)
            con71_out = self.con71(relu72_out)
            affine_con70_out = self.affine_con70(con71_out)
            add_prev69_out = self.add_prev69(affine_con70_out, relu76_out)
            relu68_out = self.relu68(add_prev69_out)
            con66_out = self.con66(relu68_out)
            affine_con65_out = self.affine_con65(con66_out)
            relu64_out = self.relu64(affine_con65_out)
            con63_out = self.con63(relu64_out)
            affine_con62_out = self.affine_con62(con63_out)
            avg_pool59_out = self.avg_pool59(relu68_out)
            padded_avg_pool59_out = F.pad(input=avg_pool59_out, pad=[0, 0, 0, 0, 0, 64, 0, 0], mode='constant', value=0)
            add_prev58_out = self.add_prev58(padded_avg_pool59_out, affine_con62_out)
            relu57_out = self.relu57(add_prev58_out)
            con55_out = self.con55(relu57_out)
            affine_con54_out = self.affine_con54(con55_out)
            relu53_out = self.relu53(affine_con54_out)
            con52_out = self.con52(relu53_out)
            affine_con51_out = self.affine_con51(con52_out)
            add_prev50_out = self.add_prev50(affine_con51_out, relu57_out)
            relu49_out = self.relu49(add_prev50_out)
            con47_out = self.con47(relu49_out)
            affine_con46_out = self.affine_con46(con47_out)
            relu45_out = self.relu45(affine_con46_out)
            con44_out = self.con44(relu45_out)
            affine_con43_out = self.affine_con43(con44_out)
            add_prev42_out = self.add_prev42(affine_con43_out, relu49_out)
            relu41_out = self.relu41(add_prev42_out)
            con39_out = self.con39(relu41_out)
            affine_con38_out = self.affine_con38(con39_out)
            relu37_out = self.relu37(affine_con38_out)
            con36_out = self.con36(relu37_out)
            affine_con35_out = self.affine_con35(con36_out)
            avg_pool32_out = self.avg_pool32(relu41_out)
            padded_avg_pool32_out = F.pad(input=avg_pool32_out, pad=[0, 0, 0, 0, 0, 128, 0, 0], mode='constant', value=0)
            padded_affine_con35_out = F.pad(input=affine_con35_out, pad=[0, 1, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
            add_prev31_out = self.add_prev31(padded_avg_pool32_out, padded_affine_con35_out)
            relu30_out = self.relu30(add_prev31_out)
            con28_out = self.con28(relu30_out)
            affine_con27_out = self.affine_con27(con28_out)
            relu26_out = self.relu26(affine_con27_out)
            con25_out = self.con25(relu26_out)
            affine_con24_out = self.affine_con24(con25_out)
            add_prev23_out = self.add_prev23(affine_con24_out, relu30_out)
            relu22_out = self.relu22(add_prev23_out)
            con20_out = self.con20(relu22_out)
            affine_con19_out = self.affine_con19(con20_out)
            relu18_out = self.relu18(affine_con19_out)
            con17_out = self.con17(relu18_out)
            affine_con16_out = self.affine_con16(con17_out)
            add_prev15_out = self.add_prev15(affine_con16_out, relu22_out)
            relu14_out = self.relu14(add_prev15_out)
            con12_out = self.con12(relu14_out)
            affine_con11_out = self.affine_con11(con12_out)
            relu10_out = self.relu10(affine_con11_out)
            con9_out = self.con9(relu10_out)
            affine_con8_out = self.affine_con8(con9_out)
            avg_pool5_out = self.avg_pool5(relu14_out)
            padded_affine_con8_out = F.pad(input=affine_con8_out, pad=[0, 1, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
            add_prev4_out = self.add_prev4(avg_pool5_out, padded_affine_con8_out)
            relu3_out = self.relu3(add_prev4_out)
            avg_pool2_out = self.avg_pool2(relu3_out)
            pre_fc_no_bias1_out = self.pre_fc_no_bias1(avg_pool2_out)
            fc_no_bias1_out = self.fc_no_bias1(pre_fc_no_bias1_out)

            return fc_no_bias1_out