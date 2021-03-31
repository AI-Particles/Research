from contextlib import nullcontext

import torch
import torch.nn as nn

import dlib_routines_port as dp


class MmodHumanFaceDetector(nn.Module):
    def __init__(self, no_grad_mode=True):
        super(MmodHumanFaceDetector, self).__init__()

        self.no_grad_mode = no_grad_mode

        self.conv_layers_params = [[1, 1, 4, 4, 4, 4], [1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 2, 2], [2, 2, 0, 0, 2, 2], [2, 2, 0, 0, 2, 2], [2, 2, 0, 0, 2, 2]]

        self.input_rgb_image_pyramid20 = dp.InputRgbImagePyramyd()
        self.con19 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con18 = torch.nn.BatchNorm2d(num_features=16)
        self.relu17 = nn.ReLU()
        self.con16 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con15 = torch.nn.BatchNorm2d(num_features=32)
        self.relu14 = nn.ReLU()
        self.con13 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), bias=True)
        self.affine_con12 = torch.nn.BatchNorm2d(num_features=32)
        self.relu11 = nn.ReLU()
        self.con10 = torch.nn.Conv2d(in_channels=32, out_channels=45, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.affine_con9 = torch.nn.BatchNorm2d(num_features=45)
        self.relu8 = nn.ReLU()
        self.con7 = torch.nn.Conv2d(in_channels=45, out_channels=45, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.affine_con6 = torch.nn.BatchNorm2d(num_features=45)
        self.relu5 = nn.ReLU()
        self.con4 = torch.nn.Conv2d(in_channels=45, out_channels=45, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.affine_con3 = torch.nn.BatchNorm2d(num_features=45)
        self.relu2 = nn.ReLU()
        self.con1 = torch.nn.Conv2d(in_channels=45, out_channels=1, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), bias=True)

    def forward(self, input):
        with torch.no_grad() if self.no_grad_mode else nullcontext():
            input_rgb_image_pyramid20_out, pyramyd_rects = self.input_rgb_image_pyramid20(input)
            con19_out = self.con19(input_rgb_image_pyramid20_out)
            affine_con18_out = self.affine_con18(con19_out)
            relu17_out = self.relu17(affine_con18_out)
            con16_out = self.con16(relu17_out)
            affine_con15_out = self.affine_con15(con16_out)
            relu14_out = self.relu14(affine_con15_out)
            con13_out = self.con13(relu14_out)
            affine_con12_out = self.affine_con12(con13_out)
            relu11_out = self.relu11(affine_con12_out)
            con10_out = self.con10(relu11_out)
            affine_con9_out = self.affine_con9(con10_out)
            relu8_out = self.relu8(affine_con9_out)
            con7_out = self.con7(relu8_out)
            affine_con6_out = self.affine_con6(con7_out)
            relu5_out = self.relu5(affine_con6_out)
            con4_out = self.con4(relu5_out)
            affine_con3_out = self.affine_con3(con4_out)
            relu2_out = self.relu2(affine_con3_out)
            con1_out = self.con1(relu2_out)

            faces_from_image = dp.to_label(con1_out, pyramyd_rects, self.conv_layers_params, adjust_threshold=0.0)

            return faces_from_image
