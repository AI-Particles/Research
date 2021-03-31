from contextlib import nullcontext

import torch
import torch.nn as nn

import dlib_routines_port as dp


class GenderPredictorNet(nn.Module):
    def __init__(self, no_grad_mode=True):
        super(GenderPredictorNet, self).__init__()

        self.no_grad_mode = no_grad_mode

        self.input_rgb_image_sized20 = dp.InputRgbImage()
        self.con19 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con18 = torch.nn.BatchNorm2d(num_features=32)
        self.relu17 = nn.ReLU()
        self.con16 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con15 = torch.nn.BatchNorm2d(num_features=32)
        self.relu14 = nn.ReLU()
        self.avg_pool13 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.con12 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con11 = torch.nn.BatchNorm2d(num_features=64)
        self.relu10 = nn.ReLU()
        self.con9 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.affine_con8 = torch.nn.BatchNorm2d(num_features=64)
        self.relu7 = nn.ReLU()
        self.avg_pool6 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.multiply5 = dp.Multiply(factor=0.5)
        self.pre_fc4 = torch.nn.Flatten()
        self.fc4 = torch.nn.Linear(in_features=4096, out_features=16, bias=True)
        self.relu3 = nn.ReLU()
        self.multiply2 = dp.Multiply(factor=0.5)
        self.fc1 = torch.nn.Linear(in_features=16, out_features=2, bias=True)
        self.loss_multiclass_log0 = torch.nn.Softmax(dim=-1)

    def forward(self, input):
        with torch.no_grad() if self.no_grad_mode else nullcontext():
            input_rgb_image_sized20_out = self.input_rgb_image_sized20(input)
            con19_out = self.con19(input_rgb_image_sized20_out)
            affine_con18_out = self.affine_con18(con19_out)
            relu17_out = self.relu17(affine_con18_out)
            con16_out = self.con16(relu17_out)
            affine_con15_out = self.affine_con15(con16_out)
            relu14_out = self.relu14(affine_con15_out)
            avg_pool13_out = self.avg_pool13(relu14_out)
            con12_out = self.con12(avg_pool13_out)
            affine_con11_out = self.affine_con11(con12_out)
            relu10_out = self.relu10(affine_con11_out)
            con9_out = self.con9(relu10_out)
            affine_con8_out = self.affine_con8(con9_out)
            relu7_out = self.relu7(affine_con8_out)
            avg_pool6_out = self.avg_pool6(relu7_out)
            multiply5_out = self.multiply5(avg_pool6_out)
            pre_fc4_out = self.pre_fc4(multiply5_out)
            fc4_out = self.fc4(pre_fc4_out)
            relu3_out = self.relu3(fc4_out)
            multiply2_out = self.multiply2(relu3_out)
            fc1_out = self.fc1(multiply2_out)
            loss_multiclass_log0_out = self.loss_multiclass_log0(fc1_out)
            output = loss_multiclass_log0_out.detach().cpu().numpy()

            gender, gender_confidence = dp.get_estimated_gender(output)

            return gender, gender_confidence
