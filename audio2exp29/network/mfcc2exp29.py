import os
from argparse import ArgumentParser
from math import cos, sin
from os.path import join
from random import randint

import numpy as np
import torch
import torch.nn as nn
# from network import init_net
from scipy.io import loadmat
import cv2

# from .base_model import BaseModel
from ..util.draw import draw_landmarks


# class MFCC2Exp29(BaseModel):

#     @staticmethod
#     def modify_commandline_options(parser: ArgumentParser, isTrain):
#         # parser.add_argument('--n_params', default=10, type=int, help='The dimension of expression coeff.')
#         parser.set_defaults(lambda_smooth=0.0000, lambda_landmark=0.01)
#         return parser

#     def __init__(self, opt):
#         super(MFCC2Exp29, self).__init__(opt)
#         self.at_net = init_net(AT_net(29), gpu_ids=opt.gpu_ids)

#         self.loss_names = ['mse', 'pred_smooth', 'atnet', 'target_smooth']
#         self.visual_names = ['pred_landmark_image', 'target_landmark_image']
#         self.optimizers = []
#         self.model_names = ['at_net']

#         if self.opt.isTrain:
#             self.mse = nn.MSELoss().to(self.device)

#             self.optimizer_atnet = torch.optim.Adam(
#                 self.at_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_atnet)
#         elif self.opt.phase == 'inference':
#             self.pred_param_dir = join(self.opt.results_dir, 'pred_param')
#             os.makedirs(self.pred_param_dir, exist_ok=True)

#         # BFM Model visulize
#         self.bfm = MorphabelModel('data/BFM/Out/BFM.mat')
#         self.template_shp = self.bfm.get_shape_para(type='random')

#         # if opt.normalized_param:
#         #     self.mean_std_param = _load('...')
#         #     self.mean_param = self.mean_std_param['mean']
#         #     self.std_param = self.mean_std_param['std']

#     def set_input(self, input):
#         """Set input to the network.
#         args:
#             input: A dict of data with the following key-values:
#                 mfcc: ...
#                 param: ...
#         """
#         self.mfcc = input['mfcc_seq'].to(self.device)
#         if self.opt.isTrain:
#             self.target_expression_coeff = input['param_seq'].to(self.device)
#         elif self.opt.phase == 'test':
#             self.target_expression_coeff = input['param_seq'].to(self.device)
#         self.image_path_seq = input['image_idx_seq']

#     def forward(self):
#         self.pred_expression_coeff = self.at_net(self.mfcc, device=self.device)

#     def backward(self):
#         self.loss_mse = self.mse(self.pred_expression_coeff, self.target_expression_coeff)
#         # add time smooth
#         diff = self.pred_expression_coeff[:, 1:, :] - self.pred_expression_coeff[:, :-1, :]
#         diff = torch.sum(torch.mul(diff, diff), (1, 2))
#         self.loss_pred_smooth = torch.mean(diff)

#         diff = self.target_expression_coeff[:, 1:, :] - self.target_expression_coeff[:, :-1, :]
#         diff = torch.sum(torch.mul(diff, diff), (1, 2))
#         self.loss_target_smooth = torch.mean(diff)

#         self.loss_atnet = self.loss_mse + self.opt.lambda_smooth * self.loss_pred_smooth
#         self.loss_atnet.backward()

#     def optimize_parameters(self):
#         self.forward()
#         self.optimizer_atnet.zero_grad()
#         self.backward()
#         self.optimizer_atnet.step()

#     def compute_losses(self):
#         self.loss_mse = self.mse(self.pred_expression_coeff, self.target_expression_coeff)
#         # add time smooth
#         diff = self.pred_expression_coeff[:, 1:, :] - self.pred_expression_coeff[:, :-1, :]
#         diff = torch.sum(torch.mul(diff, diff), (1, 2))
#         self.loss_pred_smooth = torch.mean(diff)

#         diff = self.target_expression_coeff[:, 1:, :] - self.target_expression_coeff[:, :-1, :]
#         diff = torch.sum(torch.mul(diff, diff), (1, 2))
#         self.loss_target_smooth = torch.mean(diff)

#         self.loss_atnet = self.loss_mse + self.opt.lambda_smooth * self.loss_pred_smooth

#     def compute_visuals(self):
#         r = randint(0, self.opt.length_seq - 1)
#         # if self.opt.normalized_param:
#         #     pass

#         # pred
#         # print(self.pred_expression_coeff.shape)
#         pred_param = self.pred_expression_coeff[0, r].cpu().detach().numpy().reshape(-1, 1)
#         vers = self.bfm.generate_landmark(self.template_shp, pred_param)
#         vers = transform_to_roibox(vers, (0, 0, 255, 255))
#         pred_landmark_image = np.zeros((255, 255, 3), dtype=np.uint8)
#         self.pred_landmark_image = draw_landmarks(pred_landmark_image, vers.T)

#         # target
#         # print(self.target_expression_coeff.shape)
#         target_param = self.target_expression_coeff[0, r].cpu().detach().numpy().reshape(-1, 1)
#         vers = self.bfm.generate_landmark(self.template_shp, target_param)
#         vers = transform_to_roibox(vers, (0, 0, 255, 255))
#         target_landmark_image = np.zeros((255, 255, 3), dtype=np.uint8)
#         self.target_landmark_image = draw_landmarks(target_landmark_image, vers.T)

#         self.image_paths = [self.image_path_seq[r][0] for visual_name in self.visual_names]

#     @torch.no_grad()
#     def inference(self):
#         """python neural_renderer/inference.py --dataroot /media/liublack/data/datasets/LRW/lrw_audio_exp --model mfcc2exp29 --dataset_mode mfcc2exp29 --length_seq 16 --gpu_ids 0 --name mfcc2exp29 --audio """
#         """
#         """
#         self.forward()
#         pred_param = self.pred_expression_coeff[0].cpu().numpy()  # at inference phase, batchsize must be one
#         for step_t in range(pred_param.shape[0]):
#             param = pred_param[step_t]
#             frame_id = int(self.image_path_seq[step_t][0])
#             np.savetxt(join(self.pred_param_dir, '%05d.txt' % frame_id), param)


class AT_net(nn.Module):
    """The AT_net network descriped in https://github.com/lelechen63/ATVGnet.
    We modify it's output dimension to 10 and drop the landmark encoder.
    """

    def __init__(self, n_output=10):
        super(AT_net, self).__init__()

        self.audio_eocder = nn.Sequential(
            conv2d(1, 64, 3, 1, 1),
            conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(3, stride=(1, 2)),
            conv2d(128, 256, 3, 1, 1),
            conv2d(256, 256, 3, 1, 1),
            conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(3, stride=(2, 2))
        )
        self.audio_eocder_fc = nn.Sequential(
            # nn.Linear(1024 * 12, 2048),
            nn.Linear(20480, 2048),
            # nn.Linear(7680, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 256),
            nn.ReLU(True))
        self.lstm = nn.LSTM(256, 256, 3, batch_first=True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256, n_output),
        )

    # def forward(self, audio, device='cpu'):
    #     # print(audio.shape)
    #     hidden = (torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).to(device)),
    #               torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).to(device)))
    #     lstm_input = []
    #     for step_t in range(audio.size(1)):
    #         current_audio = audio[:, step_t, :, :].unsqueeze(1)
    #         current_feature = self.audio_eocder(current_audio)
    #         current_feature = current_feature.view(current_feature.size(0), -1)
    #         # print(current_feature.shape)
    #         current_feature = self.audio_eocder_fc(current_feature)
    #         lstm_input.append(current_feature)
    #     lstm_input = torch.stack(lstm_input, dim=1)
    #     lstm_out, out_hidden = self.lstm(lstm_input, hidden)
    #     fc_out = []
    #     for step_t in range(audio.size(1)):
    #         fc_in = lstm_out[:, step_t, :]
    #         fc_out.append(self.lstm_fc(fc_in))
    #     return torch.stack(fc_out, dim=1)

    def forward(self, audio, hidden=None):
        if hidden is None:
            hidden = (torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).to(device)),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).to(device)))
        current_audio = audio.unsqueeze(1)
        current_feature = self.audio_eocder(current_audio)
        current_feature = current_feature.view(current_feature.size(0), -1)
        current_feature = self.audio_eocder_fc(current_feature)

        lstm_input = torch.stack([current_feature], dim=1)
        lstm_out, out_hidden = self.lstm(lstm_input, hidden)
        fc_in = lstm_out[:, 0, :]
        fc_out = self.lstm_fc(fc_in)
        return fc_out, out_hidden


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out, ksize, stride, padding, bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        layer.append(activation())
    return layer


