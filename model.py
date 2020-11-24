import numpy as np
import torch
from collections import deque
import cfg
from audio2exp29.network import AT_net
from audio2exp29.util.bfm import transform_to_roibox, MorphabelModel
from audio2exp29.util.tddfa import BFMModel, mean_param, std_param
from audio2exp29.util.draw import draw_landmarks


class Model():
    def __init__(self, mfcc_num_context=9, mfcc_numcep=26):
        self.device = torch.device('cuda:1')
        state_dict = torch.load(cfg.ckpt, map_location=torch.device('cuda:1'))
        self.atnet = AT_net(10).to(self.device).eval()
        self.atnet.load_state_dict(state_dict)

        self.bfm = MorphabelModel(cfg.bfm.bfm_path)
        self.bfm_template_param = self.bfm.get_shape_para(type='random')
        self.tddfa = BFMModel(cfg.tddfa.bfm_path)
        self.tddfa_template_param = np.loadtxt(cfg.tddfa.template_param_path)

        self.hidden = (torch.autograd.Variable(torch.zeros(3, 1, 256).to(self.device)),
                       torch.autograd.Variable(torch.zeros(3, 1, 256).to(self.device)))
        self.mfcc_count = 0
        self.mfcc_history_winlen = 2 * mfcc_num_context + 1
        self.mfcc_history = deque(
            [np.zeros(mfcc_numcep) for i in range(self.mfcc_history_winlen)],
            maxlen=self.mfcc_history_winlen
        )

    @torch.no_grad()
    def update_mfcc(self, mfcc):
        self.mfcc_history.append(mfcc)
        self.mfcc_count += 1
        if self.mfcc_count % 4 == 1:
            mfccs = np.stack(list(self.mfcc_history), axis=0)
            mfccs = torch.from_numpy(mfccs).unsqueeze(0).float().to(self.device)
            self.alpha_exp, self.hidden = self.atnet(mfccs, self.hidden)
        if self.mfcc_count % 32 == 0:
            self.hidden = (torch.autograd.Variable(torch.zeros(3, 1, 256).to(self.device)),
                           torch.autograd.Variable(torch.zeros(3, 1, 256).to(self.device)))

    @torch.no_grad()
    def render(self, frame):
        alpha_exp = self.alpha_exp.cpu().numpy().reshape(-1)
        alpha_exp = alpha_exp * std_param[52:] + mean_param[52:] 
        if alpha_exp.shape[0] == 29:
            vers = self.bfm.generate_landmark(self.template_shp, alpha_exp.reshape(-1, 1))
            vers = transform_to_roibox(vers, (0, 0, 255, 255))
            vers = vers.T
        else:
            face_param = np.concatenate([self.tddfa_template_param[:52], alpha_exp], axis=0)
            vers = self.tddfa.recon_vers(face_param, (0, 0, 255, 255))
        pred_landmark_image = np.zeros((255, 255, 3), dtype=np.uint8)
        pred_landmark_image = draw_landmarks(pred_landmark_image, vers)
        return pred_landmark_image
