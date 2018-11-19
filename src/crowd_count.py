import torch.nn as nn
import network
from models import MCNN


class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.DME = MCNN()
        self.loss_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self,  im_data, gt_data=None, mask=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        assert mask is not None:
        mask_var = network.np_to_variable(mask, is_cuda=True, is_training=self.training)
        # else:
        #     mask_var = 1.0
        density_map = self.DME(im_data) * mask_var

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training) * mask_var
            self.loss_mse = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss
