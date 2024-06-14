import torch
import torch.nn as nn
from .layers import ResidualBlockWithStride

class IntraPredictorModel(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.in_ch = 128
        self.h_ch1 = 192
        self.h_ch2 = 256

        self.global_h = 8
        self.global_w = 8

        self.inner_dim1 = 512
        self.inner_dim2 = 256

        self.inner_psnr_dim = 10

        self.lstm_inner_dim = self.inner_dim2 + self.inner_psnr_dim
    
        self.res_block1 = ResidualBlockWithStride(self.in_ch, self.h_ch1, stride=2)
        self.bn1 = nn.BatchNorm2d(self.h_ch1)
        self.res_block2 = ResidualBlockWithStride(self.h_ch1, self.h_ch2, stride=2)
        self.bn2 = nn.BatchNorm2d(self.h_ch2)
        self.global_pooling = nn.AdaptiveAvgPool2d((self.global_w, self.global_h))
        self.linear1 = nn.Linear(self.global_h * self.global_w * self.h_ch2, self.inner_dim1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(self.inner_dim1, self.inner_dim2)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.lstm = nn.LSTM(self.lstm_inner_dim, self.lstm_inner_dim, batch_first=True) # [N, L, inner_dim2]
        self.linear_out = nn.Linear(self.lstm_inner_dim, 1)
        self.softmax = nn.Sigmoid()

        self.psnr_linear = nn.Linear(1, 10)

        self.hidden = None

    def reset_state(self):
        self.hidden = None

    def forward(self, x, psnr):
        # x: [N, 128, H, W]
        # psnr: [N, 1]

        # Batch first
        is_single_frame = x.size(0) == 1

        x = self.res_block1(x)
        x = self.bn1(x)
        x = self.res_block2(x)
        x = self.bn2(x)

        x = self.global_pooling(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.leaky_relu(x)


        psnr = self.psnr_linear(psnr)
        psnr = self.leaky_relu(psnr)

        x = torch.cat([x, psnr], dim=1)

        x = x.unsqueeze(0)

        if is_single_frame:
            if self.hidden is None:
                self.hidden = (torch.zeros(1, 1, self.lstm_inner_dim).to(x.device).to(x.dtype),
                               torch.zeros(1, 1, self.lstm_inner_dim).to(x.device).to(x.dtype))
            x, self.hidden = self.lstm(x, self.hidden)
        else:
            x, _ = self.lstm(x)

        x = x.squeeze(0)

        x = self.linear_out(x)
        logits = self.softmax(x)

        return x, logits