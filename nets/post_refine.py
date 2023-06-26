import torch
import torch.nn as nn

from torch.autograd import Variable



inter_channels = [128, 256]
fc_out = inter_channels[1]
fc_unit = 1024
class post_refine(nn.Module):


    def __init__(self, opt):
        super().__init__()

        out_seqlen = 1
        fc_in = opt.out_channels*2*out_seqlen*opt.n_joints
        self.device = opt.device

        fc_out = opt.in_channels * opt.n_joints
        self.post_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),
            nn.ReLU(),
            nn.Dropout(0.5,inplace=False),
            nn.Linear(fc_unit, fc_out),
            nn.Sigmoid()
                )

    # def __init__(self, opt):
    #     super().__init__()

    #     out_seqlen = 1
    #     fc_in = opt['out_channels']*2*out_seqlen*opt['n_joints']
    #     self.device = 'cpu'
    #     # self.device = opt.device

    #     fc_out = opt['in_channels'] * opt['n_joints'] 
    #     self.post_refine = nn.Sequential(
    #         nn.Linear(fc_in, fc_unit),
    #         nn.ReLU(),
    #         nn.Dropout(0.5,inplace=True),
    #         nn.Linear(fc_unit, fc_out),
    #         nn.Sigmoid()
    #     )


    def forward(self, x:torch.tensor, x_1:torch.tensor):
        """

        :param x:  N*T*V*3
        :param x_1: N*T*V*2
        :return:
        """
        # data normalization
        N, T, V,_ = x.size()
        x_in = torch.cat((x, x_1), -1)  #N*T*V*5
        x_in = x_in.view(N, -1)


        device = torch.device(self.device)
        score = self.post_refine(x_in).view(N,T,V,2)
        score_cm = Variable(torch.ones(score.size()), requires_grad=False).to(device) - score
        x_out = x.clone()
        x_out[:, :, :, :2] = score * x[:, :, :, :2] + score_cm * x_1[:, :, :, :2]

        return x_out
    
    
if __name__ == '__main__':
    opt = {}
    opt['in_channels'] = 16
    opt['out_channels'] = 20
    opt['n_joints'] = 16 

    test_1 = torch.randn(2, 16, 8, 3)
    test_2 = torch.randn(2, 16, 8, 2)

    model = post_refine(opt=opt)
    y = model.forward(test_1, test_2)

    pass


