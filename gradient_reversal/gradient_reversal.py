import torch

class GradientReversal(torch.autograd.Function):

    def __init__(self, scale_):
        super(GradientReversal, self).__init__()

        self.scale = scale_

    def forward(self, inp):
        return inp.clone()

    def backward(self, grad_out):
        return -self.scale * grad_out.clone()

