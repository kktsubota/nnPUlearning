from chainer import Variable
import numpy as np
import torch
import sys


def test():
    x = np.arange(10).reshape(10, 1).astype(np.float32)
    t = np.arange(10).astype(np.int32) % 2 * 2 - 1
    # when prior = 0.99: negative risk is smaller than 0
    priors = {0.1, 0.99}

    def test_chainer(x, t, prior):
        x, t = Variable(x), Variable(t)

        sys.path = ['../nnPUlearning-chainer'] + sys.path
        from pu_loss import pu_loss as pu_loss_chainer
        loss = pu_loss_chainer(x, t, prior=prior, nnpu=True)
        loss.backward()
        return loss, x.grad

    def test_torch(x, t, prior):
        x, t = torch.from_numpy(x), torch.from_numpy(t)
        x.requires_grad = True

        from pu_loss import pu_loss as pu_loss_torch
        loss, loss_total = pu_loss_torch(x, t, prior, nnpu=True)
        loss.backward()
        return loss_total, x.grad
    
    for prior in priors:
        test_chainer(x, t, prior) == test_torch(x, t, prior)
        