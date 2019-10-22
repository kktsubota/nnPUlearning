import torch
from torch.nn import functional as F


class PULoss(torch.nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnpu=True):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnpu = nnpu
        self.positive = 1
        self.unlabeled = -1

    def forward(self, x, t):
        return pu_loss(x, t, self.prior, self.loss_func, gamma=self.gamma, beta=self.beta, nnpu=self.nnpu)


class PULossFunction(torch.autograd.Function):
    """wrapper of loss function for PU learning"""

    def __init__(self):
        super(PULossFunction, self).__init__()

    @staticmethod
    def forward(ctx, x, t, prior, loss_func=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnpu=True):
        x_in = x
        t = t[:, None]

        positive, unlabeled = (t == 1), (t == -1)
        n_positive, n_unlabeled = max([1., positive.sum()]), max([1., unlabeled.sum()])

        y_positive = loss_func(x_in)
        y_unlabeled = loss_func(-x_in)  # sigmoid
        positive_risk = torch.sum(prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - prior * positive / n_positive) * y_unlabeled)
        objective = positive_risk + negative_risk
        if nnpu:
            if negative_risk.data < -beta:
                objective = positive_risk - beta
                x_out = -gamma * negative_risk
            else:
                x_out = objective
        else:
            x_out = objective

        ctx.save_for_backward(x_in, x_out)
        return objective

    @staticmethod
    def backward(ctx, grad_output):
        x_in, x_out = ctx.saved_tensors
        x_out.backward()
        grad_input = grad_output[0].reshape(grad_output[0].shape + (1,) * (x_in.ndim - 1)) * x_in.grad
        return grad_input


def pu_loss(x, t, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnpu=True):
    """wrapper of loss function for non-negative/unbiased PU learning
        .. math::
            \\begin{array}{lc}
            L_[\\pi E_1[l(f(x))]+\\max(E_X[l(-f(x))]-\\pi E_1[l(-f(x))], \\beta) & {\\rm if nnPU learning}\\\\
            L_[\\pi E_1[l(f(x))]+E_X[l(-f(x))]-\\pi E_1[l(-f(x))] & {\\rm otherwise}
            \\end{array}
    Args:
        x (~chainer.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, 1).
        t (~chainer.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, ).
        prior (float): Constant variable for class prior.
        loss (~chainer.function): loss function.
            The loss function should be non-increasing.
        nnpu (bool): Whether use non-negative PU learning or unbiased PU learning.
            In default setting, non-negative PU learning will be used.
    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            PU loss.
    See:
        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
        "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
        Advances in neural information processing systems. 2017.
        du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama.
        "Convex formulation for learning from positive and unlabeled data."
        Proceedings of The 32nd International Conference on Machine Learning. 2015.
    """
    x_in, t = x, t[:, None]
    positive, unlabeled = (t == 1), (t == -1)
    n_positive, n_unlabeled = max([1., positive.sum()]), max([1., unlabeled.sum()])

    y_positive = loss(x_in)
    y_unlabeled = loss(-x_in)  # sigmoid
    positive_risk = torch.sum(prior * positive / n_positive * y_positive)
    negative_risk = torch.sum((unlabeled / n_unlabeled - prior * positive / n_positive) * y_unlabeled)
    objective = positive_risk + negative_risk
    if nnpu and negative_risk.data < -beta:
        objective = positive_risk - beta
        x_out = -gamma * negative_risk
    else:
        x_out = objective
    # x_out: loss in this time
    # objective: total loss
    return x_out, objective
