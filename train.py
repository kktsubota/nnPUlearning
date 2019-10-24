import copy
import argparse
import numpy as np
import os
import sys

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from model import LinearClassifier, ThreeLayerPerceptron, MultiLayerPerceptron, CNN
from pu_loss import PULoss
from dataset import load_dataset
from torchnet.meter import AverageValueMeter


def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='non-negative / unbiased PU learning Chainer implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=30000,
                        help='Mini batch size')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['figure1', 'exp-mnist', 'exp-cifar'],
                        help="Preset of configuration\n"
                             "figure1: The setting of Figure1\n"
                             "exp-mnist: The setting of MNIST experiment in Experiment\n"
                             "exp-cifar: The setting of CIFAR10 experiment in Experiment")
    parser.add_argument('--dataset', '-d', default='mnist', type=str, choices=['mnist', 'cifar10'],
                        help='The dataset name')
    parser.add_argument('--labeled', '-l', default=100, type=int,
                        help='# of labeled data')
    parser.add_argument('--unlabeled', '-u', default=59900, type=int,
                        help='# of unlabeled data')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPU')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPU')
    parser.add_argument('--loss', type=str, default="sigmoid", choices=['logistic', 'sigmoid'],
                        help='The name of a loss function')
    parser.add_argument('--model', '-m', default='3lp', choices=['linear', '3lp', 'mlp'],
                        help='The name of a classification model')
    parser.add_argument('--stepsize', '-s', default=1e-3, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args(arguments)
    if args.preset == "figure1":
        args.labeled = 100
        args.unlabeled = 59900
        args.dataset = "mnist"
        args.batchsize = 30000
        args.model = "3lp"
    elif args.preset == "exp-mnist":
        args.labeled = 1000
        args.unlabeled = 60000
        args.dataset = "mnist"
        args.batchsize = 30000
        args.model = "mlp"
    elif args.preset == "exp-cifar":
        args.labeled = 1000
        args.unlabeled = 50000
        args.dataset = "cifar10"
        args.batchsize = 500
        args.model = "cnn"
        args.stepsize = 1e-5
    assert (args.batchsize > 0)
    assert (args.epoch > 0)
    assert (0 < args.labeled < 30000)
    if args.dataset == "mnist":
        assert (0 < args.unlabeled <= 60000)
    else:
        assert (0 < args.unlabeled <= 50000)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
    return args


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x), "sigmoid": lambda x: torch.sigmoid(-x)}
    return losses[loss_name]


def select_model(model_name):
    models = {"linear": LinearClassifier, "3lp": ThreeLayerPerceptron,
              "mlp": MultiLayerPerceptron, "cnn": CNN}
    return models[model_name]


def make_optimizer(model, stepsize):
    optimizer = torch.optim.Adam(model.parameters(), lr=stepsize, weight_decay=5e-3)
    return optimizer


def main(arguments):
    args = process_args(arguments)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    # dataset setup
    XYtrain, XYtest, prior = load_dataset(args.dataset, args.labeled, args.unlabeled)
    dim = XYtrain[0][0].size // len(XYtrain[0][0])
    train_iter = torch.utils.data.DataLoader(XYtrain, args.batchsize, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(XYtrain, args.batchsize, shuffle=False)
    test_iter = torch.utils.data.DataLoader(XYtest, args.batchsize, shuffle=False)

    # model setup
    loss_type = select_loss(args.loss)
    selected_model = select_model(args.model)
    model = selected_model(prior, dim)
    models = {"nnPU": copy.deepcopy(model), "uPU": copy.deepcopy(model)}
    loss_funcs = {"nnPU": PULoss(prior, loss=loss_type, nnpu=True, gamma=args.gamma, beta=args.beta),
                  "uPU": PULoss(prior, loss=loss_type, nnpu=False)}
    if torch.cuda.is_available():
        for m in models.values():
            m.to(device)

    # trainer setup
    optimizers = {k: make_optimizer(v, args.stepsize) for k, v in models.items()}
    print("prior: {}".format(prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
    print("model: {}".format(selected_model))
    print("beta: {}".format(args.beta))
    print("gamma: {}".format(args.gamma))
    print("")

    os.makedirs(args.out)

    writer = SummaryWriter(os.path.join(args.out, 'logdir'))
    # run training
    for epoch in range(args.epoch):
        # train
        for x, t in train_iter:
            x, t = x.to(device), t.to(device)
            for key in optimizers.keys():
                models[key].train()
                loss, loss_total = models[key](x, t, loss_funcs[key])
                models[key].zero_grad()
                loss.backward()
                optimizers[key].step()

        # validation
        # key: [t_p, t_n, f_p, f_n]
        summary = {key: np.zeros(4) for key in models.keys()}
        for x, t in valid_iter:
            x, t = x.to(device), t.to(device)
            for key, model in models.items():
                model.eval()
                summary[key] += model.compute_prediction_summary(x, t)

        computed_summary = {}
        for k, values in summary.items():
            t_p, t_u, f_p, f_u = values
            n_p = t_p + f_u
            n_u = t_u + f_p
            error_p = 1 - float(t_p) / n_p
            error_u = 1 - float(t_u) / n_u
            computed_summary[k] = 2 * prior * error_p + error_u - prior

        # should calculate compute_mean instead of value
        for key, value in computed_summary.items():
            writer.add_scalar('train/{}/error'.format(key), value, epoch)

        # test:
        for x, t in test_iter:
            x, t = x.to(device), t.to(device)
            for key, model in models.items():
                model.eval()
                err = model.error(x, t)
                writer.add_scalar('test/{}/error'.format(key), err, epoch)
    
    torch.save(model.cpu(), os.path.join(args.out, 'model.pt'))



if __name__ == '__main__':
    main(sys.argv[1:])
