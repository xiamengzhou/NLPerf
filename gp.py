import gpytorch as gp
import torch
import pandas as pd
import numpy as np


def get_mean_module(mean_module):
    mean_name = mean_module["name"]
    paras = mean_module["paras"]
    if mean_name == "constant_mean":
        return gp.means.ConstantMean()
    elif mean_name == "constant_mean_grad":
        return gp.means.ConstantMeanGrad()
    elif mean_name == "linear_mean":
        return gp.means.LinearMean(**paras)
    elif mean_name == "zero_mean":
        return gp.means.ZeroMean()
    else:
        print("What ???")


def get_covar_module(covar_module):
    covar_name = covar_module["name"]
    paras = covar_module["paras"]
    dims = paras["dims"]
    if covar_name == "rbf":
        return gp.kernels.ScaleKernel(gp.kernels.RBFKernel(ard_num_dims=dims),
                                      num_dims=dims)
    elif covar_name == "rbfgrad":
        return gp.kernels.ScaleKernel(gp.kernels.RBFKernelGrad(ard_num_dims=dims),
                                      num_dims=dims)
    elif covar_name == "cosine":
        return gp.kernels.ScaleKernel(gp.kernels.CosineKernel(ard_num_dims=dims),
                                      num_dims=dims)
    elif covar_name == "matern":
        return gp.kernels.ScaleKernel(gp.kernels.MaternKernel(ard_num_dims=dims),
                                      num_dims=dims)
    elif covar_name == "periodic":
        return gp.kernels.ScaleKernel(gp.kernels.PeriodicKernel(ard_num_dims=dims),
                                      num_dims=dims)
    elif covar_name == "com1":
        print("here")
        kernel = None
        for i in range(dims):
            if i == 0:
                kernel = gp.kernels.ScaleKernel(gp.kernels.RBFKernel(active_dims=[i]), num_dims=1)
            else:
                kernel += gp.kernels.ScaleKernel(gp.kernels.RBFKernel(active_dims=[i]), num_dims=1)
        return kernel
    else:
        print("What ???")


class ExactGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        covar_module["paras"]["dims"] = train_x.shape[-1]
        self.mean_module = get_mean_module(mean_module)
        self.covar_module = get_covar_module(covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def run_gp_train(feats, labels, mean_module, covar_module, verbose=True):
    feats = tensorize_module(feats)
    labels = tensorize_module(labels)

    likelihood = gp.likelihoods.GaussianLikelihood()
    model = ExactGPModel(feats, labels, likelihood, mean_module, covar_module)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(feats)
        # Calc loss and backprop gradients
        loss = -mll(output, labels)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                0,
                # model.covar_module.base_kernel.lengthscale.mean(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    return model, likelihood


def run_gp_test(reg, test_feats, get_ci=True):
    test_feats = tensorize_module(test_feats)
    model, likelihood = reg
    model.eval()
    likelihood.eval()
    observed_pred = likelihood(model(test_feats))
    lower = None; upper = None; rmse = None
    if get_ci:
        lower, upper = observed_pred.confidence_region()
        lower = lower.detach().numpy(); upper = upper.detach().numpy()
    preds = observed_pred.mean.detach().numpy()
    return preds, lower, upper


def tensorize_module(feats):
    ttt = type(feats)
    tensor = None
    if ttt == pd.DataFrame:
        tensor = torch.Tensor(feats.values)
    elif ttt == np.ndarray:
        tensor = torch.Tensor(feats)
    else:
        print("The type of the coming data is {}, which is not supported ...".format(ttt))
    print("The shape of the tensor is {} ..".format(tensor.shape))
    return tensor

