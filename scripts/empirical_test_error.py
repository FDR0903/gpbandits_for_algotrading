import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gpytorch
from tqdm import tqdm
from AOE.gp_bandit import ExactGPModel
import seaborn as sns
import numpy as np
from AOE.gp_utils import Wasserstein_GP_mean
import seaborn as sns
dtype = torch.float
import itertools

save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plot/empirical_exp/")
if not os.path.exists(save_path):
    os.makedirs(save_path)


class GpGenerator:
    def __init__(self, noise=1e-2, lengthscale=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise
        likelihood.noise_covar.raw_noise.requires_grad_(False)
        self.model = ExactGPModel(train_x = torch.zeros((0, 1), dtype=dtype), train_y = torch.zeros(0, dtype=dtype), likelihood = likelihood)
        if lengthscale:
            self.model.covar_module.base_kernel.lengthscale = lengthscale

    def sample_posterior(self, x):
        self.model.eval()
        with torch.no_grad():
            y = self.model(x).rsample()
        self.model.add_point(x, y)
        return y
    
    def posterior_samples(self, n_wass=200, lv=-1 , uv=1):
        model, likelihood = self.model, self.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_wass, dtype=dtype)
            observed_pred = likelihood(model(test_x))
            posterior_mean_1  = observed_pred.mean
            posterior_covar_1 = observed_pred.covariance_matrix
        return posterior_mean_1, posterior_covar_1
    
    def sample_prior(self, x):
        self.model.eval()
        with torch.no_grad():
            y = self.model(x).rsample()
        return y

def generate_x(n):
    x_new = 2*torch.rand(n, dtype=dtype) - 1
    return x_new


def run(size_window, noise, lengthscale = None, L=100, T=100,  N_wasserstein_compute=250):

    ## Hypothesis 0
    result = []
    for _ in tqdm(range(L), desc=f"Hypothesis 0, size_window {size_window} noise {noise}, lengthscale {lengthscale}"):
        g = GpGenerator(noise=noise, lengthscale=lengthscale)
        for _ in range(T):
            dataset = generate_x(size_window)
            y = g.sample_prior(dataset)
            dataset_1 = dataset[(size_window//2):]
            y_1 = y[(size_window//2):]
            dataset_2 = dataset[:(-size_window//2)]
            y_2 = y[:(-size_window//2)]

            g.model.set_train_data(dataset_1, y_1, strict=False)
            posterior_mean_1, posterior_covar_1 = g.posterior_samples(n_wass=N_wasserstein_compute)
            g.model.set_train_data(dataset_2, y_2, strict=False)
            posterior_mean_2, posterior_covar_2 = g.posterior_samples(n_wass=N_wasserstein_compute)
            
            d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), posterior_mean_2.numpy(), posterior_covar_2.numpy())
            result.append(d)
    result = np.array(result)

    ## Hypothesis 1
    result_diff = []
    for _ in tqdm(range(L), desc=f"Hypothesis 1, size_window {size_window} noise {noise}, lengthscale {lengthscale}"):
        g1 = GpGenerator()
        g2 = GpGenerator()
        for _ in range(T):
            dataset = generate_x(size_window)
            dataset_1 = dataset[(size_window//2):]
            dataset_2 = dataset[:(-size_window//2)]

            y = g.sample_prior(dataset)
            y_1 = g1.sample_prior(dataset_1)
            y_2 = g2.sample_prior(dataset_2)

            g1.model.set_train_data(dataset_1, y_1, strict=False)
            posterior_mean_1, posterior_covar_1 = g1.posterior_samples(n_wass=N_wasserstein_compute)
            g2.model.set_train_data(dataset_2, y_2, strict=False)
            posterior_mean_2, posterior_covar_2 = g2.posterior_samples(n_wass=N_wasserstein_compute)
            
            d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), posterior_mean_2.numpy(), posterior_covar_2.numpy())
            result_diff.append(d)
    result_diff = np.array(result_diff)

    dir_path = os.path.join(save_path, f'hypothesis_test_error-size_window_{size_window}-noise_{noise}-lengthscale_{lengthscale}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Save values
    np.save(os.path.join(dir_path, f'hypothesis_0-size_window_{size_window}-noise_{noise}-lengthscale_{lengthscale}'), result)
    np.save(os.path.join(dir_path, f'hypothesis_1-size_window_{size_window}-noise_{noise}-lengthscale_{lengthscale}'), result_diff)

    # Create and saveplot
    sns.distplot(np.log(result), hist=True, kde=True, label="Hypothesis H0")
    sns.distplot(np.log(result_diff), hist=True, kde=True, label="Hypothesis H1")
    plt.xlabel("log was distance")
    plt.legend()
    plt.title("Distribution was distance under H0 and H1")

    plt.savefig(os.path.join(dir_path, f'hypothesis_test_error-size_window_{size_window}-noise_{noise}-lengthscale_{lengthscale}.pdf'))
    plt.savefig(os.path.join(dir_path, f'hypothesis_test_error-size_window_{size_window}-noise_{noise}-lengthscale_{lengthscale}.png'))

    plt.close()


if __name__ == "__main__":
    
    ## Parameters Environment
    size_window = [10, 20, 50]
    noise_param = [1e-4, 1e-2, 1e-1, 0.5]
    lengthscale = [0.1, None, 1., 2.]
    paramlist = list(itertools.product(size_window, noise_param, lengthscale))

    #Parameters Monte Carlo
    L = 80
    T = 80
    N_wasserstein_compute = 150

    for size_w, noise, l in paramlist:
        run(size_window=size_w, noise=noise, lengthscale=l, L=L, T=T,  N_wasserstein_compute=N_wasserstein_compute)
