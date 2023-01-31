from email.policy import strict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
import torch
from .plots import rescale_plot
from .gp_utils import Wasserstein_GP

class gp_bandit:
    def __init__(self, likelihood,
                        bandit_algo,
                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                        train_y = torch.zeros(0, dtype=torch.float64),
                        bandit_params=0.1,
                        training_iter=10,
                        verbose = False):
        self.model = ExactGPModel(train_x, train_y, likelihood)
        self.training_iter = training_iter ## Number of epochs hyperparameter retraining
        self.bandit_algo   = bandit_algo
        self.bandit_params = bandit_params
        self.verbose       = verbose
    
    def train(self, training_iter=None):

        #Get model and data
        model, likelihood = self.model, self.model.likelihood
        model.train()
        likelihood.train()

        x_train = model.train_inputs[0]
        y_train = model.train_targets

        if training_iter == None:
            training_iter = self.training_iter

        # Find optimal model hyperparameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters            
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_train)
            # Calc loss and backprop gradients

            loss = -mll(output, y_train)
            loss.backward()
            
            if self.verbose:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, self.training_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))
            
            optimizer.step()
        
    def update_data(self, x_new, y_new):
        
        #Add point to model
        self.model.add_point(x_new, y_new)

    def update_data_nst(self,  x_new, y_new, size_buffer):
        "One get reward, update data and retrain gp of the corresponding strat"

        #Add point to model
        self.model.add_point_nst(x_new, y_new, size_buffer)
    
    def change_data(self, x_train, y_train):
        """Change data"""
        self.model.set_train_data(x_train, y_train, strict=False)
    
    def compute_ucb(self, test_x):
        lamb = self.bandit_params['lambda']
        "Compute ucb for one strat"
        model, likelihood = self.model, self.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.verbose: print('test_x:', test_x)

            f_pred = likelihood(model(test_x))
            if self.verbose: print('f_pred:', f_pred)

        return f_pred.mean + lamb*torch.sqrt(f_pred.variance)
    
    def compute_ts(self, test_x):
        "Compute thompson sampling for one strat"
        model, likelihood = self.model, self.model.likelihood
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.verbose: print('test_x:', test_x)
            observed_pred = likelihood(model(test_x))
            if self.verbose: print('observed_pred:', observed_pred)

            with gpytorch.settings.fast_computations(covar_root_decomposition=False):
                the_draw = observed_pred.rsample()
                if self.verbose: print('Thompson sampling draw:', the_draw)

        return the_draw

    def build_plot(self, ax, lv=None, uv=None, n_test=100, xlabel=None):
        """Build matplotlib graph for one strat

        Args:
            ax (matplotlib.axe): graph to plot on
            lv (int, optional): lower x value. Defaults to -1.
            uv (int, optional): upper x value. Defaults to 1.
            n_test (int, optional): number of test points. Defaults to 100.
            xlabel (String, optional): label for x axis. Defaults to None.

        Returns:
            matplotlib.axe: graph plotted
        """
        
        # Get model and predictions
        model, likelihood = self.model, self.model.likelihood
        model.eval()
        likelihood.eval()
        
        if lv is None:
            lv = model.train_inputs[0].min().item()
            uv = model.train_inputs[0].max().item()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))

        train_x = model.train_inputs[0]
        train_y = model.train_targets

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'k', linewidth=3)
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, 
                          facecolor='silver', hatch="ooo", edgecolor="gray")



        ax.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)

        
        if xlabel:
            ax.set_xlabel(xlabel)
        return ax
    
    def posterior_sliding_window_plot(self, strat, n_test = 100, lv = -1, uv = 1):
        """return posterior mean and confidence region over window
        """
        model = self.strat_model_dict[strat]
        
        train_x = model.train_inputs[0]
        train_y = model.train_targets

        if train_y.shape[0] < self.size_window:
            return False

        train_x_1 = train_x[(-self.size_window//2):]
        train_y_1 = train_y[(-self.size_window//2):]
        train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
        train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]
        
        ### Posterior mean and covariance for each dataset
        model.set_train_data(train_x_1, train_y_1, strict=False)

        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        lower1, upper1 = observed_pred.confidence_region()
        
        model.set_train_data(train_x_2, train_y_2, strict=False)

        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_2 = observed_pred.mean
        lower2, upper2 = observed_pred.confidence_region()
        
        model.set_train_data(train_x, train_y, strict=False)

        return posterior_mean_1, lower1, upper1, posterior_mean_2, lower2, upper2

    def posterior_sliding_window(self, strat, n_test = 100, lv = -1, uv = 1):
        """return posterior mean and covariance over window
        """
        model = self.strat_model_dict[strat]
        
        train_x = model.train_inputs[0]
        train_y = model.train_targets

        # Not enough datapoints
        if train_y.shape[0] < self.size_window:
            return False

        train_x_1 = train_x[(-self.size_window//2):]
        train_y_1 = train_y[(-self.size_window//2):]
        train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
        train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]
        
        ### Posterior mean and covariance for each dataset
        model.set_train_data(train_x_1, train_y_1, strict=False)

        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        posterior_covar_1 = observed_pred.covariance_matrix

        
        model.set_train_data(train_x_2, train_y_2, strict=False)

        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_2 = observed_pred.mean
        posterior_covar_2 = observed_pred.covariance_matrix

        model.set_train_data(train_x, train_y, strict=False)

        return posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2


    def change_point(self, strat, n_test = 100, lv = -1, uv = 1):
        """Test for change point detection

        Args:
            strat (str): strategy to test

        Return:
            bool: whether chage point is detected
        """        
        posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2 = self.posterior_sliding_window(strat, n_test, lv, uv)

        ### Compute wasserstein distance between gp:
        #distance_mean = (posterior_mean_1 - posterior_mean_2).pow(2).sum()
        d = Wasserstein_GP(posterior_mean_1.numpy(), posterior_covar_1.numpy(), posterior_mean_2.numpy(), posterior_covar_2.numpy())
        return (d > self.b)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
#         likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        self.reward_observation_times = []
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def add_point(self, x_new, y_new):
        """Implement a finite replay buffer for non stationarity
        If limit buffer reached, delete oldest point
        """
        
        x_train = self.train_inputs[0]
        y_train = self.train_targets
        x_train = torch.cat((x_train, torch.tensor([x_new]).reshape(1,1)))
        y_train = torch.cat((y_train, torch.tensor([y_new])))
            
        self.set_train_data(x_train, y_train, strict=False)

    def add_point_nst(self, x_new, y_new, size_buffer):
        """Implement a finite replay buffer for non stationarity
        If limit buffer reached, delete oldest point
        """
        
        x_train = self.train_inputs[0]
        y_train = self.train_targets
        x_train = torch.cat((x_train, torch.tensor([x_new]).reshape(1,1)))
        y_train = torch.cat((y_train, torch.tensor([y_new])))

        if x_train.shape[0] > size_buffer:
            x_train = x_train[1:-1]
            y_train = y_train[1:-1]

        self.set_train_data(x_train, y_train, strict=False)

