import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from .plots import rescale_plot

class gp_bandit:
    def __init__(self, strategies, likelihood, size_buffer, bandit_algo, bandit_params=0.1, training_iter=10, verbose=False):
        self.strategies = strategies
        self.strat_model_dict = {}
        for strat in self.strategies.keys():
            self.strat_model_dict[strat] = ExactGPModel(torch.zeros((0, 1), dtype=torch.float64), 
                                                        torch.zeros(0, dtype=torch.float64),
                                                        likelihood,
                                                        size_buffer)
        
        self.training_iter = training_iter ## Number of epochs hyperparameter retraining
        self.bandit_algo   = bandit_algo
        self.bandit_params = bandit_params
        self.verbose       = verbose

    def select_best_strategy(self, features, algo='UCB'):
        if self.bandit_algo == 'UCB':
            "Compute ucb for each gp and select best"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_ucb(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat
        elif self.bandit_algo == 'TS':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ts(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat
        else:
            best_strat = 'NOT IMPLEMENTED'
            
        return best_strat
    
    def retrain_model(self, strat):
        #Get model and data
        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        
    def update_data(self, features, strat, reward, retrain_hyperparameters = False):
        "One get reward, update data and retrain gp of the corresponding strat"

        # Get model and data
        model = self.strat_model_dict[strat]
        
        #select feature
        x_new = features[self.strategies[strat]['contextual_params']['feature_name']]
        y_new = reward

        #Add point to model
        model.add_point(x_new, y_new)

        #Potentially retrain model fit
        if retrain_hyperparameters:
            #Get model and data
            model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
            model.train()
            likelihood.train()

            x_train = model.train_inputs[0]
            y_train = model.train_targets

            # Find optimal model hyperparameters
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters            
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(self.training_iter):
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
                
    def update_data_nst(self, features, strat, reward, reward_time, retrain_hyperparameters = False):
        "One get reward, update data and retrain gp of the corresponding strat"

        # Get model and data
        model = self.strat_model_dict[strat]
        
        #select feature
        x_new = features[self.strategies[strat]['contextual_params']['feature_name']]
        y_new = reward

        #Add point to model
        model.add_point_nst(x_new, y_new, reward_time)

        #Potentially retrain model fit
        if retrain_hyperparameters:
            #Get model and data
            model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
            model.train()
            likelihood.train()

            x_train = model.train_inputs[0]
            y_train = model.train_targets

            # Find optimal model hyperparameters
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters            
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(self.training_iter):
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
    
    def compute_ucb(self, strat, features):
        lamb = self.bandit_params 
        "Compute ucb for one strat"
        if self.verbose: print('Computing UCB for strategy:', strat)
        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
            if self.verbose: print('test_x:', test_x)

            f_pred = likelihood(model(test_x))
            if self.verbose: print('f_pred:', f_pred)

        return f_pred.mean + lamb*torch.sqrt(f_pred.variance)
    
    def compute_ts(self, strat, features):
        "Compute thompson sampling for one strat"
        if self.verbose: print('Computing thompson sampling for strategy:', strat)
        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
            if self.verbose: print('test_x:', test_x)
            observed_pred = likelihood(model(test_x))
            if self.verbose: print('observed_pred:', observed_pred)

            with gpytorch.settings.fast_computations(covar_root_decomposition=False):
                the_draw = observed_pred.rsample()
                if self.verbose: print('Thompson sampling draw:', the_draw)

        return the_draw

    def plot_fit_strat(self, strat, lv=-3, uv=3, xlabel=None, plot_path=None):
        """Plot the fit for sanity check"""
        
        # Get model and predictions
        model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, 100).double()
            observed_pred = likelihood(model(test_x))

        train_x = model.train_inputs[0]
        train_y = model.train_targets

        f, ax = plt.subplots(1, 1)

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        print("test_x:", test_x.numpy())
        print("observed_pred:", observed_pred.mean.numpy())
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Rewards', 'Mean', 'Confidence'])
        ax.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        #ax.yaxis.set_major_formatter('{x:,.2f}\%')
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if plot_path is not None:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

    def plot_fit_all(self, lv=-3, uv=3, plot_path=None, W=9):
        """Plot the fit of all strategies for sanity check"""
        
        nb_strategies = len(self.strategies.keys())

        #rescale_plot(W=W)
        f, axs = plt.subplots(1, nb_strategies, figsize=(20,7))
        
        for ((index, strat), ax) in zip(enumerate(self.strategies.keys()), np.ndarray.flatten(axs)):
            
            # Get model and predictions
            model, likelihood = self.strat_model_dict[strat], self.strat_model_dict[strat].likelihood
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(lv, uv, 100).double()
                observed_pred = likelihood(model(test_x))

            train_x = model.train_inputs[0]
            train_y = model.train_targets

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()            

            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            ax.set_title(strat)
            #ax.tick_params(axis='x', rotation=90)

        #plt.tight_layout()
        
        if plot_path is not None:
            plt.savefig(plot_path, dpi=150)

        plt.show()


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, size_buffer):
#         likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.size_buffer  = size_buffer
        
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

        if x_train.shape[0] > self.size_buffer:
            x_train = x_train[1:-1]
            y_train = y_train[1:-1]

        self.set_train_data(x_train, y_train, strict=False)
    
    # nonstationary with time windows
    def add_point_nst(self, x_new, y_new, reward_time):
        """Implement a finite replay buffer for non stationarity
        If limit buffer reached, delete oldest point
        """
        x_train = self.train_inputs[0]
        y_train = self.train_targets
        x_train = torch.cat((x_train, torch.tensor([x_new]).reshape(1,1)))
        y_train = torch.cat((y_train, torch.tensor([y_new])))
        
        
        self.reward_observation_times += [reward_time]
        
        if (reward_time - self.reward_observation_times[0])/ np.timedelta64(1, 's') > self.size_buffer:
        #if (reward_time - self.reward_observation_times[0]) > self.size_buffer:
            if self.verbose: print('I added a reward observed at', reward_time, 'and deleting reward observed at',self.reward_observation_times[0])
            x_train = x_train[1:-1]
            y_train = y_train[1:-1]
            self.reward_observation_times = self.reward_observation_times[1:-1]

        self.set_train_data(x_train, y_train, strict=False)        