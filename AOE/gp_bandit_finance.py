
from email.policy import strict
import math
import os
from this import d
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from .plots import rescale_plot
from .gp_bandit import gp_bandit
from .gp_utils import Wasserstein_GP_mean, Wasserstein_GP

class gp_bandit_finance:
    def __init__(self, strategies, bandit_algo='TS', 
                            likelihood = gpytorch.likelihoods.GaussianLikelihood(),
                            train_x = torch.zeros((0, 1), dtype=torch.float64),
                            train_y = torch.zeros(0, dtype=torch.float64),
                            bandit_params=0.1,
                            training_iter=10,
                            verbose = False,
                            threshold=10):

        self.strategies = strategies
        self.training_iter = training_iter ## Number of epochs hyperparameter retraining
        self.bandit_algo   = bandit_algo
        self.bandit_params = bandit_params
        self.verbose       =  verbose
        self.likelihood    = likelihood

        if bandit_algo == 'TS_NS':
            self.size_buffer   = self.bandit_params['size_buffer']
        elif bandit_algo == 'UCB_NS':
            self.size_buffer   = self.bandit_params['size_buffer']
            self.lamb          = self.bandit_params['lambda']
        elif bandit_algo == 'TS_WAS':
            self.size_window   = self.bandit_params['size_window']
            self.b             = self.bandit_params['threshold'] ### Must be even to divide last points by two
            self.size_buffer   = self.bandit_params['size_buffer']
        elif bandit_algo == 'UCB_WAS':
            self.size_window   = self.bandit_params['size_window']
            self.b             = self.bandit_params['threshold'] ### Must be even to divide last points by two
            self.lamb          = self.bandit_params['lambda']
            self.size_buffer   = self.bandit_params['size_buffer']
        
        self.strat_gp_dict = {}
        for strat in self.strategies.keys():
            self.strat_gp_dict[strat] = gp_bandit(likelihood,
                                                    bandit_algo,
                                                    train_x,
                                                    train_y,
                                                    bandit_params = bandit_params,
                                                    training_iter = training_iter)

    def select_best_strategy(self, features):
        if self.bandit_algo == 'UCB_NS':
            "Compute ucb for each gp and select best"
            best_strat, best_ucb = "", -np.inf
            for strat in self.strategies.keys():
                ucb_strat = self.compute_ucb(strat, features)
                if ucb_strat > best_ucb:
                    best_strat, best_ucb = strat, ucb_strat

        elif self.bandit_algo == 'TS_NS':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ts(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat

        elif self.bandit_algo == 'UCB':
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

        elif self.bandit_algo == 'TS_WAS':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ts_was(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat

        elif self.bandit_algo == 'UCB_WAS':
            "Compute ts for each gp and select best"
            best_strat, best_ts = "", -np.inf
            for strat in self.strategies.keys():
                ts_strat = self.compute_ucb_was(strat, features)
                if ts_strat > best_ts:
                    best_strat, best_ts = strat, ts_strat
        else:
            best_strat = 'NOT IMPLEMENTED'
            
        return best_strat

    def update_data(self, features, strat, reward, retrain_hyperparameters = False):
        "One get reward, update data and retrain gp of the corresponding strat"

        # Get model and data
        model = self.strat_gp_dict[strat]
        
        #select feature
        x_new = features[self.strategies[strat]['contextual_params']['feature_name']]
        y_new = reward

        #Add point to model
        if self.bandit_algo == 'UCB_NS':
            model.update_data_nst(x_new, y_new, self.size_buffer)

        elif self.bandit_algo == 'TS_NS':
            model.update_data_nst(x_new, y_new, self.size_buffer)

        elif self.bandit_algo == 'UCB':
            model.update_data(x_new, y_new)

        elif self.bandit_algo == 'TS':
            model.update_data(x_new, y_new)

        elif self.bandit_algo == 'TS_WAS':
            model.update_data_nst(x_new, y_new, self.size_buffer)
#            model.update_data(x_new, y_new)

        elif self.bandit_algo == 'UCB_WAS':
            model.update_data_nst(x_new, y_new, self.size_buffer)
            # model.update_data(x_new, y_new)

        else:
            best_strat = 'NOT IMPLEMENTED'
        

        if retrain_hyperparameters:
            model.train()

#    def update_data_nst(self, features, strat, reward, size_buffer, retrain_hyperparameters = False):
#        "One get reward, update data and retrain gp of the corresponding strat"#
#
        # Get model and data
#        model = self.strat_gp_dict[strat]
        
        #select feature
#        x_new = features[self.strategies[strat]['contextual_params']['feature_name']]
#        y_new = reward

        #Add point to model
#        model.update_data_nst(x_new, y_new, size_buffer)

#        if retrain_hyperparameters:
#            model.train()
    
    def compute_ucb(self, strat, features):
        "Compute ucb for one strat"
        if self.verbose: print('Computing UCB for strategy:', strat)

        gp = self.strat_gp_dict[strat]

        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ucb(t_x)
    
    def compute_ts(self, strat, features):
        "Compute thompson sampling for one strat"
        if self.verbose: print('Computing thompson sampling for strategy:', strat)
        
        gp = self.strat_gp_dict[strat]

        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ts(t_x)

    def compute_ts_was(self, strat, features):
        "Compute thompson sampling for one strat with was check first"
        if self.verbose: print('Computing thompson sampling waserstein for strategy:', strat)   

        # First Compute the waser distancde
        #print('testing train target is', self.strat_gp_dict[strat].model.train_targets.shape[0], ' and window is ', self.size_window)
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets
            
            lv = train_x.min().item()
            uv = train_x.max().item()

            if self.change_point(strat, lv = lv, uv = uv):
                # update the gp
                self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x       = train_x[(-self.size_window):(- self.size_window//2)],
                                                        train_y       = train_y[(-self.size_window):(- self.size_window//2)],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()
        
        return gp.compute_ts(t_x)

    def compute_ucb_was(self, strat, features):
        "Compute thompson sampling for one strat with was check first"
        if self.verbose: print('Computing thompson sampling waserstein for strategy:', strat)   

        # First Compute the waser distancde
        if self.strat_gp_dict[strat].model.train_targets.shape[0] > self.size_window:
            gp = self.strat_gp_dict[strat]
        
            train_x = gp.model.train_inputs[0]
            train_y = gp.model.train_targets
            
            lv = train_x.min().item()
            uv = train_x.max().item()

            if self.change_point(strat, lv = lv, uv = uv):
                # print('REGIME CHANGE UCB WAS !!!!!! for strategy:', strat)
                
                # posterior_mean_1, lower1, upper1, \
                #     posterior_mean_2, lower2, upper2 = \
                #         self.posterior_sliding_window_confidence(strat, n_test = 100, lv=lv, uv=uv)

                # posterior_mean_1, posterior_covar_1, \
                #     posterior_mean_2, posterior_covar_2 = \
                #         self.posterior_sliding_window_covar(strat, 100, lv, uv)

                # ### Compute wasserstein distance between gp:
                # d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), 
                #             posterior_mean_2.numpy(), posterior_covar_2.numpy())

                
                # # Plot
                # fig, ax = plt.subplots(1, 1)
                # test_x = torch.linspace(lv, uv, 100).double()
                # ax.plot(test_x.numpy(), posterior_mean_1.numpy(), 'b')
                # ax.plot(test_x.numpy(), posterior_mean_2.numpy(), 'r')
                # # Shade between the lower and upper confidence bounds
                # ax.fill_between(test_x.numpy(), lower1.detach().numpy(), upper1.detach().numpy(), alpha=0.5)
                # ax.fill_between(test_x.numpy(), lower2.detach().numpy(), upper2.detach().numpy(), alpha=0.5)
                # # legend / title
                # ax.legend(['mean 1', 'mean 2'])
                # ax.set_title(f'Bandit UCB WAS \n strat {strat} \n Was dist= {round(d, 3)} ')
                # plt.show()


                self.strat_gp_dict[strat] = gp_bandit(self.likelihood,
                                                        self.bandit_algo,
                                                        train_x       = train_x[(-self.size_window):(- self.size_window//2)],
                                                        train_y       = train_y[(-self.size_window):(- self.size_window//2)],
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)

        gp = self.strat_gp_dict[strat]
        t_x = torch.tensor([features[self.strategies[strat]['contextual_params']['feature_name']]]).double()

        return gp.compute_ucb(t_x)

    def plot_strategies(self, strats = "all", lv=None, uv=None, n_test=100, xlabel=None, plot_path=None, W=9):
        """Plot the fit of all strategies for sanity check"""
        
        if strats == "all":
            nb_strategies = len(self.strategies.keys())
            strats = self.strategies.keys()
        else: nb_strategies = len(strats)

        #rescale_plot(W=W)
        f, axs = plt.subplots(1, nb_strategies, figsize=(4*nb_strategies,3), sharey=True)
        axs = np.array([axs])
        for ((index, strat), ax) in zip(enumerate(strats), np.ndarray.flatten(axs)):
            gp = self.strat_gp_dict[strat]
            ax = gp.build_plot(ax, lv=lv, uv=uv, n_test=100, xlabel=None)
            ax.set_title(strat)
            # Get model and predictions
            #ax.tick_params(axis='x', rotation=90)

        #plt.tight_layout()
        
        if plot_path is not None:
            plt.savefig(plot_path, dpi=150)

        plt.show()

    def posterior_sliding_window_confidence(self, strat, n_test = 100, lv = -1, uv = 1, training_iter=50):
        """return posterior mean and confidence region over window
        - Create a new gp model for the posterior and optimize hyperparameters
        """
        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        try:
            if train_y.shape[0] < self.size_window:
                return None, None, None, None, None, None
        except Exception as e:
            self.size_window = train_y.shape[0]
            if train_y.shape[0] < self.size_window:
                return None, None, None, None, None, None

        train_x_1 = train_x[(-self.size_window//2):]
        train_y_1 = train_y[(-self.size_window//2):]
        train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
        train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]
        
        ### Posterior mean and covariance for each dataset
        gp_1 = gp_bandit(gpytorch.likelihoods.GaussianLikelihood(),
                        self.bandit_algo,
                        train_x_1,
                        train_y_1,
                        bandit_params = self.bandit_params,
                        training_iter = training_iter)

        gp_1.train()

        model, likelihood = gp_1.model, gp_1.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        lower1, upper1 = observed_pred.confidence_region()

        ### Second posterior distribution
        gp_2 = gp_bandit(gpytorch.likelihoods.GaussianLikelihood(),
                        self.bandit_algo,
                        train_x_2,
                        train_y_2,
                        bandit_params = self.bandit_params,
                        training_iter = training_iter)

        gp_2.train()

        model, likelihood = gp_2.model, gp_2.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_2 = observed_pred.mean
        lower2, upper2 = observed_pred.confidence_region()

        return posterior_mean_1, lower1, upper1, posterior_mean_2, lower2, upper2

    def posterior_sliding_window_covar(self, strat, n_test = 100, lv = -1, uv = 1):
        """return posterior mean and confidence region over window
        - Create a new gp model for the posterior and optimize hyperparameters
        """
        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        try:
            if train_y.shape[0] < self.size_window:
                return None, None, None, None, None, None
        except Exception as e:
            self.size_window = train_y.shape[0]
            if train_y.shape[0] < self.size_window:
                return None, None, None, None, None, None

        train_x_1 = train_x[(-self.size_window//2):]
        train_y_1 = train_y[(-self.size_window//2):]
        train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
        train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]
        
        ### Posterior mean and covariance for each dataset
        gp_1 = gp_bandit(gpytorch.likelihoods.GaussianLikelihood(),
                        self.bandit_algo,
                        train_x_1,
                        train_y_1,
                        bandit_params = self.bandit_params,
                        training_iter = self.training_iter)

        gp_1.train()

        model, likelihood = gp_1.model, gp_1.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        posterior_covar_1 = observed_pred.covariance_matrix

        ### Second posterior distribution
        gp_2 = gp_bandit(gpytorch.likelihoods.GaussianLikelihood(),
                        self.bandit_algo,
                        train_x_2,
                        train_y_2,
                        bandit_params = self.bandit_params,
                        training_iter = self.training_iter)

        gp_2.train()

        model, likelihood = gp_2.model, gp_2.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_2 = observed_pred.mean
        posterior_covar_2 = observed_pred.covariance_matrix

        return posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2

    def change_point(self, strat, n_test = 100, lv = -1, uv = 1):
        """Test for change point detection

        Args:
            strat (str): strategy to test

        Return:
            bool: whether chage point is detected
        """        
        posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2 = self.posterior_sliding_window_covar(strat, n_test, lv, uv)

        ### Compute wasserstein distance between gp:
        #distance_mean = (posterior_mean_1 - posterior_mean_2).pow(2).sum()
        d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), posterior_mean_2.numpy(), posterior_covar_2.numpy())
        
        return (d > self.b)