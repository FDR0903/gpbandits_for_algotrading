import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from copy import deepcopy


class gp_bandit_finance:
    def __init__(self, strategies, bandit_algo='TS', 
                            likelihood = gpytorch.likelihoods.GaussianLikelihood(),
                            train_x = torch.zeros((0, 1), dtype=torch.float64),
                            train_y = torch.zeros(0, dtype=torch.float64),
                            bandit_params=0.1,
                            training_iter=10,
                            verbose = False,
                            size_buffer = 100,
                            size_window = 10,
                            size_subwindow = 5,
                            type_1_error=0.05,
                            threshold = 0.1):

        self.strategies = strategies
        self.training_iter  = training_iter ## Number of epochs hyperparameter retraining
        self.bandit_algo    = bandit_algo
        self.bandit_params  = bandit_params
        self.verbose        = verbose
        self.size_buffer    = size_buffer
        self.size_window    = size_window
        self.size_subwindow = size_subwindow
        self.type_1_error   = type_1_error ### Must be even to divide last points by two
        self.threshold   = threshold

        self.strat_gp_dict = {}
        for strat in self.strategies.keys():
            self.strat_gp_dict[strat] = gp_bandit(likelihood,
                                                    bandit_algo,
                                                    train_x,
                                                    train_y,
                                                    bandit_params = bandit_params,
                                                    training_iter = training_iter,
                                                    size_buffer = self.size_buffer)

    def select_best_strategy(self, features):
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

    def update_data(self, features, strat, reward, retrain_hyperparameters = False):
        "One get reward, update data and retrain gp of the corresponding strat"

        # Get model and data
        model = self.strat_gp_dict[strat]
        
        #select feature
        
        x_new = features[self.strategies[strat]['contextual_params']['feature_name']]
        y_new = reward

        #Add point to model
        model.update_data(x_new, y_new)

        if retrain_hyperparameters:
            model.train()

    
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

    def plot_strategies(self, strats = "all", lv=-1, uv=1, n_test=100, xlabel=None, plot_path=None, W=9):
        """Plot the fit of all strategies for sanity check"""
        
        if strats == "all":
            nb_strategies = len(self.strategies.keys())
            strats = self.strategies.keys()
        else: nb_strategies = len(strats)

        #rescale_plot(W=W)
        f, axs = plt.subplots(1, nb_strategies, figsize=(7*nb_strategies,7))
        axs = np.array([axs])
        for ((_, strat), ax) in zip(enumerate(strats), np.ndarray.flatten(axs)):
            gp = self.strat_gp_dict[strat]
            ax = gp.build_plot(ax, lv=lv, uv=uv, n_test=100, xlabel=None)
            ax.set_title(strat)
            # Get model and predictions
            #ax.tick_params(axis='x', rotation=90)

        #plt.tight_layout()
        
        if plot_path is not None:
            plt.savefig(plot_path, dpi=150)

        plt.show()
    
    def posterior_S2(self, strat, lv=-1, uv=1, n_test=100):
        """Return mean and confidence from posterior ever S1

        Args:
            strat (string): strategy

        Return:
            trainx, targety, mean, confidence
        """

        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        if train_y.shape[0] < self.size_window:
            return False

        train_x_1, train_y_1 = train_x[(-self.size_window):(-self.size_subwindow)], train_y[(-self.size_window):(-self.size_subwindow)]

        gp.change_data(train_x_1, train_y_1)

        model, likelihood = gp.model, gp.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        lower1, upper1 = observed_pred.confidence_region()

        # Put back original dataset
        gp.change_data(train_x, train_y)

        return train_x_1, train_y_1, posterior_mean_1, lower1, upper1
    
    def prior_S2(self, strat, lv=-1, uv=1, n_test=100):
        """Return mean and confidence from prior (hyperparameters are trained)

        Args:
            strat (string): strategy

        Return:
            trainx, targety, mean, confidence
        """

        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        train_x_1, train_y_1 = train_x[(-self.size_subwindow):], train_y[(-self.size_subwindow):]


        likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        model = ExactGPModel(torch.zeros((0, 1), dtype=torch.float64), torch.zeros(0, dtype=torch.float64), likelihood)
        model.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        lower1, upper1 = observed_pred.confidence_region()

        return train_x_1, train_y_1, posterior_mean_1, lower1, upper1
    
    


    def posterior_sliding_window_confidence(self, strat, n_test = 100, lv = -1, uv = 1, training_iter=None, likelihood = None, model = None):
        """return posterior mean and confidence region over window
        - Create a new gp model for the posterior and optimize hyperparameters
        - nothing : reset to zero // parent:copy parent // specified: ok
        """
        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        if train_y.shape[0] < self.size_window:
            return False
        
        train_x_1, train_x_2 = train_x[(-self.size_window):(-self.size_subwindow)], train_x[(-self.size_subwindow):]
        train_y_1, train_y_2 = train_y[(-self.size_window):(-self.size_subwindow)], train_y[(-self.size_subwindow):]

        if likelihood == None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        elif likelihood == "same":
            likelihood = deepcopy(gp.model.likelihood)
        
        if model == None:
            gp_1 = gp_bandit(likelihood,
                            self.bandit_algo,
                            train_x_1,
                            train_y_1,
                            bandit_params = self.bandit_params,
                            training_iter = training_iter)

            gp_2 = gp_bandit(likelihood,
                self.bandit_algo,
                train_x_2,
                train_y_2,
                bandit_params = self.bandit_params,
                training_iter = training_iter)

        elif model == "same":
            gp_1 = deepcopy(gp)
            gp_1.change_data(train_x_1, train_y_1)
            gp_2 = deepcopy(gp)
            gp_2.change_data(train_x_2, train_y_2)

        
        if training_iter != None:
            gp_1.train()
            gp_2.train()

        model, likelihood = gp_1.model, gp_1.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_1 = observed_pred.mean
        lower1, upper1 = observed_pred.confidence_region()

        model, likelihood = gp_2.model, gp_2.model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(lv, uv, n_test).double()
            observed_pred = likelihood(model(test_x))
        posterior_mean_2 = observed_pred.mean
        lower2, upper2 = observed_pred.confidence_region()

        return train_x_1, train_y_1, posterior_mean_1, lower1, upper1, train_x_2, train_y_2, posterior_mean_2, lower2, upper2

    def posterior_sliding_window_covar(self, strat, n_test = 100, lv = -1, uv = 1, training_iter=None):
        """return posterior mean and confidence region over window
        - Create a new gp model for the posterior and optimize hyperparameters
        """
        gp = self.strat_gp_dict[strat]
        
        train_x = gp.model.train_inputs[0]
        train_y = gp.model.train_targets

        if train_y.shape[0] < self.size_window:
            return False

        train_x_1 = train_x[(-self.size_window//2):]
        train_y_1 = train_y[(-self.size_window//2):]
        train_x_2 = train_x[(-self.size_window):(- self.size_window//2)]
        train_y_2 = train_y[(-self.size_window):(- self.size_window//2)]

        if training_iter==None:
            training_iter = self.training_iter

        ### Posterior mean and covariance for each dataset
        gp_1 = gp_bandit(gpytorch.likelihoods.GaussianLikelihood(),
                        self.bandit_algo,
                        train_x_1,
                        train_y_1,
                        bandit_params = self.bandit_params,
                        training_iter = self.training_iter,
                        verbose = self.verbose)

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
                        training_iter = self.training_iter,
                        verbose = self.verbose)

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

    def change_point(self, strat, n_test = 100, lv = -1, uv = 1, training_iter=None):
        """Test for change point detection according to algorithm, compute all quantities

        Args:
            strat (str): strategy to test

        Return:
            bool: whether chage point is detected
        """
        
        # Compute corresponding type 2 error
        # Compute Likelihood ratio test
        # If ratio test big enough reinitialize
        gp = self.strat_gp_dict[strat]
        train_x, train_y = gp.model.train_inputs[0], gp.model.train_targets
        if gp.model.train_inputs[0].shape[0] < self.size_window:
            return False
        
        #train_x, test_x = gp.model.train_inputs[0][:self.p], gp.model.train_inputs[0][self.p:self.P]
        S_1, S_2 = train_x[(-self.size_window):(-self.size_subwindow)], train_x[(-self.size_subwindow):]
        y_1, y_2 = train_y[(-self.size_window):(-self.size_subwindow)], train_y[(-self.size_subwindow):]

        # Compute Ktilde Mutilde from covariance eval model(), K** from the prior model in train with changed train dataset
        gp.change_data(S_1, y_1)
        likelihood, model = gp.model.likelihood, gp.model
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_tilde = model(S_2)
            observed_pred = likelihood(latent_f_tilde)
        mu_tilde, K_tilde = observed_pred.mean, observed_pred.covariance_matrix
        v_h0 = K_tilde + (likelihood.noise**2)*torch.eye(self.size_subwindow)


        # Compute K** mu** from covariance eval model(), K** from the prior model in train with changed train dataset
        gp.change_data(train_x, train_y)

        likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        model = ExactGPModel(torch.zeros((0, 1), dtype=torch.float64), torch.zeros(0, dtype=torch.float64), likelihood)
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_f_star = model(S_2)
            observed_pred = likelihood(latent_f_star)
        # Compute Ktilde Mutilde from covariance eval model(), K** from the prior model in train with changed train dataset
        mu, K = observed_pred.mean, observed_pred.covariance_matrix
        v_h1 = K + (likelihood.noise**2)*torch.eye(self.size_subwindow)


        # Compute delta Vh0 Vh1
        v_h0_inv = torch.inverse(v_h0)
        v_h1_inv = torch.inverse(v_h1)
        delta = v_h1_inv - v_h0_inv
        target_bis = y_2 - mu_tilde

        # Compute test R:
        #R = -y_2 @ v_h1_inv @ y_2 - torch.log(torch.det(v_h1)) + target_bis @ v_h0_inv @ target_bis + torch.log(torch.det(v_h0))
        R = -y_2 @ v_h1_inv @ y_2 + target_bis @ v_h0_inv @ target_bis

        # Conpute Threshold based on type 1 error
        aux = v_h0 @ delta
        #mu_h0 = -torch.trace(aux) - mu_tilde @ (v_h0_inv + v_h1_inv) @ mu_tilde
        mu_h0 = v_h0.shape[0] - mu_tilde @ v_h1_inv @ mu_tilde - torch.trace( v_h1_inv @ v_h0)
        
        sum_lambda_0 = torch.trace(aux @ aux)
        largest_eih0 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (self.size_subwindow - 1, self.size_subwindow - 1))
        smallest_eih0 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (0, 0))
        largest_eih0 = max(np.absolute(largest_eih0), np.absolute(smallest_eih0))

        tau_I = mu_h0 + max(torch.sqrt(-8*np.log(self.type_1_error)*(sum_lambda_0 + mu_tilde @ v_h1_inv @ v_h0 @ v_h1_inv @ mu_tilde)), torch.tensor(-8*np.log(self.type_1_error)*largest_eih0))

        #In case significant test compute equivalent second threshold
        aux = v_h1 @ delta
        #mu_h1 = -torch.trace(aux)
        mu_h1 = -v_h1.shape[0] + mu_tilde @ v_h0_inv @ mu_tilde + torch.trace( v_h0_inv @ v_h1)
        sum_lambda_1 = torch.trace(aux @ aux)
        largest_eih1 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (self.size_subwindow - 1, self.size_subwindow - 1))
        smallest_eih1 = eigh(aux.detach().numpy(), eigvals_only=True, eigvals = (0, 0))
        largest_eih1 = max(np.absolute(largest_eih1), np.absolute(smallest_eih1))
        error_II = torch.max(torch.exp(-((mu_h1 - tau_I)**2)/(8*(sum_lambda_1 + mu_tilde @ v_h0_inv @ v_h1 @ v_h0_inv @ mu_tilde))), torch.exp(-((mu_h1 - tau_I)/(8*torch.tensor(largest_eih1))))) # = lambda 2


        # If test inferior, we already know there is no change
        if R <= tau_I:
            return False, R, tau_I, error_II

        #elif error_II < self.threshold:
            #return True, R, tau_I, error_II
        
        else:
            return True, R, tau_I, error_II


    def reinitialize_arm(self, strat=None, verbose=True):
        if strat != None:
            self.strat_gp_dict[strat] = gp_bandit(likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                                                    bandit_algo = self.bandit_algo,
                                                    train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                    train_y = torch.zeros(0, dtype=torch.float64),
                                                    bandit_params = self.bandit_params,
                                                    training_iter = self.training_iter)
            if verbose:
                print("Regime change detected for strategy " + strat + ", corresponding arm reinitialized")
                
        else:
            for strat in self.strat_gp_dict.keys():
                self.strat_gp_dict[strat] = gp_bandit(likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                                                        bandit_algo = self.bandit_algo,
                                                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                                                        train_y = torch.zeros(0, dtype=torch.float64),
                                                        bandit_params = self.bandit_params,
                                                        training_iter = self.training_iter)
            if verbose:
                print("Regime change detected every arms are reinitialized")

class gp_bandit:
    def __init__(self, likelihood,
                        bandit_algo,
                        train_x = torch.zeros((0, 1), dtype=torch.float64),
                        train_y = torch.zeros(0, dtype=torch.float64),
                        bandit_params=0.1,
                        training_iter=10,
                        verbose = False,
                        size_buffer = None):
        self.model = ExactGPModel(train_x, train_y, likelihood, size_buffer=size_buffer)
        
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

    def update_data_nst(self,  x_new, y_new):
        "One get reward, update data and retrain gp of the corresponding strat"

        #Add point to model
        self.model.add_point(x_new, y_new)
    
    def change_data(self, x_train, y_train):
        """Change data"""
        self.model.set_train_data(x_train, y_train, strict=False)
    
    def compute_ucb(self, test_x):
        lamb = self.bandit_params 
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

    def build_plot(self, ax, lv=-1, uv=1, n_test=100, xlabel=None):
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
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        #ax.yaxis.set_major_formatter('{x:,.2f}\%')    
        
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
        result = self.posterior_sliding_window(strat, n_test, lv, uv)
        if result != False:
            posterior_mean_1, posterior_covar_1, posterior_mean_2, posterior_covar_2 = result
        else:
            return 0, False
        
        ### Compute wasserstein distance between gp:
        print("What am I doing here")
        d = Wasserstein_GP_mean(posterior_mean_1.numpy(), posterior_covar_1.numpy(), posterior_mean_2.numpy(), posterior_covar_2.numpy())
        return d, (d > self.b)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale=None, size_buffer=None):
#         likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ZeroMean()
        self.size_buffer = size_buffer
        if lengthscale==None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel().initialize(lengthscale=lengthscale))
        #self.covar_module = gpytorch.kernels.RBFKernel()
        
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
        if (self.size_buffer == None) or (y_train.size()[0] <= self.size_buffer):
            self.set_train_data(x_train, y_train, strict=False)
        elif y_train.size()[0] > self.size_buffer:
            x_new = x_train[1:]
            y_new = y_train[1:]
            self.set_train_data(x_new, y_new, strict=False)
            

def Wasserstein_GP(mean1, cov1, mean2, cov2):

    mu_0 = mean1
    K_0 = cov1

    mu_1 = mean2
    K_1 = cov2

    sqrtK_0 = scipy.linalg.sqrtm(K_0)
    first_term = np.dot(sqrtK_0, K_1)
    K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

    cov_dist = np.trace(K_0) + np.trace(K_1) - 2 * np.trace(scipy.linalg.sqrtm(K_0_K_1_K_0))
    l2norm = (np.sum(np.square(abs(mu_0 - mu_1))))
    d = np.real(np.sqrt(l2norm + cov_dist))

    return d

def Wasserstein_GP_mean(mean1, cov1, mean2, cov2):
    n = mean1.shape[0]

    mu_0 = mean1
    K_0 = cov1

    mu_1 = mean2
    K_1 = cov2

    sqrtK_0 = scipy.linalg.sqrtm(K_0)
    first_term = np.dot(sqrtK_0, K_1)
    K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

    cov_dist = np.trace(K_0) + np.trace(K_1) - 2 * np.trace(scipy.linalg.sqrtm(K_0_K_1_K_0))
    l2norm = (np.mean(np.square(abs(mu_0 - mu_1))))
    d = np.real(l2norm + cov_dist/n)

    return d

if __name__ == "__main__":
    print("hello")

    ### Do a test
    strategies  = {'arm1' : {'contextual_params': {'feature_name'  : 'feature'}},
               'arm2' : {'contextual_params': {'feature_name'  : 'feature'}}}

    # Parameters bandit
    size_buffer   = 100
    size_window   = 16
    size_subwindow   = 8
    training_iter = 30
    bandit_params = 0.1
    type_1_error = 0.05
    likelihood    = gpytorch.likelihoods.GaussianLikelihood()
    bandit_algo   = 'TS'
    threshold = 2*type_1_error
    bandit = gp_bandit_finance(strategies, bandit_algo=bandit_algo, bandit_params=bandit_params, training_iter=training_iter, size_buffer=size_buffer, size_window=size_window, size_subwindow=size_subwindow, type_1_error=type_1_error, threshold = threshold)

    function = lambda x: np.sin(2*np.pi*x)

    x_train = np.linspace(0, 1, 30)
    y_train = function(x_train)
    x_train = x_train.reshape(-1,1)
    y_train, x_train = torch.tensor(y_train), torch.tensor(x_train)

    bandit.strat_gp_dict["arm1"].change_data(x_train, y_train)
    change_point = bandit.change_point_all("arm1")

