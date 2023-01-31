import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm
from AOE.gp_bandit import ExactGPModel
from AOE.gp_utils import Plot_animation
from AOE.gp_bandit_finance import gp_bandit_finance
import itertools
import multiprocessing as mp

class GpGenerator:
    def __init__(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 1e-4
        likelihood.noise_covar.raw_noise.requires_grad_(False)
        self.model = ExactGPModel(train_x = torch.zeros((0, 1), dtype=torch.float), train_y = torch.zeros(0, dtype=torch.float), likelihood = likelihood)
        
    def sample_posterior(self, x):
        self.model.eval()
        with torch.no_grad():
            y = self.model(x).rsample()
        self.model.add_point(x, y)
        return y
        
    def sample_prior(self, x):
        self.model.eval()
        with torch.no_grad():
            y = self.model(x).rsample()
        return y
    
def normalize_regret(x):
    return (torch.tanh(x) + 1)/2

def generate_x():
    x_new = 2*torch.rand(1) - 1
    return x_new, pd.Series(data={'feature': x_new}, index=['feature'])
    
def sim(dict):
    # Variable Parameters

    ## Parameters Bandits
    seed          = dict["seed"]
    torch.manual_seed(seed)

    size_buffer   = dict["size_buffer"]
    threshold     = dict["threshold"]

    # Parameters environment
    T = dict["T"]
    N = dict["N"]
    T_N = T//N

    # Parameters that are not changed for now
    strategies  = {'arm1' : {'contextual_params': {'feature_name'  : 'feature'}},
                   'arm2' : {'contextual_params': {'feature_name'  : 'feature'}}}
    bandit_params = 0.1
    #bandit_algo   = 'UCB_WAS'
    bandit_algo   = 'UCB_ADAGA'
    size_window   = 20
    training_iter = 30
    bandit_params = {'size_buffer': 50,
                     'lambda': bandit_params,
                     'size_window': size_window,
                     'threshold': 10,
                     'delta': 0.6}

    #likelihood    = gpytorch.likelihoods.GaussianLikelihood()

    # Bandit initialization
    bandit = gp_bandit_finance(strategies, bandit_algo=bandit_algo, bandit_params=bandit_params, training_iter=training_iter)
    
    # Simulation bandit
    regret_sim = torch.zeros(T)
    anim = Plot_animation(save_path = f"./plot/plot_buffer_{size_buffer}_N_{N}_T_{T}_threshold_{threshold}_seed_{seed}")
    for i in range(N):
        # New underlying reward distribution for both arms
        reward1 = GpGenerator()
        reward2 = GpGenerator()
        for t in tqdm(range(T_N)):
            x_new, df_new = generate_x() # Mimic market dynamic
            best_strategy_bandit  = bandit.select_best_strategy(df_new)
            r1, r2 = reward1.sample_posterior(x_new), reward2.sample_posterior(x_new)
            
            if best_strategy_bandit == "arm1":
                regret_t = normalize_regret(torch.max(r1, r2)) - normalize_regret(r1)
                bandit.update_data(df_new, best_strategy_bandit, r1, retrain_hyperparameters = True)
                
            elif best_strategy_bandit == "arm2":
                regret_t = normalize_regret(torch.max(r1, r2)) - normalize_regret(r2)
                bandit.update_data(df_new, best_strategy_bandit, r2, retrain_hyperparameters = True)
            
            #change_point = bandit.change_point(best_strategy_bandit) 
            #if change_point:
            #    bandit.reinitialize_arm()
            regret_sim[i*T_N + t] = regret_t
            
            #f, ax = bandit.plot_strategies()
            anim.add_frame(bandit)
    #cum_regret = torch.cumsum(regret_sim, dim=0)
    anim.make_animation()
    #plt.plot(range(T), cum_regret)
    #plt.savefig(f"./plot/plot_buffer_{size_buffer}_N_{N}_T_{T}_threshold_{threshold}_seed_{seed}.png")
    return None

if __name__ == '__main__':

    size_buffer_list = [50, 100]
    N_list = [2, 5, 10]
    T_list = [100, 500, 1000, 10000]
    threshold_list = [0.1, 0.5, 0.7, 1.]
    seed_list = [0, 1, 2]

    paramlist = list(itertools.product(size_buffer_list, N_list, T_list, threshold_list, seed_list))
    list_dict = []

    for buffer, N, T, threshold, seed in paramlist:
        list_dict.append({"size_buffer": buffer, "N": N, "T": T, "threshold": threshold, "seed": seed})
    
    """
    N = mp.cpu_count()
    print('Number of parallelisable cores: ', N)

    with mp.Pool(processes = N) as p:
        p.map(sim, list_dict)
    """

    sim(list_dict[0])