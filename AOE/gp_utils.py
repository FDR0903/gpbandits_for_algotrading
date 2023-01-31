import numpy as np
import scipy
import os
import imageio
import matplotlib.pyplot as plt



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

class Plot_animation:
    def __init__(
        self,
        save_path: str,
    ) -> None:
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.frames = []
        self.n_iters = 0

    def add_frame(self, bandit, strats = "all", lv=-1, uv=1, n_test=100, xlabel=None):
        
        if strats == "all":
            nb_strategies = len(bandit.strategies.keys())
            strats = bandit.strategies.keys()
        else: nb_strategies = len(strats)

        f, axs = plt.subplots(1, nb_strategies, figsize=(4*nb_strategies,3), sharey=True)
        axs = np.array([axs])
        for ((_, strat), ax) in zip(enumerate(strats), np.ndarray.flatten(axs)):
            gp = bandit.strat_gp_dict[strat]
            ax = gp.build_plot(ax, lv=lv, uv=uv, n_test = n_test, xlabel=xlabel)
            ax.set_title(strat)
            f.savefig(os.path.join(self.save_path, f"{str(self.n_iters).zfill(4)}.png"))
        plt.clf()
        self.n_iters += 1

    def make_animation(self):
        images = []
        for i in range(self.n_iters):
            filename = os.path.join(self.save_path, f"{str(i).zfill(4)}.png")
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(self.save_path, f"animation.gif"), images)