import os
# import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')


from experiment import Experiment
from itertools import product
from scipy.stats import pearsonr
from tqdm import tqdm


data_path = "data"



class Analysis:
    def __init__(self, exp, immediate_pc=False, initialized_pc=False):
        self.exp = exp
        self.find_active_cells()
        if immediate_pc or initialized_pc:
            self.find_place_cells(initialized=initialized_pc)
    
    def find_active_cells(self):
        self.active_per_env = {env: set(pfs.get_active_cells().cpu().numpy())
                               for env, pfs in self.exp.pfs_per_env.items()}
    
    def find_place_cells(self, initialized=False):
        if initialized:
            self.place_cells_per_env = {env: set(pfs.get_place_cells().cpu().numpy())
                                        for env, pfs in self.exp.pfs_per_env.items()}
            return
        
        self.place_cells_per_env = dict()
        for env in self.exp.pfs_per_env.keys():
            self.exp.compile_grid_cells(env)
            self.exp.load_pfs()
            self.place_cells_per_env[env] = set(self.exp.pfs.get_place_cells().cpu().numpy())
    
    def collect_stats(self, train_env=1):
        self.stats = dict()
        self.stats['active'] = self.collect_active_cell_stats(train_env)
        if hasattr(self, 'place_cells_per_env'):
            self.stats['place'] = self.collect_place_cell_stats(train_env)
    
    def collect_active_cell_stats(self, train_env):
        stats = dict()
        act_train = self.active_per_env[train_env]
        for env, active in self.active_per_env.items():
            pfs = self.exp.pfs_per_env[env]
            stats[(env, 'proportion')] = len(active) / pfs.N
            stats[(env, 'coverage')] = torch.sum(pfs.targets.sum(0) > 1e-3).item() / self.exp.res**2
            if train_env != env:
                stats[(env, 'intersection')] = len(act_train.intersection(active))
                stats[(env, 'union')] = len(act_train.union(active))
                stats[(env, 'IoU')] = len(act_train.intersection(active)) / len(act_train.union(active))
                stats[(env, 'turnover')] = self.get_turnover(train_env, env, all_active=True)
        return stats

    def collect_place_cell_stats(self, train_env):
        stats = dict()
        pc_train = self.active_per_env[train_env]
        for env, pc in self.place_cells_per_env.items():
            pfs = self.exp.pfs_per_env[env]
            coverage = pfs.get_coverage()
            stats[(env, 'proportion')] = len(pc) / pfs.N
            stats[(env, 'scales')] = pfs.scales[list(pc)].mean().item()
            stats[(env, 'sizes')] = torch.mean(coverage[list(pc)].sum((-2, -1)) / self.exp.res**2).item()
            stats[(env, 'coverage')] = coverage[list(pc)].any(0).sum().item() / self.exp.res**2
            if train_env != env:
                stats[(env, 'intersection')] = len(pc_train.intersection(pc))
                stats[(env, 'union')] = len(pc_train.union(pc))
                stats[(env, 'IoU')] = len(pc_train.intersection(pc)) / len(pc_train.union(pc))
                stats[(env, 'turnover')] = self.get_turnover(train_env, env)
                stats[(env, 'remapping')] = self.get_remapping(train_env, env)
        return stats
    
    def get_remapping(self, env1, env2):
        pc1, pc2 = self.place_cells_per_env[env1], self.place_cells_per_env[env2]
        pairs = np.asarray(list(product(pc1, pc2)))
        if pairs.size == 0:
            return np.nan
        pairs = pairs[pairs[:,0] != pairs[:,1]]

        with torch.no_grad():
            d1 = self.exp.pfs_per_env[env1].pairwise_distances(pairs).cpu()
            d2 = self.exp.pfs_per_env[env2].pairwise_distances(pairs).cpu()
            return 1 - max(0, pearsonr(d1, d2).statistic)
    
    def get_turnover(self, env1, env2, all_active=False):
        units = self.active_per_env if all_active else self.place_cells_per_env
        u1, u2 = units[env1], units[env2]
        s = 1 - (len(u1) + len(u2)) / 2 / self.exp.pfs.N
        rmsd = lambda x, y: np.sqrt(((x - y)**2).sum())
        
        alpha_0 = np.asarray([s, 0, 1 - s])
        beta = np.asarray([s**2, 2 * s * (1 - s), (1 - s)**2])
        alpha = np.asarray([self.exp.pfs.N - len(u1.union(u2)),
                            len(u1.union(u2) - u1.intersection(u2)),
                            len(u1.intersection(u2))]) / self.exp.pfs.N
        
        return 1 - (rmsd(alpha, beta) / rmsd(alpha_0, beta))
    
    def place_cell_stats(self, train_env=1):
        self.check_and_initialize_stats(train_env)
        
        print(self.exp.name + ':')
        for (env, k), v in self.stats['place'].items():
            print(f"Env '{env}' {k}: {v:.03f}")
        print()
    
    def active_cell_stats(self, train_env=1):
        self.check_and_initialize_stats(train_env)
        
        print(self.exp.name + ':')
        for (env, k), v in self.stats['active'].items():
            print(f"Env '{env}' {k}: {v:.03f}")
        print()
    
    def check_and_initialize_stats(self, train_env):
        if not hasattr(self, 'stats'):
            self.collect_stats(train_env=train_env)
    
    def save_stats(self, path):
        pd.DataFrame(self.stats).to_json(os.path.join(path, "stats.json"))
    
    @staticmethod
    def load_stats(filename):
        df = pd.read_json(filename)
        df.index = pd.MultiIndex.from_tuples(df.index.map(eval))
        return df


class MultiAnalysis:
    def __init__(self, data_path, exp_names, immediate_pc=False):
        self.exps = [Experiment.load_experiment(data_path, name) for name in exp_names]
        self.anls = [Analysis(exp, immediate_pc=immediate_pc) for exp in self.exps]
    
    def print_active_cells(self):
        for anl in self.anls:
            anl.active_cell_stats()
    
    def print_place_cells(self):
        for anl in self.anls:
            anl.place_cell_stats()
        
    def get_df(self):
        return pd.DataFrame({(anl.exp.name, units): values
                             for anl in self.anls
                             for units, values in anl.stats.items()})


def get_cell_data(exps):
    data = dict()
    for e in tqdm(exps):
        exp = Experiment.load_experiment(data_path, e)
        anl = Analysis(exp, immediate_pc=True)

        env1, env2 = exp.pfs_per_env.keys()
        pc1, pc2 = anl.place_cells_per_env[env1], anl.place_cells_per_env[env2]
        ac1, ac2 = anl.active_per_env[env1], anl.active_per_env[env2]

        data[e] = dict(
            env1_pc_prop=len(pc1) / exp.pfs.N,
            env2_pc_prop=len(pc2) / exp.pfs.N,
            env1_ac_prop=len(ac1) / exp.pfs.N,
            env2_ac_prop=len(ac2) / exp.pfs.N,
            remap=anl.get_remapping(env1, env2),
            turn_pc=anl.get_turnover(env1, env2),
            turn_ac=anl.get_turnover(env1, env2, all_active=True),
            pc1=pc1,
            pc2=pc2,
            ac1=ac1,
            ac2=ac2
        )
    return data


def plot_lines(df, filename=None, linthresh=1e-3, linear=False):
    _, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 5))

    ax1.plot(df.iloc[-2:,[0, 2]])
    ax1.plot(df.iloc[-2:,[1, 3]], '--')
    ax1.set_prop_cycle(None)
    ax1.semilogx(df.iloc[:-1,[0, 2]])
    ax1.semilogx(df.iloc[:-1,[1, 3]], '--')

    if linear:
        ax1.set_xscale('linear')
    else:
        ax1.set_xscale('symlog', linthresh=linthresh, linscale=0.5, subs=range(2, 10))
    ax1.set_ylabel("Proportion Active", fontsize=12)
    ax1.legend(df.columns[[0, 2, 1, 3]])


    ax2.plot(df.iloc[-2:,4:7])
    ax2.set_prop_cycle(None)
    ax2.semilogx(df.iloc[:-1,4:7])

    if linear:
        ax2.set_xscale('linear')
    else:
        ax2.set_xscale('symlog', linthresh=linthresh, linscale=0.5, subs=range(2, 10))
    ax2.set_xlabel(df.index.name, fontsize=16)
    ax2.set_ylabel("Remapping/Turnover", fontsize=12)
    ax2.legend(df.columns[4:7])

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()