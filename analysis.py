import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')


from experiment import Experiment
from itertools import product
from scipy.stats import pearsonr


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
    
    def get_weights(self):
        self.exp.agent.actor.clamp_weights()
        return self.exp.agent.actor.w1.data.detach()
    
    def save_retrain_remap(self, path, other_pfs, other_acs, other_pcs):
        self.exp.pfs_per_env |= other_pfs
        self.active_per_env |= other_acs
        self.place_cells_per_env |= other_pcs
        remaps = dict()
        for env1, env2 in zip(self.place_cells_per_env.keys(), other_acs.keys()):
            remaps[f"{env1}_{env1}'"] = dict(
                turnover_ac=self.get_turnover(env1, env2, all_active=True),
                turnover_pc=self.get_turnover(env1, env2),
                remappping=self.get_remapping(env1, env2)
            )
        
        pd.DataFrame(remaps).to_json(os.path.join(path, 'remaps.json'))
        
        # Remove new items
        for k in other_acs.keys():
            self.exp.pfs_per_env.pop(k)
            self.active_per_env.pop(k)
            self.place_cells_per_env.pop(k)
    
    def save_stats(self, path):
        pd.DataFrame(self.stats).to_json(os.path.join(path, "stats.json"))
    
    @staticmethod
    def load_stats(path):
        df = pd.read_json(os.path.join(path, 'stats.json'))
        df.index = pd.MultiIndex.from_tuples(df.index.map(eval))
        return df


class MultiRunAnalysis:
    def __init__(self, data_path, name, immediate_pc=False):
        data_path = os.path.join(data_path, name)
        dfs = [Analysis.load_stats(os.path.join(data_path, n)) for n in os.listdir(data_path)]
        self.exps = [Experiment.load_experiment(data_path, n) for n in os.listdir(data_path)]
        self.anls = [Analysis(exp, immediate_pc=immediate_pc) for exp in self.exps]

        self.N = len(dfs)
        df = pd.concat(dfs, keys=range(self.N))

        df_trn = df.loc[:, 1, :].groupby(level=1)
        df_unt = df.loc[:, df.index.get_level_values(1).drop(1), :].groupby(level=2)

        self.means = pd.concat([df_trn.mean(), df_unt.mean()], keys=['trn', 'unt'])
        self.stds = pd.concat([df_trn.std(), df_unt.std()], keys=['trn', 'unt'])

        frmt = lambda x: f"{x:.02f}" if x > 0.01 or x == 0 else f"{x:.01e}"
        self.ci95 = self.stds * 1.96 / np.sqrt(self.N)
        self.ci95 = self.means.map(frmt) + "$\pm$" + self.ci95.map(frmt)
        self.df = df
        
        self.stats = ...

    def get_pos_losses(self):
        return np.asarray([exp.pos_losses for exp in self.exps])

    def get_pfs_losses(self):
        losses = {env: list() for env in self.exps[0].pfs_losses.keys()}
        for exp in self.exps:
            for env, loss in exp.pfs_losses.items():
                losses[env].append(loss)
        return losses
    
    def get_weights(self):
        return torch.stack([anl.get_weights() for anl in self.anls])
    
    def print_active_cells(self):
        raise NotImplementedError()
    
    def print_place_cells(self):
        raise NotImplementedError()


class MultiAnalysis:
    LABELS = ["Place Cells (trn)", "Place Cells (unt)",
                "Active Cells (trn)", "Active Cells (unt)",
                "Remapping", "Turnover (PCs)", "Turnover (ACs)"]
    
    def __init__(self, data_path, exp_names, immediate_pc=False, multirun=True):
        self.exp_names = exp_names
        if multirun:
            self.anls = [MultiRunAnalysis(data_path,  name) for name in exp_names]
            self.df = pd.concat([anl.df for anl in self.anls], keys=exp_names)
        else:
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
    
    def nonzero_weights(self, threshold=1e-3):
        means, stds = list(), list()
        for anl in self.anls:
            nonzeros = (anl.get_weights() > threshold).sum(-1)
            means.append(nonzeros.mean(dtype=float).cpu().item())
            stds.append(nonzeros.to(torch.float32).std().cpu().item())
        return means, stds
    
    def coverage_stats(self):
        df = self.df.unstack(0).groupby(level=(1, 2)).mean()
        t1 = df.loc[1].loc[['coverage', 'scales', 'sizes']]
        t2 = df[df.index.get_level_values(0) != 1].groupby(level=1).mean().loc[['coverage', 'scales', 'sizes']]
        return pd.concat([t1, t2], keys=['trn', 'unt'])
    
    # Very rudimentary but will suffice for now
    def plot_coverage_stats(self):
        df = self.coverage_stats().loc[:, 'place']
        ax = df.loc['trn'].T.plot()
        ax.set_prop_cycle(None)
        df.loc['unt'].T.plot(ax=ax, linestyle='--')
        plt.show()

    def plot_lines(self, xticks, xlabel, filename=None, linthresh=1e-3, linear=False):
        means, cis = self.get_plot_dfs(xticks, xlabel)
        _, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 5))

        ax1.errorbar(means.index, means.iloc[:,0], yerr=cis.iloc[:,0], fmt='s-', capsize=5)
        ax1.errorbar(means.index, means.iloc[:,2], yerr=cis.iloc[:,2], fmt='s-', capsize=5)
        ax1.set_prop_cycle(None)
        ax1.errorbar(means.index, means.iloc[:,1], yerr=cis.iloc[:,1], fmt='s--', capsize=5)
        ax1.errorbar(means.index, means.iloc[:,3], yerr=cis.iloc[:,3], fmt='s--', capsize=5)

        if linear:
            ax1.set_xscale('linear')
        else:
            ax1.set_xscale('symlog', linthresh=linthresh, linscale=0.5, subs=range(2, 10))
        ax1.set_ylabel("Proportion Active", fontsize=12)
        ax1.legend(means.columns[[0, 2, 1, 3]])

        ax2.errorbar(means.index, means.iloc[:,4], yerr=cis.iloc[:,4], fmt='s-', capsize=5)
        ax2.errorbar(means.index, means.iloc[:,5], yerr=cis.iloc[:,5], fmt='s-', capsize=5)
        ax2.errorbar(means.index, means.iloc[:,6], yerr=cis.iloc[:,6], fmt='s-', capsize=5)

        if linear:
            ax2.set_xscale('linear')
        else:
            ax2.set_xscale('symlog', linthresh=linthresh, linscale=0.5, subs=range(2, 10))
        ax2.set_xlabel(means.index.name, fontsize=16)
        ax2.set_ylabel("Remapping/Turnover", fontsize=12)
        ax2.legend(means.columns[4:7])

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def get_plot_dfs(self, xticks, xlabel):
        
        means, stds = self.get_cell_data()

        means = pd.DataFrame(means).T
        means = means.rename(index={k: v for k, v in zip(self.exp_names, xticks)})
        means.index = means.index.rename(xlabel)
        means.iloc[:,:7] = means.iloc[:,:7].astype(float).fillna(0)
        means = means.rename(columns={c: l for c, l in zip(means.columns, self.LABELS)})
        
        stds = pd.DataFrame(stds)
        stds.iloc[:7] = stds.iloc[:7].astype(float).fillna(0)
        cis = stds * 2.576 / np.sqrt([anl.N for anl in self.anls]) # 99% confidence intervals
        
        return means, cis.T
    
    def get_cell_data(self):
        means, stds = dict(), dict()
        for e, anl in zip(self.exp_names, self.anls):
            means[e] = dict(
                env1_pc_prop=anl.means.loc[('trn', 'proportion'), 'place'],
                env2_pc_prop=anl.means.loc[('unt', 'proportion'), 'place'],
                env1_ac_prop=anl.means.loc[('trn', 'proportion'), 'active'],
                env2_ac_prop=anl.means.loc[('unt', 'proportion'), 'active'],
                remap=anl.means.loc[('unt', 'remapping'), 'place'],
                turn_pc=anl.means.loc[('unt', 'turnover'), 'place'],
                turn_ac=anl.means.loc[('unt', 'turnover'), 'active']
            )
            stds[e] = dict(
                env1_pc_prop=anl.stds.loc[('trn', 'proportion'), 'place'],
                env2_pc_prop=anl.stds.loc[('unt', 'proportion'), 'place'],
                env1_ac_prop=anl.stds.loc[('trn', 'proportion'), 'active'],
                env2_ac_prop=anl.stds.loc[('unt', 'proportion'), 'active'],
                remap=anl.stds.loc[('unt', 'remapping'), 'place'],
                turn_pc=anl.stds.loc[('unt', 'turnover'), 'place'],
                turn_ac=anl.stds.loc[('unt', 'turnover'), 'active']
            )
        return means, stds