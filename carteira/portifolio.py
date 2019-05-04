import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class PortifolioOptimization:

    def __init__(self, carteira):
        self.carteira = carteira
        self._num_papeis = len(self.carteira.retornos.columns)

        self.all_weights = None
        self.ret_arr = None
        self.vol_arr = None
        self.sharpe_arr = None
        self.max_sr_ret = None
        self.max_sr_vol = None

    @property
    def num_papeis(self):
        return self._num_papeis

    @staticmethod
    def print_pesos(papeis, pesos):
        for papel, peso in zip(papeis, pesos):
            print('{}: {}'.format(papel, round(peso, 3)))

    def summary_monte_carlo(self, arg_max_ind):
        print()
        print('------ Monte Carlo Simulation ----------------')
        self.print_pesos(self.carteira.papeis, self.all_weights[arg_max_ind, :])
        print('Max Ret: ', self.ret_arr[arg_max_ind])
        print('Max Vol: ', self.vol_arr[arg_max_ind])
        print('Max Sharp: ', self.sharpe_arr.max())
        print('----------------------------------------------')

    def summary_minimize_sharpe_ratio(self, opt):
        print()
        print('------ Sharpe Ratio Minimize ----------------')
        self.print_pesos(self.carteira.papeis, opt.x)
        self.carteira.summary()
        print('---------------------------------------------')

    def monte_carto_portifolios(self, num_ports=5000):
        self.all_weights = np.zeros((num_ports, self.num_papeis))
        self.ret_arr = np.zeros(num_ports)
        self.vol_arr = np.zeros(num_ports)
        self.sharpe_arr = np.zeros(num_ports)

        for ind in range(num_ports):
            pesos = np.array(np.random.random(self.num_papeis))
            pesos = pesos / np.sum(pesos)
            self.carteira.determina_grandezas(pesos)

            self.all_weights[ind, :] = pesos
            self.ret_arr[ind] = self.carteira.port_retorno
            self.vol_arr[ind] = self.carteira.port_vol
            self.sharpe_arr[ind] = self.carteira.indice_sharp

        arg_max_ind = self.sharpe_arr.argmax()
        self.max_sr_ret = self.ret_arr[arg_max_ind]
        self.max_sr_vol = self.vol_arr[arg_max_ind]

        self.summary_monte_carlo(arg_max_ind)

    def minimize_sharpe_ratio(self):

        def neg_sharpe(pesos):
            self.carteira.determina_grandezas(pesos)
            return -1 * self.carteira.indice_sharp

        def check_sum(pesos):
            return np.sum(pesos) - 1

        cons = ({'type': 'eq', 'fun': check_sum})
        bounds = tuple([(0, 1) for _ in range(self.num_papeis)])
        init_guess = (1 / self.num_papeis) * np.ones(self.num_papeis)

        opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        self.all_weights = opt_results.x

        self.summary_minimize_sharpe_ratio(opt_results)

    def efficient_frontier(self, inicio=0.15, fim=0.65, step=100):

        def check_sum(pesos):
            return np.sum(pesos) - 1

        def calcula_ret(peso):
            self.carteira.calcula_port_retornos(peso)
            return self.carteira.port_retorno

        def calcula_vol(peso):
            self.carteira.calcula_port_vol(peso)
            return self.carteira.port_vol

        bounds = tuple([(0, 1) for _ in range(self.num_papeis)])
        init_guess = (1 / self.num_papeis) * np.ones(self.num_papeis)

        frontier_y = np.linspace(inicio, fim, step)
        frontier_volatility = []

        for possible_return in frontier_y:
            cons = ({'type': 'eq', 'fun': check_sum},
                    {'type': 'eq', 'fun': lambda w: calcula_ret(w) - possible_return})

            result = minimize(calcula_vol,
                              init_guess,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=cons)

            frontier_volatility.append(result['fun'])

        return frontier_y, frontier_volatility

    def plot_monte_carlo_ports(self):
        plt.figure(1, figsize=(12, 8))
        plt.scatter(self.vol_arr, self.ret_arr, c=self.sharpe_arr, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.scatter(self.max_sr_vol, self.max_sr_ret, c='red', s=50, edgecolors='red')

        plt.show()

    def plot_efficient_frontier(self,frontier_volatility, frontier_y):
        plt.figure(1)
        plt.plot(frontier_volatility, frontier_y, 'g--', linewidth=3)
        plt.scatter(self.carteira.port_vol, self.carteira.port_retorno, c='black', s=50, edgecolors='black')
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.show()
