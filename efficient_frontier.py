from datetime import datetime
from matplotlib.pyplot import ion

from carteira.carteira import Carteira
from carteira.portifolio import PortifolioOptimization


if __name__ == '__main__':

    ion()

    data_inicio = datetime(2015, 1, 1)
    data_fim = datetime(2018, 12, 31)

    papeis = [
        'ITUB3',
        'BBDC4',
        'PETR4',
        'VALE3',
        'B3SA3',
        'MGLU3',
        'CVCB3'
    ]

    carteira = Carteira(papeis, data_inicio, data_fim, arquivos=True)
    #carteira.salva_papeis_csv()

    port_opt = PortifolioOptimization(carteira)
    port_opt.monte_carto_portifolios(50000)
    frontier_y, frontier_volatility = port_opt.efficient_frontier()
    port_opt.minimize_sharpe_ratio()

    port_opt.plot_monte_carlo_ports()
    port_opt.plot_efficient_frontier(frontier_volatility, frontier_y)