import unittest
from datetime import datetime

import mock

from carteira.carteira import Carteira
from carteira.portifolio import PortifolioOptimization


class TestEfficientFrontier(unittest.TestCase):
	PAPEIS = ['ITUB3', 'CVCB3']

	DATA_INICIO = datetime(2017, 1, 1)
	DATA_FIM = datetime(2018, 12, 31)

	@classmethod
	def setUpClass(cls):
		Carteira.PATH = 'testes/'
		cls.carteira = Carteira(cls.PAPEIS, cls.DATA_INICIO, cls.DATA_FIM, arquivos=True)
		cls.port_opt = PortifolioOptimization(cls.carteira)

	def test_efficient_frontier(self):
		x, y = self.port_opt.efficient_frontier()

		self.assertAlmostEqual(sum(x), 39.99999999999997, places=6)
		self.assertAlmostEqual(sum(y), 29.274905405643363, places=6)

	@mock.patch('numpy.random.random')
	def test_monte_carlo_portfolio(self, random_call):
		random_call.return_value = [0.5, 0.5]

		self.port_opt.monte_carlo_portifolios(1)
		
		self.assertEqual(self.carteira.port_retorno, 0.3415798576049637)
		self.assertEqual(self.carteira.port_vol, 0.25600244589466864)
		self.assertEqual(self.carteira.indice_sharp, 1.3342835706558271)

	def test_minimize_sharp_ratio(self):
		self.port_opt.minimize_sharpe_ratio()
		
		self.assertListEqual(self.port_opt.all_weights.tolist(), [0.1941240929978592, 0.8058759070021408])

		self.assertEqual(self.carteira.port_retorno, 0.4250143425377688)
		self.assertEqual(self.carteira.port_vol, 0.30521901684235153)
		self.assertEqual(self.carteira.indice_sharp, 1.3924897174978212)
