import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

import pandas_datareader.data as web
import numpy as np

FATOR_ANUAL = 252
CSV = '.csv'
SA = '.sa'


class SerieHistorica:

    def __init__(self, papel, data_inicio=None, data_fim=None, arquivo=None):
        self.papel = papel
        self.data_inicio = data_inicio
        self.data_fim = data_fim
        self.arquivo = arquivo

        self._dados_brutos = None

        self.carrega_dados_brutos()

    @property
    def dados_brutos(self):
        return self._dados_brutos

    @property
    def preco_fechamento(self):
        return self._dados_brutos.Close

    def calcula_retornos(self):
        self._dados_brutos['Return'] = np.log(self.preco_fechamento / self.preco_fechamento.shift(1))

    def carrega_dados_brutos(self):
        if self.arquivo:
            self._dados_brutos = pd.read_csv(self.arquivo, index_col='Date', parse_dates=True)
        else:
            self._dados_brutos = web.DataReader(self.papel, "yahoo", self.data_inicio, self.data_fim)
        self._dados_brutos = self._dados_brutos.interpolate()


class Carteira:
    PATH = 'dados/'
    
    def __init__(self, papeis, data_inicio=None, data_fim=None, arquivos=False):
        self.papeis = papeis
        self.data_inicio = data_inicio
        self.data_fim = data_fim
        self.arquivos = arquivos

        self._retornos_completos = None
        self._retornos_avg = None
        self._port_retorno = None
        self._port_vol = None
        self._indice_sharp = None

        self._trata_inputs()

        self.carrega_papeis()

        self._retornos = self.filtra_retorno_por_data(self.data_inicio, self.data_fim)

    @property
    def retornos(self):
        return self._retornos

    @property
    def port_retorno(self):
        return self._port_retorno

    @property
    def port_vol(self):
        return self._port_vol

    @property
    def indice_sharp(self):
        return self._indice_sharp

    def nome_arquivo(self, nome):
        nome = nome.replace(SA, '')
        return self.PATH + nome.upper() + CSV

    def summary(self):
        print('Expected Portfolio Return: ', self.port_retorno)
        print('Expected Volatility: ', self.port_vol)
        print('Sharp Ratio: ', self.indice_sharp)

    def _trata_inputs(self):
        self.papeis = [papel + SA for papel in self.papeis]

    def filtra_retorno_por_data(self, inicio, fim):
        mask = (self._retornos_completos.index > inicio) & (self._retornos_completos.index <= fim)
        return self._retornos_completos.loc[mask].dropna()

    def carrega_papeis(self):
        dict_return = dict()

        if self.arquivos:
            for papel in self.papeis:
                serie_hist = SerieHistorica(papel, arquivo=self.nome_arquivo(papel))
                serie_hist.calcula_retornos()
                dict_return[papel] = serie_hist.dados_brutos['Return']
        else:
            for papel in self.papeis:
                serie_hist = SerieHistorica(papel, self.data_inicio, self.data_fim)
                serie_hist.calcula_retornos()
                dict_return[papel] = serie_hist.dados_brutos['Return']

        aux_list = list()
        for papel in self.papeis:
            aux_list.append(dict_return[papel])
        self._retornos_completos = pd.concat(aux_list, axis=1)
        self._retornos_completos.columns = self.papeis

    def salva_papeis_csv(self):
        for papel in self.papeis:
            serie_hist = SerieHistorica(papel,  self.data_inicio, self.data_fim)
            serie_hist.dados_brutos.to_csv(self.nome_arquivo(papel))

    def calcula_retornos_avg(self):
        self._retornos_avg = self._retornos.mean()

    def calcula_port_retornos(self, pesos):
        self.calcula_retornos_avg()
        self._port_retorno = np.sum((self._retornos_avg * pesos) * FATOR_ANUAL)

    def calcula_port_vol(self, pesos):
        aux = np.dot(self._retornos.cov(), pesos)
        self._port_vol = np.sqrt(np.dot(pesos.T, aux)) * np.sqrt(FATOR_ANUAL)

    def calcula_indice_sharp(self, risk_free=0.0):
        self._indice_sharp = (self._port_retorno - risk_free) / self._port_vol

    def determina_grandezas(self, pesos):
        self.calcula_port_retornos(pesos)
        self.calcula_port_vol(pesos)
        self.calcula_indice_sharp()