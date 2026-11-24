"""Recorte histórico compartilhado pela comparação de modelos e pelo SISSO."""
from pathlib import Path

import numpy as np
import pandas as pd

from eis_utils import band_average, read_all


def carregar_recorte(arquivo):
    aplicacoes = read_all(arquivo)
    ves = aplicacoes[aplicacoes.category.eq('VEs')]
    antigos = ves.electrode.isin(['E01', 'E03', 'E04', 'E05', 'E10', 'E13', 'E14'])
    concentracoes = ves.filename.str.contains('10e6|10e8', case=False, regex=True)
    e7 = ves.electrode.eq('E07') & ves.folder.str.endswith(('01_Campanha_inicial', '04_Treinamento_V2'))
    df = ves[(antigos & concentracoes) | ves.electrode.eq('E09') | e7].copy()
    df = df.reset_index(drop=True)
    df['classe'] = df.filename.map(identificar_linhagem)
    assert len(df) == 49 and df.electrode.nunique() == 9 and df.application.is_unique

    bordas = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    x = np.array([
        [band_average(r.f, r.z.mean(axis=0), lo, hi, log=True)
         for lo, hi in zip(bordas, bordas[1:])]
        for r in df.itertuples(index=False)
    ])
    assert x.shape == (49, 6) and np.isfinite(x).all()
    return df, x


def identificar_linhagem(nome):
    aliases = {
        'hcc': 'HCC1806', 'hcc1806': 'HCC1806',
        'mcf': 'MCF10A', 'mcf10a': 'MCF10A', 'mfc': 'MCF10A',
    }
    for token in Path(nome).stem.lower().split('_'):
        if token in aliases:
            return aliases[token]
    raise ValueError(f'Linhagem não identificada: {nome}')
