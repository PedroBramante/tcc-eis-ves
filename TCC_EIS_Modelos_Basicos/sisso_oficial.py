"""Execução do SISSO oficial 3.0.2 em cada treino de classificação."""
import ast
import operator
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROTULO_SISSO = 'SISSO 3.0.2 + SVM linear'
CLASSES = ['MCF10A', 'HCC1806']
DIMENSAO = 2


def localizar_sisso(raiz):
    exe_local = raiz / '.venv' / 'sisso-cygwin' / 'bin' / 'SISSO.exe'
    exe = Path(os.environ.get('SISSO_EXE') or shutil.which('SISSO') or exe_local).resolve()
    if not exe.is_file():
        raise FileNotFoundError('Aponte SISSO_EXE para o executável oficial SISSO 3.0.2.')
    return exe


def avaliar_expressao(expressao, matriz):
    colunas = {f'z{i}': coluna for i, coluna in enumerate(matriz.T, start=1)}
    operacoes = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def calcular(no):
        if isinstance(no, ast.Name) and no.id in colunas:
            return colunas[no.id]
        if isinstance(no, ast.BinOp) and type(no.op) in operacoes:
            return operacoes[type(no.op)](calcular(no.left), calcular(no.right))
        if isinstance(no, ast.UnaryOp) and isinstance(no.op, ast.USub):
            return -calcular(no.operand)
        raise ValueError(f'Expressão fora dos operadores configurados: {expressao}')

    return calcular(ast.parse(expressao, mode='eval').body)


def ajustar_sisso(x_treino, y_treino, exe):
    nomes = [f'z{i}' for i in range(1, x_treino.shape[1] + 1)]
    ordem = np.concatenate([np.flatnonzero(y_treino == classe) for classe in CLASSES])
    contagens = [int(np.sum(y_treino == classe)) for classe in CLASSES]
    assert min(contagens) > 0 and len(ordem) == len(x_treino)
    parametros = f'''ptype=2
ntask=1
desc_dim={DIMENSAO}
nsample=({contagens[0]},{contagens[1]})
restart=.false.
nsf={len(nomes)}
opset='(+)(-)(*)'
rung=1
subs_sis=30
method='L0'
nm_output=1
isconvex=(1,1)
width=0.001
'''
    with TemporaryDirectory(prefix='eis_sisso_') as diretorio:
        pasta = Path(diretorio)
        linhas = ['amostra ' + ' '.join(nomes)]
        for i in ordem:
            valores = ' '.join(f'{v:.17g}' for v in x_treino[i])
            linhas.append(f'app{i:03d} {valores}')
        (pasta / 'train.dat').write_text('\n'.join(linhas) + '\n', encoding='ascii')
        (pasta / 'SISSO.in').write_text(parametros, encoding='ascii')
        for nome in ('desc_dat', 'feature_space', 'residual', 'models'):
            (pasta / nome).mkdir()
        ambiente = dict(os.environ, OMP_NUM_THREADS='1')
        ambiente['PATH'] = str(exe.parent) + os.pathsep + ambiente.get('PATH', '')
        processo = subprocess.run([str(exe)], cwd=pasta, env=ambiente, capture_output=True,
                                  text=True, errors='replace', timeout=120)
        saida = pasta / 'SISSO.out'
        log = saida.read_text(encoding='ascii', errors='replace') if saida.is_file() else ''
        cabecalho = f'{DIMENSAO}D descriptor (model):'
        if processo.returncode or cabecalho not in log or 'SISSO.3.0.2' not in log:
            raise RuntimeError(f'SISSO falhou:\n{processo.stdout}\n{processo.stderr}\n{log[-4000:]}')
        expressoes = re.findall(r'(?m)^[ \t]+\d+:\[(.+?)\][ \t]*$',
                                log.split(cabecalho, 1)[1])
        assert len(expressoes) == DIMENSAO, expressoes
        coordenadas = np.loadtxt(pasta / 'desc_dat' / f'desc_{DIMENSAO:03d}d_p001.dat',
                                 skiprows=1, usecols=tuple(range(2, 2 + DIMENSAO)))

    reconstruido = np.column_stack([avaliar_expressao(e, x_treino) for e in expressoes])
    np.testing.assert_allclose(reconstruido[ordem], coordenadas, rtol=5e-8, atol=5e-10)
    return expressoes, reconstruido


def prever_sisso(x_treino, y_treino, x_teste, exe):
    expressoes, coordenadas_treino = ajustar_sisso(x_treino, y_treino, exe)
    coordenadas_teste = np.column_stack([avaliar_expressao(e, x_teste) for e in expressoes])
    svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
    svm.fit(coordenadas_treino, y_treino)
    return svm.predict(coordenadas_teste), expressoes
