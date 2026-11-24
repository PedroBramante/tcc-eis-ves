"""Leitura e agregação reprodutível de espectros EIS.

Não salva figuras nem modifica os arquivos brutos.
Uma aplicação física pode conter múltiplas leituras técnicas.
"""
from __future__ import annotations
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd

FREQUENCY='Frequency (Hz)'
MODULE='Z (Ω)'
PHASE='-Phase (°)'


def _category(electrode: str, path: str) -> str:
    base=Path(path).name.lower()
    if electrode=='E02':
        if re.match(r'^p\d{2}_', base): return 'PBS'
        if re.match(r'^a\d{2}_', base): return 'VEs'
    if ('h2o' in base) or ('h20' in base) or ('agua' in base): return 'Água'
    if base.startswith('controle_') and not base.startswith('controle_pbs'):
        return 'Controle processado'
    if 'pbs' in base or re.match(r'^p\d{2}_',base): return 'PBS'
    if electrode=='E07' and re.match(r'^t\d{2}_', base): return 'VEs'
    if 'ves' in base or re.search(r'\bve\b',base): return 'VEs'
    if 'controle' in base or 'control' in base: return 'Controle processado'
    return 'Outro'


def _stem_apply(electrode, path):
    """Normaliza o nome apenas para agrupar triplicatas técnicas."""
    p=Path(path)
    basename=p.name
    if electrode=='E09':
        basename=re.sub(r'_R\d+(?=\.txt$)', '', basename, flags=re.IGNORECASE)
    if electrode in ('E07','E02','E09'):
        basename=re.sub(r'\(\d+\)(?=\.txt$)', '', basename)
    return (str(p.parent),basename)


def _sort_readings(electrode, filename):
    if electrode=='E09':
        m=re.search(r'_R(\d+)(?=\.txt$)',filename,re.IGNORECASE)
        return int(m.group(1))-1 if m else 0
    m=re.search(r'\((\d+)\)(?=\.txt$)', filename)
    return int(m.group(1)) if m else 0


def read_all(zip_path, drop_known_short=True):
    """Retorna uma tabela com uma linha por *aplicação*, listando os espectros técnicos.

    A leitura P05_eletrodo2(1).txt, ~1.45 Ω, é excluída apenas se drop_known_short=True.
    A aplicação P05 permanece com três outras leituras.
    """
    grouped=defaultdict(list)
    skipped=[]
    with zipfile.ZipFile(zip_path) as z:
        for path in z.namelist():
            if not path.lower().endswith('.txt'): continue
            elect=path.split('/',1)[0]
            if not re.fullmatch(r'E\d{2}',elect): continue
            if (drop_known_short and elect=='E02' and Path(path).name=='P05_eletrodo2(1).txt'):
                skipped.append(path)
                continue
            try: df=pd.read_csv(BytesIO(z.read(path)))
            except Exception as exc: raise RuntimeError(f'Falha ao ler {path}: {exc}') from exc
            needed=[FREQUENCY,MODULE,PHASE]
            if not set(needed).issubset(df.columns): raise ValueError((path,df.columns.to_list()))
            df=df.sort_values(FREQUENCY).reset_index(drop=True)
            df=df[[FREQUENCY,MODULE,PHASE,"Z' (Ω)","-Z'' (Ω)"]].copy()
            df[needed]=df[needed].apply(pd.to_numeric,errors='coerce')
            key=(elect,*_stem_apply(elect,path))
            grouped[key].append((path,df))
    rows=[]
    for (elect,parent,base),readings in grouped.items():
        readings=sorted(readings,key=lambda a: _sort_readings(elect,Path(a[0]).name))
        fs=readings[0][1][FREQUENCY].to_numpy(float)
        for nm,d in readings[1:]:
            if len(d)!=len(fs) or not np.allclose(d[FREQUENCY],fs,rtol=1e-4):
                raise ValueError(f'Malha de frequências incompatível em {nm}')
        z_stack=np.stack([d[MODULE].to_numpy(float) for _,d in readings])
        p_stack=np.stack([d[PHASE].to_numpy(float) for _,d in readings])
        rows.append(dict(electrode=elect,category=_category(elect,base),application=f'{parent}/{base}',
                         filename=base,folder=parent,n_scans=len(readings),
                         files=[a for a,_ in readings],f=fs, z=z_stack,p=p_stack))
    result=pd.DataFrame(rows).sort_values(['electrode','application']).reset_index(drop=True)
    result.attrs['excluded_files']=skipped
    return result


def band_average(fs,values,low,high,log=False):
    """Média das frequências pertencentes à década [low,high) (última inclui ponto final)."""
    fs=np.asarray(fs)
    mask=(fs>=low-1e-8)&(fs<(high+1e-8 if high>=1e6 else high))
    x=np.asarray(values)[...,mask]
    if not x.size: return np.nan
    if log: x=np.log10(np.clip(x,1e-30,None))
    return float(np.mean(x))


def at_frequency(fs,values,target):
    idx=int(np.argmin(np.abs(np.log10(np.asarray(fs))-np.log10(target))))
    return float(np.asarray(values)[...,idx].mean())


def get_e02_pair_table(df):
    """Faz o pareamento explícito Axx/Pxx: referência PBS que precede Axx."""
    sel=df[df.electrode.eq('E02')].copy()
    rows=[]
    for i in range(1,13):
        a=sel[sel.filename.str.startswith(f'A{i:02d}_')]
        p=sel[sel.filename.str.startswith(f'P{i:02d}_')]
        if len(a)!=1 or len(p)!=1:
            raise ValueError(f'Esperava exatamente um A{i:02d} e um P{i:02d}: {len(a)}, {len(p)}')
        a,p=a.iloc[0],p.iloc[0]
        name=a.filename.lower()
        condition=('Controle1' if 'controle1' in name else 'Controle2' if 'controle2' in name else 'cond.HCC')
        if not np.allclose(a.f,p.f): raise ValueError(f'Frequências distintas A{i:02d}/P{i:02d}')
        z_a,z_p=a.z.mean(axis=0),p.z.mean(axis=0)
        phase_diff=a.p.mean(axis=0)-p.p.mean(axis=0)
        rows.append(dict(order=i,condition=condition,application=f'A{i:02d}',
                         scans_ves=a.n_scans,scans_pbs=p.n_scans,f=a.f,
                         z_ves=z_a,z_pbs=z_p,phase_delta=phase_diff,
                         z_percent=100*(z_a/z_p-1),logratio_z=np.log10(z_a/z_p)))
    return pd.DataFrame(rows)
