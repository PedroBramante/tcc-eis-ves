# Dados e modelos de EIS do TCC

Material complementar da monografia sobre a caracterização de preparações de vesículas extracelulares por espectroscopia de impedância eletroquímica (EIS).

- `Medicoes_EIS_Por_Dispositivo/` e `inventario.csv` reúnem as medições e o inventário das aquisições.
- `Protocolo_teste_final/` e `previsoes_knn_teste_final.csv` contêm as medições do teste final e as previsões por aplicação.
- Os notebooks numerados na raiz documentam as comparações espectrais, a repetibilidade, os modelos do E7, o teste final e o PCA do E2.
- `TCC_EIS_Modelos_Basicos/` inclui o código dos modelos, dados processados e notebooks, incluindo a avaliação entre eletrodos.
- As pastas `imagens_prontas*` contêm as figuras em PDF.

## Reprodução

Instale as dependências com `python -m pip install -r requirements.txt` e abra o JupyterLab na raiz do repositório. Para as células que executam o SISSO, instale o SISSO oficial 3.0.2 separadamente e aponte a variável de ambiente `SISSO_EXE` para o executável. Os notebooks registram os resultados das execuções.
