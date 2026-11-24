# Notebooks de modelos simples para a narrativa do TCC — EIS

**Pacote:** regressão logística, KNN, Random Forest, modelo simbólico **inspirado em SISSO** (não implementação oficial) e PCA.

## Ordem recomendada

1. `01_modelos_basicos_entre_dispositivos.ipynb`: 49 aplicações em 9 IDEs, transferência via LOEO, comparação ilustrativa com LOO.
2. `02_modelos_basicos_E7_30_aplicacoes.ipynb`: 30 aplicações históricas do E7, 7 bandas de fase normalizadas pelo PBS; LOBO, LOCO, LOO; PCA adicional das 30 aplicações.
3. `03_PCA_exploratoria_E2_MCF10A.ipynb`: 12 aplicações de MCF10A em três condições, com PBS pareado e gráfico de deriva.

## Dependências e execução

Abra um terminal na pasta deste pacote:

```bash
pip install jupyterlab numpy pandas matplotlib scikit-learn
jupyter lab
```

Abra qualquer notebook e execute **Run All**. É necessário manter a pasta `dados/` e os arquivos `eis_utils.py` e `modelos_basicos.py` ao lado dos notebooks.

O notebook de transferência entre dispositivos exporta a figura LOEO para `../imagens_prontas/generalizacao_entre_eletrodos.pdf`. Os demais exibem suas figuras com `plt.show()`.

## Sobre o SISSO

A classe `SymbolicOneTermClassifier` é uma implementação didática restrita: cria expressões algébricas simples entre as bandas, seleciona **um único descritor** com triagem baseada no treino, depois ajusta uma regressão logística. Esse exercício demonstra parte da lógica de seleção esparsa, mas **não é uma execução do SISSO oficial**, não deve ser relatado como tal. O notebook 04_sisso_entre_dispositivos.ipynb executa o SISSO oficial 3.0.2 em modo de classificação, ajusta um SVM linear por divisão LOEO e exporta as previsões para a figura comparativa do notebook 01.

## Não contaminar o teste final prospectivo

O pacote contém apenas:
- o inventário histórico de espectros de EIS, sem o novo teste final E7;
- as características das 30 aplicações já utilizadas para desenvolvimento histórico.

**Não há leitura automática de diretórios externos ou do arquivo `Protocolo_teste_final.zip`**. O novo teste final prospectivo não foi aberto, treinado, avaliado ou usado para selecionar métodos neste pacote.

## Notas para a discussão

- Aplicações físicas ≠ réplicas biológicas; triplicatas de espectros da mesma gota são medidas técnicas.
- Entre os 9 eletrodos do recorte de 49 aplicações, as concentrações não foram uniformes; diferenças entre IDEs podem estar confundidas com concentrações/lotes.
- LOEO mede transferência para um eletrodo não visto; LOO não.
- LOBO/LOCO são avaliações históricas internas ao conjunto de 30 aplicações do E7, não o teste final prospectivo.
- PCA não é modelo de classificação, portanto não tem métrica de acurácia.
- No E2, a leitura de PBS `P05_eletrodo2(1).txt` (~1,45 Ω) é descartada da análise por curto evidente. Arquivo original preservado no ZIP de dados.

**Recomendação editorial:** mantenha no corpo principal do TCC uma figura de transferência multieletrodo, uma figura comparando os modelos básicos do E7 e uma PCA exploratória E2. Resultados adicionais podem ficar no material suplementar. Evite selecionar apenas os gráficos mais favoráveis.
