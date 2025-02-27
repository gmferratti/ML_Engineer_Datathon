# Pipeline de Treino

Este diretório contém o pipeline de preparação dos dados de treino e salvamento dos conjuntos de features e target. Ele centraliza os passos necessários para:

1. **Carregar dados brutos (features finais e target).**  
2. **Pré-processar e dividir em conjuntos de treino e teste.**  
3. **Salvar os datasets de treino em formato Parquet.**  
4. **Validar o carregamento dos dados salvos.**

---

## Arquivos Principais

- **`train_model.py`**  
  - **Função `train_model()`**  
    1. Lê o arquivo `final_feats_with_target.parquet` (contendo as features e o target).  
    2. Chama `prepare_features(...)` para aplicar frequency encoding em colunas categóricas e dividir os dados em treino e teste.  
    3. Salva os conjuntos de treino/teste (X_train, X_test, y_train, y_test) em formato Parquet.  
    4. Executa `load_train_data()` ao final para garantir que os arquivos foram salvos corretamente.

- **`utils.py`** (importado pelo pipeline)  
  - **Função `prepare_features(...)`**  
    Executa a preparação dos dados, incluindo:
    - **Frequency Encoding:** Para cada coluna categórica, calcula a frequência relativa dos valores e cria uma nova coluna com o sufixo `Freq`. Essa técnica transforma as variáveis categóricas em dados numéricos, refletindo a importância de cada categoria com base em sua ocorrência.
    - **Train-Test Split:** Divide os dados em conjuntos de treino e teste, permitindo a avaliação da performance do modelo em dados não vistos, prevenindo overfitting e garantindo uma melhor generalização.
  - **Função `load_train_data(...)`**  
    Carrega os arquivos `X_train.parquet` e `y_train.parquet` gerados no passo anterior, validando o pipeline.

---

## Frequency Encoding

**Frequency Encoding** é uma técnica utilizada para transformar variáveis categóricas em numéricas. Ao invés de aplicar um mapeamento arbitrário ou utilizar o one-hot encoding (que pode gerar muitas colunas), o frequency encoding substitui cada categoria pelo valor relativo de sua ocorrência no conjunto de dados. Essa abordagem traz as seguintes vantagens:

- **Redução de Dimensionalidade:** Evita a criação excessiva de colunas, o que pode ocorrer com o one-hot encoding, especialmente em variáveis com muitas categorias.
- **Informação de Relevância:** A frequência de uma categoria pode refletir sua importância no contexto do problema, fornecendo ao modelo um sinal numérico que pode ser útil para a tomada de decisão.

---

## Importância do Train-Test Split

A divisão dos dados em conjuntos de treino e teste é uma prática fundamental em machine learning, pois:

- **Avaliação Realista:** Permite treinar o modelo em um conjunto de dados (treino) e avaliar sua performance em outro conjunto (teste) que nunca foi visto durante o treinamento.
- **Prevenção de Overfitting:** Ao separar os dados, evita-se que o modelo memorize os exemplos de treinamento, promovendo uma melhor generalização para dados novos.
- **Medição de Generalização:** Utilizar dados de teste não vistos durante o treinamento possibilita avaliar a capacidade do modelo de lidar com novos cenários, garantindo uma avaliação mais robusta.

---