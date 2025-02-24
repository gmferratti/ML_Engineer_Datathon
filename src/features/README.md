# Descrição das Colunas de Usuário

## Colunas e Tipos de Dados

1. **`userId`**
   - **Descrição:** ID único do usuário.
   - **Tipo de Dado:** `int`
   - **Observação:** Identificação única para cada usuário.

2. **`userType`**
   - **Descrição:** Indica se o usuário está logado ou é anônimo.
   - **Tipo de Dado:** `category`
   - **Observação:** Variável categórica com valores limitados (ex.: "logado", "anônimo").

3. **`HistorySize`**
   - **Descrição:** Quantidade de notícias lidas pelo usuário.
   - **Tipo de Dado:** `int`
   - **Observação:** Representa a contagem de notícias visitadas.

4. **`history`**
   - **Descrição:** Lista de notícias visitadas pelo usuário.
   - **Tipo de Dado:** `object`
   - **Observação:** É convertida de string para lista e explodida. É a FK para mergear com as notícias.

5. **`TimestampHistory`**
   - **Descrição:** Momento em que o usuário visitou a página.
   - **Tipo de Dado:** `datetime`
   - **Observação:** Armazena a data e hora de cada visita.

6. **`timeOnPageHistory`**
   - **Descrição:** Quantidade de tempo (em milissegundos) que o usuário permaneceu na página.
   - **Tipo de Dado:** `int`
   - **Observação:** Representa o tempo total que o usuário ficou na página.

7. **`numberOfClicksHistory`**
   - **Descrição:** Quantidade de cliques feitos pelo usuário na matéria.
   - **Tipo de Dado:** `int`
   - **Observação:** Representa o número total de cliques.

8. **`scrollPercentageHistory`**
   - **Descrição:** Porcentagem da página visualizada pelo usuário.
   - **Tipo de Dado:** `float`
   - **Observação:** Representa um valor decimal indicando a porcentagem de visualização.

9. **`pageVisitsCountHistory`**
   - **Descrição:** Número de vezes que o usuário visitou a mesma matéria.
   - **Tipo de Dado:** `int`
   - **Observação:** Representa a quantidade de visitas repetidas à matéria.

## Observações Adicionais

- A coluna `TimestampHistory` foi convertida para o formato `datetime` para facilitar análises temporais.
- O uso de tipos de dados simples (`int`, `float`, `category`) visa facilitar o processamento e otimização do dataset.

# Descrição das Colunas de Notícia



1. **`page`**
   - **Descrição:** Número da página da notícia.
   - **Tipo de Dado:** `int`
   - **Observação:** Indica a página da notícia.

2. **`url`**
   - **Descrição:** URL da notícia.
   - **Tipo de Dado:** `string`
   - **Observação:** Endereço da notícia na web.

3. **`issued`**
   - **Descrição:** Data de publicação da notícia.
   - **Tipo de Dado:** `datetime`
   - **Observação:** Data e hora em que a notícia foi publicada.

4. **`modified`**
   - **Descrição:** Data de modificação da notícia.
   - **Tipo de Dado:** `datetime`
   - **Observação:** Data e hora em que a notícia foi modificada.

5. **`title`**
   - **Descrição:** Título da notícia.
   - **Tipo de Dado:** `string`
   - **Observação:** Título da notícia.

6. **`body`**
   - **Descrição:** Corpo da notícia.
   - **Tipo de Dado:** `string`
   - **Observação:** Conteúdo principal da notícia.

7. **`caption`**
   - **Descrição:** Legenda da notícia.
   - **Tipo de Dado:** `string`
   - **Observação:** Texto de legenda da notícia.
