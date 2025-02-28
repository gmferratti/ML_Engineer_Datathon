# Features & Target

Este documento descreve as principais **features** que farão parte do treino do nosso modelo, bem como a lógica para construção do **TARGET** — um score de engajamento calculado por par **usuário-notícia**.

---

## 1. Estrutura de Features

As variáveis de entrada do modelo (features) foram definidas para capturar diversos aspectos do comportamento do usuário, da natureza do conteúdo e do momento de consumo da notícia. Abaixo, detalhamos a forma como essas colunas estão organizadas.

### 1.1 Identificação & Datas

Para cada interação (par **usuário-notícia**), registram-se as seguintes colunas:

- **`userId`**: Uma hash que identifica unicamente o usuário.  
- **`pageId`**: Uma hash que garante a unicidade de cada notícia dentro da base.  
- **`issuedDatetime`**: Data e hora de **publicação** da notícia.  
- **`timestampHistoryDatetime`**: Data e hora de **consumo** da notícia.

Essas informações são cruciais para avaliar **quando** e **como** o usuário se engajou com o conteúdo, tornando possível relacionar o **tempo de vida** de uma notícia com o momento em que ela foi lida.

### 1.2 Categorias

As colunas de categoria a seguir são obtidas a partir da **decomposição das URLs** do G1:

- **`localState`**: Estado referente à notícia (por exemplo: “sp”, “rj”, “mg”).  
- **`localRegion`**: Subdivisão ou microrregião do estado (por exemplo: “sao-paulo”, “rio-de-janeiro”).  
- **`themeMain`**: Tema principal da notícia (por exemplo: “economia”, “politica”, “esporte”).  
- **`themeSub`**: Subtema específico (por exemplo: “bolsa-de-valores”, “eleicoes-2022”).

Essas variáveis permitem relacionar o conteúdo consumido pelo usuário com **regiões** e **temas** que ele costuma acessar, dando suporte a análises mais segmentadas.

### 1.3 Features Sugeridas

As **features sugeridas** agregam contexto adicional, levando em conta aspectos do usuário e do momento do consumo:

- **`userType`**: Indica se o usuário está logado ou não.  
- **`isWeekend`**: Flag booleana que identifica se o consumo ocorreu no fim de semana.  
- **`dayPeriod`**: Período do dia em que a notícia foi consumida (por exemplo: “morning”, “afternoon”, “night”).  
- **`coldStart`**: Indica se o usuário é novo na plataforma, assumindo `True` para quem visitou menos de 5 notícias.  
- **`relLocalState`**, **`relLocalRegion`**, **`relThemeMain`**, **`relThemeSub`**: Representam a fração de consumo do usuário para cada local ou tema, considerando o total de notícias que ele já acessou. Por exemplo, se `relLocalState` = 0.40, significa que 40% das notícias consumidas por aquele usuário pertencem a determinado estado.

A combinação dessas colunas possibilita analisar o **comportamento do usuário** sob diversos ângulos, como tema preferido, horário de consumo, localidade mais acessada, etc.

---

## 2. Score de Engajamento (TARGET)

A coluna **TARGET** avalia o engajamento do usuário com cada notícia. O engajamento leva em conta diferentes elementos, como cliques, tempo de leitura, percentual de scroll, recência da visita e o intervalo entre a publicação e o consumo da notícia.

### 2.1 Estratégia de Cálculo

1. **Par Usuário-Notícia**  
   Cada valor do TARGET é calculado de forma individual, refletindo o engajamento daquele **usuário** em relação a uma **notícia** específica.

2. **Componentes**  
   - **Número de Cliques**: Indica ações diretas que o usuário faz na notícia.  
   - **Tempo na Página**: Tempo total em que a notícia permaneceu aberta, refletindo aprofundamento na leitura.  
   - **Scroll**: Percentual que representa o quanto da página foi percorrido pelo usuário (quanto mais, maior o engajamento).  
   - **Recência**: Penaliza o score conforme aumenta o tempo desde a última visita (usuários que voltam rápido são considerados mais engajados).  
   - **Histórico (historySize)**: Usuários com maior volume de páginas visitadas recebem um multiplicador maior.  
   - **Gap de Publicação (timeGapDays)**: Avalia o intervalo entre a data de publicação e o consumo; quanto maior o gap, menor a pontuação de engajamento.

3. **Penalizações e Ajustes**  
   - **Tempo** é normalizado para ficar na mesma ordem de grandeza de outras variáveis (convertido de ms para segundos e multiplicado por um fator).  
   - **Cliques** e **Scroll** são somados ao tempo para formar um “score base”.  
   - **Recência** é subtraída, reduzindo o score se o usuário demora a retornar.  
   - **Histórico** multiplica esse resultado, normalizando pela média (tipicamente 130).  
   - **Gap de Publicação** aplica um fator decrescente: quanto maior o intervalo entre publicação e consumo, menor o score final.

### 2.2 Fórmula de Cálculo Inicial

A expressão geral para o cálculo do **score base** (antes das transformações) pode ser representada como:

~~~text
scoreBase = (numberOfClicksHistory
             + 1.5 * (timeOnPageHistory / 1000)
             + scrollPercentageHistory
             - (minutesSinceLastVisit / 60))
~~~

e então multiplicamos pelos fatores de histórico e o gap em dias:

~~~text
rawScore = scoreBase
           * (historySize / 130)
           * (1 / (1 + (timeGapDays / 50)))
~~~

### 2.3 Transformações e Escalonamento

Para aumentar a variância e impedir que valores altos dominem o conjunto, aplicamos:

1. **Corte em Zero**: Se o `rawScore` for negativo, definimos como 0 para evitar problemas no log.
2. **Transformação Logarítmica**: `log1p(rawScore)`, que espalha mais a distribuição.
3. **Min-Max Scaling**: Escalona o resultado para **[0, SCALING_RANGE]**.
4. **Arredondamento**: Converte o valor final para inteiro.

Dessa forma, valores muito baixos não ficam “espremidos” em torno de zero, e valores muito altos são comprimidos, criando uma variação mais equilibrada.

---
