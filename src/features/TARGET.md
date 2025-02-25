# Racional do Cálculo do TARGET

O score de engajamento (coluna **TARGET**) foi construído para refletir o comportamento dos usuários, sendo calculado por usuário. Para isso, utilizamos diferentes estratégias de agregação para cada variável, considerando a natureza dos dados (valores absolutos ou percentuais) e a importância de cada métrica no engajamento.

## Estratégia de Agregação

- **Agrupamento por Usuário:**  
  Os dados são agrupados pelo `userId` para que o score reflita o comportamento agregado de cada usuário.

- **Agregações:**  
  - Variáveis de **valor absoluto** (como `numberOfClicksHistory`, `timeOnPageHistory` e `pageVisitsCountHistory`) são somadas.
  - Variáveis **percentuais** (como `scrollPercentageHistory`) são calculadas como média.
  - Variáveis temporais de recência (como `minutesSinceLastVisit`) são agregadas por média, de modo a refletir o comportamento típico do usuário.

## Variáveis e Justificativas

- **numberOfClicksHistory:**  
  - **Observação:** Valores baixos (mediana = 1, média ≈ 11,87).  
  - **Peso:** 1  
  - **Justificativa:** Cada clique é contado de forma simples, sem necessidade de ajustes, já que os valores estão em escala reduzida.

- **timeOnPageHistory:**  
  - **Observação:** Valores elevados (mediana = 60.000, média ≈ 87.829).  
  - **Peso:** A variável é dividida por 500 para reduzir sua escala e, em seguida, multiplicada por 1.5 para enfatizar sua importância.  
  - **Justificativa:** O tempo de permanência na página é considerado a variável de maior importância, pois reflete fortemente o engajamento.

- **scrollPercentageHistory:**  
  - **Observação:** Valores em torno de 43 a 53 (mediana).  
  - **Peso:** 1  
  - **Justificativa:** A variável já apresenta uma escala adequada e reflete o engajamento de forma complementar, sem necessidade de ajuste adicional.

- **pageVisitsCountHistory:**  
  - **Observação:** Mediana de 1, com alguns outliers (máximo 654).  
  - **Peso:** Multiplicada por 2  
  - **Justificativa:** Para mitigar o efeito dos outliers, aplicamos uma transformação logarítmica (usando `log1p`), que suaviza a influência de valores extremamente altos. Usuários que visitam várias páginas têm seu engajamento valorizado de forma mais robusta.

- **minutesSinceLastVisit:**  
  - **Observação:** Mediana baixa (47), mas média alta (≈ 1187).  
  - **Peso:** Dividida por 20  
  - **Justificativa:** Penaliza usuários que não retornam com frequência (subtraindo do score). A divisão por 20 gera uma penalização proporcional, sem exagerar na influência dos outliers.

## Fórmula do TARGET

A fórmula ajustada para o cálculo do TARGET é:

\[
\text{TARGET} = \text{cliques} + 1.5\left(\frac{\text{tempo na página}}{500}\right) + \text{scroll} + 2 \times \text{visitas transformadas} - \frac{\text{minutos desde a última visita}}{20}
\]

Onde:
- **cliques:** `numberOfClicksHistory`
- **tempo na página:** `timeOnPageHistory`
- **scroll:** `scrollPercentageHistory`
- **visitas transformadas:** \(\log(1 + \text{pageVisitsCountHistory})\)
- **minutos desde a última visita:** `minutesSinceLastVisit`

## Considerações Finais

Este racional possibilita uma avaliação mais realista do engajamento dos usuários, alinhada com os seguintes pontos:

- **Balanceamento das escalas:**  
  Cada variável possui ordens de grandeza diferentes. As transformações (divisão, multiplicação e transformação logarítmica) permitem que todas as métricas contribuam de maneira compatível para o score.

- **Enfase na dimensão temporal:**  
  Ao aumentar o peso de `timeOnPageHistory` e ajustar a penalização de `minutesSinceLastVisit`, a métrica valoriza mais o tempo que o usuário passa na página e penaliza a falta de visitas recentes.

- **Atenuação de Outliers:**  
  A transformação logarítmica em `pageVisitsCountHistory` reduz o impacto de outliers, garantindo que valores extremos não distorçam a métrica de engajamento.

- **Valorizar a Atividade:**  
  Variáveis que indicam ação (cliques e visitas) são somadas diretamente, enquanto o scroll, representativo do engajamento, mantém seu valor sem grandes ajustes.

Com essa abordagem, o score TARGET reflete de forma robusta e equilibrada o engajamento dos usuários na plataforma.
