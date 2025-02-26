# Racional do Cálculo do TARGET

O score de engajamento (coluna **TARGET**) foi criado para refletir o comportamento de cada par **usuário-notícia**. Dessa forma, o cálculo avalia individualmente como o usuário interage com cada notícia (representada pela coluna `pageId`), permitindo identificar, de forma granular, os níveis de engajamento e orientar ações específicas conforme o consumo de cada conteúdo.

## Estratégia de Agregação

- **Agrupamento por Par Usuário-Notícia:**  
  O cálculo é realizado para cada combinação de `userId` e `pageId`, possibilitando uma avaliação individualizada do engajamento para cada notícia consumida.

- **Agregações e Ajustes:**  
  - Variáveis de **valor absoluto** (como `numberOfClicksHistory` e `timeOnPageHistory`) são utilizadas após ajustes que compatibilizam as escalas.  
  - Variáveis **percentuais** (como `scrollPercentageHistory`) são mantidas em seus valores originais.  
  - Variáveis de recência, como `minutesSinceLastVisit`, penalizam o score para pares em que o usuário demora mais a retornar, convertendo minutos para horas (divisão por 60).  
  - **historySize:** Multiplica o score base, normalizado pela média (130), para recompensar usuários com um maior volume de visitas.  
  - **Gap de Tempo:** Um fator multiplicativo, baseado em `timeGapDays`, penaliza os pares com maior intervalo entre a publicação da notícia e sua visualização, suavizado ao dividir o gap por 50.

## Variáveis e Justificativas

- **numberOfClicksHistory:**  
  - **Observação:** Geralmente apresenta valores baixos.  
  - **Peso:** 1  
  - **Justificativa:** Cada clique indica uma interação direta, contribuindo positivamente para o engajamento.

- **timeOnPageHistory:**  
  - **Observação:** Apresenta valores elevados.  
  - **Ajuste:** Dividido por 1000 para adequar a escala e multiplicado por 1.5 para enfatizar sua importância.  
  - **Peso:** 1.5 (após normalização)  
  - **Justificativa:** O tempo que o usuário passa na página é um forte indicativo de interesse e interação com o conteúdo.

- **scrollPercentageHistory:**  
  - **Observação:** Reflete o percentual de conteúdo visualizado.  
  - **Peso:** 1  
  - **Justificativa:** Um maior percentual de scroll indica um aprofundamento na leitura, complementando a avaliação do engajamento.

- **minutesSinceLastVisit:**  
  - **Observação:** Pode variar significativamente entre os pares.  
  - **Ajuste:** Dividido por 60 para converter minutos em horas, aplicando uma penalização mais suave.  
  - **Justificativa:** Penaliza pares em que o usuário demora mais a retornar, sugerindo menor engajamento.

- **historySize:**  
  - **Observação:** Representa a quantidade de páginas visitadas historicamente pelo usuário.  
  - **Ajuste:** Multiplica o score base por (historySize / 130), normalizando pela média para recompensar usuários com maior volume de visitas.  
  - **Justificativa:** Usuários que visitam mais páginas recebem um impulso no score, refletindo um engajamento mais robusto.

- **timeGapDays:**  
  - **Observação:** Representa o intervalo, em dias, entre a publicação da notícia e sua visualização.  
  - **Ajuste:** Penalizado por um fator calculado como 1 / (1 + timeGapDays / 50), que suaviza a influência de gaps elevados.  
  - **Justificativa:** Quanto menor o gap, maior o engajamento, pois indica que o usuário consumiu a notícia logo após sua publicação; gaps maiores reduzem o score de forma gradual.

## Fórmula do TARGET

A fórmula para o cálculo do **TARGET** é definida como:

\[
\text{TARGET} = \left(\text{numberOfClicksHistory} + 1.5 \times \frac{\text{timeOnPageHistory}}{1000} + \text{scrollPercentageHistory} - \frac{\text{minutesSinceLastVisit}}{60}\right) \times \frac{\text{historySize}}{130} \times \frac{1}{1 + \frac{\text{timeGapDays}}{50}}
\]

Onde:
- **numberOfClicksHistory:** Número de cliques históricos.
- **timeOnPageHistory:** Tempo na página, ajustado dividindo por 1000 e multiplicado por 1.5.
- **scrollPercentageHistory:** Percentual de scroll realizado.
- **minutesSinceLastVisit:** Minutos desde a última visita, penalizados pela divisão por 60.
- **historySize:** Quantidade de páginas visitadas, utilizada para recompensar usuários com maior volume (normalizada pela média de 130).
- **timeGapDays:** Dias entre a publicação e a visualização, penalizados de forma suavizada pela divisão por 50.

## Padronização do Score

Após o cálculo bruto do **TARGET**, é aplicada uma padronização utilizando robust scaling:
- **Subtração da Mediana:** Centraliza os dados.
- **Divisão pelo IQR (Intervalo Interquartil):** Reduz a influência de outliers.

Essa padronização garante que os scores estejam em uma escala comparável entre os diferentes pares usuário-notícia.

## Considerações Finais

Esta abordagem possibilita:

- **Análise Granular:** A avaliação por par usuário-notícia permite identificar nuances no comportamento e no engajamento de forma precisa.
- **Balanceamento de Escalas:** As transformações aplicadas ajustam as diferentes ordens de grandeza das variáveis, assegurando que todas contribuam de maneira proporcional.
- **Enfoque na Dimensão Temporal:** Ao incorporar o gap de tempo (`timeGapDays`), o score valoriza a relevância temporal do conteúdo, premiando visualizações rápidas e penalizando atrasos.
- **Recompensa pelo Volume de Visitas:** Multiplicar o score base pela razão (historySize / 130) impulsiona o score de usuários com maior histórico de visitas.
- **Medição Robusta do Engajamento:** A combinação crítica das métricas históricas com ajustes de escala resulta em um score que reflete de forma equilibrada a interação dos usuários com cada notícia.

Com esse racional, o **TARGET** fornece uma métrica inédita e refinada, alinhada com o comportamento real dos usuários em relação a cada notícia, permitindo insights mais detalhados e decisões mais informadas.
