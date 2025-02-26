NEWS_COLS_TO_CLEAN = ["body", "title", "caption"]
NEWS_COLS_TO_DROP = [
    "local",
    "theme",
    "issued",
    "modified",
    "url",
    "urlExtracted",
] + NEWS_COLS_TO_CLEAN

USERS_COLS_TO_EXPLODE = [
    "history",
    "timestampHistory",
    "numberOfClicksHistory",
    "timeOnPageHistory",
    "scrollPercentageHistory",
    "pageVisitsCountHistory",
]

USERS_DTYPES = {
    "userId": "object",
    "userType": "category",
    "historySize": "int",
    "history": "object",
    "timestampHistory": "float",
    "timeOnPageHistory": "int",
    "numberOfClicksHistory": "int",
    "scrollPercentageHistory": "float",
    "pageVisitsCountHistory": "int",
}

MIX_FEATS_COLS = [
    "userId",
    "pageId",
    "issuedDate",
    "issuedTime",
    "issuedDatetime",
    "timestampHistoryDate",
    "timestampHistoryTime",
    "timestampHistoryDatetime",
    "localState",
    "localRegion",
    "themeMain",
    "themeSub",
    "coldStart",
    'userType', 
    'historySize', 
]

STATE_COLS = [
    "userId",
    "localState",
    "countLocalStateUser",
    "relLocalState",
]

REGION_COLS = [
    "userId",
    "localRegion",
    "countLocalRegionUser",
    "relLocalRegion",
]

THEME_MAIN_COLS = [
    "userId",
    "themeMain",
    "countThemeMainUser",
    "relThemeMain",
]

THEME_SUB_COLS = [
    "userId",
    "themeSub",
    "countThemeSubUser",
    "relThemeSub",
]

GAP_COLS = [
    "userId",
    "pageId",
    "timeGapDays", # Delta entre publicação e consumo (em dias)
    "timeGapHours", # Delta entre publicação e consumo (em horas)
    "timeGapMinutes", # Delta entre publicação e consumo (em minutos)
    "timeGapLessThanOneDay" # Flag que indica se o usuário consumiu a notícia no mesmo dia
]

FINAL_MIX_FEAT_COLS = [
    "userId",
    "pageId",
    "userType",
    "historySize",
    "issuedDatetime",
    "timestampHistoryDatetime",
    "coldStart",
    "localState",
    "localRegion",
    "themeMain",
    "themeSub",
]

CATEGORY_COLS = [
    "localState", # Campo de tag (da URL) do estado em que a notícia se refere
    "localRegion", # Campo de tag (da URL) da micro-região em que a notícia se refere
    "themeMain", # Campo de tag (da URL) do tema principal da notícia
    "themeSub", # Campo de tag (da URL) do subtema da notícia
]

KEY_FEAT_COLS = [
    "userId", # Id do usuário
    "pageId", # Id da notícia
    "issuedDatetime", # Data de publicação da notícia
    "timestampHistoryDatetime", # Data de consumo da notícia pelo usuário
]

SUGGESTED_FEAT_COLS = KEY_FEAT_COLS + [
    "userType", # Se o usuário está logado ou não
    "coldStart", # Flag que indica se o usuário é ou não novo na plataforma (coldStart)
    "relLocalState", # Percentual relativo do consumo de notícias daquele estado entre todas as notícias consumidas pelo usuário
    "relLocalRegion", # Percentual relativo do consumo de notícias daquela região entre todas as notícias consumidas pelo usuário
    "relThemeMain", # Percentual relativo do consumo de notícias daquele tema entre todas as notícias consumidas pelo usuário
    "relThemeSub", # Percentual relativo do consumo de notícias daquele subtema entre todas as notícias consumidas pelo usuário
] + CATEGORY_COLS

TARGET_INIT_COLS = [
    "userId", 
    "pageId",
    "historySize", # Quantidade de páginas visitadas pelo usuário
    "numberOfClicksHistory", # Quantidade de cliques na página
    "timeOnPageHistory", # Tempo despendido na página
    "scrollPercentageHistory", # Percentual de scroll da página
    "minutesSinceLastVisit", # Minutos desde a última visita
    "timeGapDays", # Gap entre release da notícia e consumo pelo usuário
]

TARGET_FINAL_COLS = [
    "userId", 
    "pageId",
    "TARGET"
]