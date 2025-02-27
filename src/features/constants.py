"""
Módulo de constantes para referenciação de colunas, dicionários de tipos
e demais valores usados no pipeline de processamento de dados.
"""

# --------------------------------------------------
#  NEWS-RELATED CONSTANTS
# --------------------------------------------------
NEWS_COLS_TO_CLEAN = [
    "body",
    "title",
    "caption"
]

NEWS_COLS_TO_DROP = [
    "local",
    "theme",
    "issued",
    "modified",
    "url",
    "urlExtracted",
] + NEWS_COLS_TO_CLEAN


# --------------------------------------------------
#  USERS-RELATED CONSTANTS
# --------------------------------------------------
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


# --------------------------------------------------
#  MIXED FEATURES CONSTANTS
# --------------------------------------------------
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
    "userType",
    "historySize",
    "dayPeriod",
    "isWeekend"
]


# --------------------------------------------------
#  FEATURE SETS (STATE, REGION, THEME)
# --------------------------------------------------
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


# --------------------------------------------------
#  GAP-RELATED FEATURES
# --------------------------------------------------
GAP_COLS = [
    "userId",
    "pageId",
    "timeGapDays",           # Delta entre publicação e consumo (em dias)
    "timeGapHours",          # Delta entre publicação e consumo (em horas)
    "timeGapMinutes",        # Delta entre publicação e consumo (em minutos)
    "timeGapLessThanOneDay", # Flag para consumo no mesmo dia
]


# --------------------------------------------------
#  MIXED FEATURE COLUMNS (FINAL)
# --------------------------------------------------
FINAL_MIX_FEAT_COLS = [
    "userId",
    "pageId",
    "userType",
    "isWeekend",
    "dayPeriod",
    "issuedDatetime",
    "timestampHistoryDatetime",
    "coldStart",
    "localState",
    "localRegion",
    "themeMain",
    "themeSub",
]


# --------------------------------------------------
#  CATEGORY COLUMNS (URL EXTRACTED)
# --------------------------------------------------
CATEGORY_COLS = [
    "localState",  # Tag do estado
    "localRegion", # Tag da micro-região
    "themeMain",   # Tag do tema principal
    "themeSub",    # Tag do subtema
]


# --------------------------------------------------
#  CHAVE PRIMÁRIA E DATAS
# --------------------------------------------------
KEY_FEAT_COLS = [
    "userId",                   # Id do usuário
    "pageId",                   # Id da notícia
    "issuedDatetime",           # Data/hora de publicação
    "timestampHistoryDatetime", # Data/hora de consumo
]


# --------------------------------------------------
#  SUGESTED FEATURES
# --------------------------------------------------
SUGGESTED_FEAT_COLS = KEY_FEAT_COLS + CATEGORY_COLS + [
    "userType",               # Usuário logado ou não
    "isWeekend",              # Consome notícias no FDS ou não
    "dayPeriod",              # Período do dia
    "coldStart",              # Usuário novo na plataforma
    "relLocalState",          # % relativo de notícias daquele estado
    "relLocalRegion",         # % relativo de notícias daquela região
    "relThemeMain",           # % relativo de notícias daquele tema
    "relThemeSub",            # % relativo de notícias daquele subtema
]


# --------------------------------------------------
#  TARGET FEATURES
# --------------------------------------------------
TARGET_INIT_COLS = [
    "userId",
    "pageId",
    "coldStart",
    "historySize",             # Qtd de páginas visitadas
    "numberOfClicksHistory",   # Qtd de cliques
    "timeOnPageHistory",       # Tempo na página
    "scrollPercentageHistory", # Scroll
    "minutesSinceLastVisit",   # Minutos desde última visita
    "timeGapDays",             # Gap entre release e consumo
]

TARGET_FINAL_COLS = [
    "userId",
    "pageId",
    "coldStart",
    "TARGET"
]
