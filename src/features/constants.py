"""
Constantes para colunas e parâmetros do pipeline.
"""

# Users-related constants
USERS_COLS_TO_EXPLODE = [
    "history", "timestampHistory", "numberOfClicksHistory",
    "timeOnPageHistory", "scrollPercentageHistory", "pageVisitsCountHistory"
]
USERS_DTYPES = {
    "userId": "object", "userType": "category", "historySize": "int",
    "history": "object", "timestampHistory": "float",
    "timeOnPageHistory": "int", "numberOfClicksHistory": "int",
    "scrollPercentageHistory": "float", "pageVisitsCountHistory": "int"
}

# News-related constants
NEWS_COLS_TO_CLEAN = ["body", "title", "caption"]
NEWS_COLS_TO_DROP = ["local", "theme", "issued", "modified", "url",
                     "urlExtracted"] + NEWS_COLS_TO_CLEAN

# Mixed features
MIX_FEATS_COLS = [
    "userId", "pageId", "issuedDate", "issuedTime", "issuedDatetime",
    "timestampHistoryDate", "timestampHistoryTime", "timestampHistoryDatetime",
    "localState", "localRegion", "themeMain", "themeSub", "coldStart",
    "userType", "historySize", "dayPeriod", "isWeekend"
]

# Feature sets
STATE_COLS = ["userId", "localState", "countLocalStateUser", "relLocalState"]
REGION_COLS = ["userId", "localRegion", "countLocalRegionUser", "relLocalRegion"]
THEME_MAIN_COLS = ["userId", "themeMain", "countThemeMainUser", "relThemeMain"]
THEME_SUB_COLS = ["userId", "themeSub", "countThemeSubUser", "relThemeSub"]

# Gap-related features
GAP_COLS = [
    "userId", "pageId", "timeGapDays", "timeGapHours",
    "timeGapMinutes", "timeGapLessThanOneDay"
]

# Final mixed feature columns
FINAL_MIX_FEAT_COLS = [
    "userId", "pageId", "userType", "isWeekend", "dayPeriod",
    "issuedDatetime", "timestampHistoryDatetime", "coldStart",
    "localState", "localRegion", "themeMain", "themeSub"
]

# Category columns
CATEGORY_COLS = ["localState", "localRegion", "themeMain", "themeSub"]

# Primary key and date columns
KEY_FEAT_COLS = [
    "userId", "pageId", "issuedDatetime", "timestampHistoryDatetime"
]

# Suggested features
SUGGESTED_FEAT_COLS = (
    KEY_FEAT_COLS + CATEGORY_COLS + [
        "userType", "isWeekend", "dayPeriod", "coldStart",
        "relLocalState", "relLocalRegion", "relThemeMain", "relThemeSub"
    ]
)

# Target features
TARGET_INIT_COLS = [
    "userId", "pageId", "coldStart", "historySize",
    "numberOfClicksHistory", "timeOnPageHistory",
    "scrollPercentageHistory", "minutesSinceLastVisit", "timeGapDays"
]
TARGET_FINAL_COLS = ["userId", "pageId", "TARGET"]
DEFAULT_TARGET_VALUES = {
    "numberOfClicksHistory": 0, "timeOnPageHistory": 0,
    "scrollPercentageHistory": 0, "minutesSinceLastVisit": 60,
    "historySize": 130, "timeGapDays": 50
}
