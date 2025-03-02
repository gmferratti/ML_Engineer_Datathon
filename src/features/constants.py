"""
Constantes para colunas e parâmetros do pipeline.
"""

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

NEWS_COLS_TO_DROP = [
    "local",
    "theme",
    "issued",
    "modified",
    "urlExtracted",
    "body",
    "caption",
]

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
    "isWeekend",
]

STATE_COLS = ["userId", "localState", "countLocalStateUser", "relLocalState"]
REGION_COLS = ["userId", "localRegion", "countLocalRegionUser", "relLocalRegion"]
THEME_MAIN_COLS = ["userId", "themeMain", "countThemeMainUser", "relThemeMain"]
THEME_SUB_COLS = ["userId", "themeSub", "countThemeSubUser", "relThemeSub"]

GAP_COLS = [
    "userId",
    "pageId",
    "timeGapDays",
    "timeGapHours",
    "timeGapMinutes",
    "timeGapLessThanOneDay",
]

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

CATEGORY_COLS = ["localState", "localRegion", "themeMain", "themeSub"]

KEY_FEAT_COLS = ["userId", "pageId", "issuedDatetime", "timestampHistoryDatetime"]

SUGGESTED_FEAT_COLS = (
    KEY_FEAT_COLS
    + CATEGORY_COLS
    + [
        "userType",
        "isWeekend",
        "dayPeriod",
        "coldStart",
        "relLocalState",
        "relLocalRegion",
        "relThemeMain",
        "relThemeSub",
    ]
)

TARGET_INIT_COLS = [
    "userId",
    "pageId",
    "coldStart",
    "historySize",
    "numberOfClicksHistory",
    "timeOnPageHistory",
    "scrollPercentageHistory",
    "minutesSinceLastVisit",
    "timeGapDays",
]
TARGET_FINAL_COLS = ["userId", "pageId", "TARGET"]
DEFAULT_TARGET_VALUES = {
    "numberOfClicksHistory": 0,
    "timeOnPageHistory": 0,
    "scrollPercentageHistory": 0,
    "minutesSinceLastVisit": 60,
    "historySize": 130,
    "timeGapDays": 50,
}
