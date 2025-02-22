# NEWS_COLS
COLS_TO_CLEAN = ["body", "title", "caption"]
COLS_TO_DROP = [
    "local",
    "theme",
    "issued",
    "modified",
    "url",
    "urlExtracted",
] + COLS_TO_CLEAN

# USERS_COLS
COLS_TO_EXPLODE = [
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

# MIX_COLS

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
    "coldStart"
]

STATE_COLS = [
    "userId",
    "localState",
    "countLocalStateUser",
    "relLocalState",
    "coldStart"
]

REGION_COLS = [
    "userId",
    "localRegion",
    "countLocalRegionUser",
    "relLocalRegion",
    "coldStart"
]

THEME_MAIN_COLS = [
    "userId",
    "themeMain",
    "countThemeMainUser",
    "relThemeMain",
    "coldStart"
]

THEME_SUB_COLS = [
    "userId",
    "themeSub",
    "countThemeSubUser",
    "relThemeSub",
    "coldStart"
]

GAP_COLS = [
    "userId",
    "pageId",
    "timeGapDays",
    "timeGapHours",
    "timeGapMinutes",
    "timeGapLessThanOneDay"
]