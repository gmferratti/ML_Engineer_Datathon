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
    # USED LATER
    "coldStart",
    'userType', 
    'historySize', 
    'numberOfClicksHistory', 
    'timeOnPageHistory', 
    'scrollPercentageHistory', 
    'pageVisitsCountHistory', 
    'minutesSinceLastVisit'
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
    "timeGapDays",
    "timeGapHours",
    "timeGapMinutes",
    "timeGapLessThanOneDay"
]

FINAL_FEAT_MIX_COLS = [
    "userId",
    "pageId",
    "userType",
    "historySize",
    "issuedDatetime",
    "timestampHistoryDatetime",
    "numberOfClicksHistory",
    "timeOnPageHistory",
    "coldStart",
    "scrollPercentageHistory",
    "pageVisitsCountHistory",
    "minutesSinceLastVisit",
    "localState",
    "localRegion",
    "themeMain",
    "themeSub",
]
