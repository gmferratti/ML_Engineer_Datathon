# NEWS_COLS
cols_to_clean = ["body", "title", "caption"]
cols_to_drop = [
    "local",
    "theme",
    "issued",
    "modified",
    "url",
    "urlExtracted",
] + cols_to_clean

# USERS_COLS
users_cols_to_explode = [
    "history",
    "timestampHistory",
    "numberOfClicksHistory",
    "timeOnPageHistory",
    "scrollPercentageHistory",
    "pageVisitsCountHistory",
]

users_dtypes = {
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
