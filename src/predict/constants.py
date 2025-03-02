# src/predict/constants.py

EXPECTED_COLUMNS = [
    'isWeekend',
    'relLocalState', 'relLocalRegion', 'relThemeMain', 'relThemeSub',
    'userTypeFreq', 'dayPeriodFreq', 'localStateFreq', 'localRegionFreq',
    'themeMainFreq', 'themeSubFreq'
]

CLIENT_FEATURES_COLUMNS = ['isWeekend', 'userTypeFreq', 'dayPeriodFreq']

NEWS_FEATURES_COLUMNS = [
    'relLocalState', 'relLocalRegion', 'relThemeMain', 'relThemeSub',
    'localStateFreq', 'localRegionFreq', 'themeMainFreq', 'themeSubFreq'
]

METADATA_COLS = ["pageId", "url", "title", "issuedDate", "issuedTime"]