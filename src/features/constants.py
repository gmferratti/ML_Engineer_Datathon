# NEWS_PATH

news_template_path = "data/challenge-webmedia-e-globo-2023/itens/itens/itens-parte{}.csv"
news_num_csv_files = 3

# NEWS_COLS


# NEWS_DICTS


# USERS_PATH

users_template_path = "data/challenge-webmedia-e-globo-2023/files/treino/treino_parte{}.csv"
users_num_csv_files = 6
cold_start_threshold = 5

# USERS_COLS

users_cols_to_explode = [
    'history', 
    'timestampHistory', 
    'numberOfClicksHistory',
    'timeOnPageHistory',
    'scrollPercentageHistory',
    'pageVisitsCountHistory'
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


# USERS_DICTS


# STATES_MAPPING