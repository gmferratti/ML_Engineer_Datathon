#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from collections import Counter

import glob
import pandas as pd

histories = defaultdict(Counter)

for fpath in glob.glob('./treino/*.csv'):
    df = pd.read_csv(fpath)
    for _, row in df.iterrows():
        user = row['userId']
        hist = row['history']
        hist = (hist.
                replace('\n', ' ').
                replace("'", ' ').
                replace("[", ' ').
                replace("]", ' ').
                strip().split()
        )
        histories[user].update(hist)

df_test = pd.read_csv('./validacao.csv')
k = 10
print('userId', 'acessos_futuros', sep=',')
for user in set(df_test.userId):
    topk = histories[user].most_common(k)
    for item in topk:
        print(user, item[0], sep=',')
