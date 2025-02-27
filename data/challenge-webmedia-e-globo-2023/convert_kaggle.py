#!/usr/bin/env python
# coding: utf-8

import pandas as pd

conjunto = 'validacao'
df = pd.read_csv(f'{conjunto}.csv')

# Faz um explode para a lista virar uma coluna nova
# Trata a string antes
order = (df['history'].
 str.replace('\n', ' ').
 str.replace("'", ' ').
 str.replace("[", ' ').
 str.replace("]", ' ').
 str.strip().
 str.split().
 explode()
)

# Cola o userId do df no explodido
aux = pd.merge(
    df['userId'], order,
    left_index=True, right_index=True
)

# Teste de sanidade (que passa).
# Verifica se os usuários estão aparecendo sempre
# consecutivos
# for userId in set(aux.userId):
#    assert (aux.userId == userId).diff().sum() <= 2


# Itera no df em ordem inversa, coloca a relevancia
# quanto maior o numero, mais relevante. por isso
# em ordem inversa, 1 é pior relevancia.
rel_col = []
prev_user = None
inverse_aux = aux[::-1].copy()
for i in range(len(aux)):
    user = inverse_aux.userId.iloc[i]
    if user != prev_user:
        relevance = 1
    else:
        relevance = relevance + 1
    rel_col.append(relevance)
    prev_user = user
inverse_aux['relevance'] = rel_col
final = inverse_aux[::-1]

final.to_csv(f'{conjunto}_kaggle.csv',
             header=True, index=False)
