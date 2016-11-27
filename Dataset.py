# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 01:21:24 2016

@author: ADMIN
"""
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
u_cols = ['user_id', 'sex', 'age','occupation', 'zip_code']

users = pd.read_csv('users.dat', sep=',', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ratings.dat', sep=',', names=r_cols,
                      encoding='latin-1')

m_cols = ['movie_id', 'title', 'genre']
movies = pd.read_csv('movies.dat', sep=',', names=m_cols,
                     encoding='latin-1')

movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(ratings)
distances, indices = nbrs.kneighbors(ratings)

#most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]
#
#movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})
#movie_stats.head()
#
#atleast_100 = movie_stats['rating']['size'] >= 2000
#movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]
#
##users.sex.plot.hist(bins=30)
##plt.title("Distribution of users' ages")
#plt.ylabel('count of users')
#plt.xlabel('sex');



#labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
#lens['age_group'] = pd.cut(lens.sex, range(0, 81, 10), right=False, labels=labels)
#lens[['age', 'age_group']].drop_duplicates()[:10]

#lens.groupby('age_group').agg({'rating': [np.size, np.mean]})

