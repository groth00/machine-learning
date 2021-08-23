#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', size = 7)          # controls default text sizes
plt.rc('axes', titlesize = 7)     # fontsize of the axes title
plt.rc('axes', labelsize = 7)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = 7)    # fontsize of the tick labels
plt.rc('ytick', labelsize = 7)    # fontsize of the tick labels
plt.rcParams['figure.dpi'] = 300
sns.set_palette(palette = 'winter', n_colors = 10)

# NeurIPS research papers from 1987-2019 and their authors & institutions
papers = pd.read_csv('nips/papers.csv')
authors = pd.read_csv('nips/authors.csv')

############################ Missing values

print(f'Missing values in authors:\n{authors.isna().sum()}\n')
print(f'Missing values in papers:\n{papers.isna().sum()}\n')

# set placeholder for missing names, 
# drop missing institutions for plotting purposes
authors_filled = authors.fillna(
    {'first_name': 'unknown_first', 
     'last_name': 'unknown_last'}, axis = 0)\
    .dropna(axis = 0)\
    .reset_index(drop = True)
    
# concatenate first and last name
authors_filled['full_name'] = authors_filled['first_name'] \
    + ' ' + authors_filled['last_name']
    
# set placeholder for empty abstracts
# drop rows with no research paper
papers_filled = papers.fillna({'abstract': 'empty abstract'})\
    .dropna(subset = ['full_text'], axis = 0)

# check that all missing values are resolved
print(f'Missing values in authors_filled:\n{authors_filled.isna().sum()}\n')
print(f'Missing values in papers_filled:\n{papers_filled.isna().sum()}\n')


############################ Visuals

# sns barh 20 institutions with the most papers
papers_institution = pd.DataFrame(
    authors_filled.groupby('institution', as_index = False)['source_id']\
        .count()\
        .sort_values('source_id', ascending = False)[:20]
    )
papers_institution.columns = ['institution', 'num_papers']
axesSubplot = sns.barplot(data = papers_institution, 
                 x = 'num_papers', y = 'institution', palette = 'winter')
axesSubplot.set_ylabel('')
axesSubplot.set_title('NeurIPS Papers per Institution')
# bbox_inches param ensures the text doesn't get cutoff when saving
axesSubplot.figure.savefig('-NeurIPS-InstitutionFreq', dpi = 300,
                           bbox_inches = 'tight')
axesSubplot.clear()

# sns barh 10 authors with the most papers
papers_authors = pd.DataFrame(authors_filled.groupby(
    ['full_name', 'institution'], as_index = False)['source_id']\
        .count().sort_values('source_id', ascending = False)[:10]
    )
papers_authors.columns = ['author', 'institution', 'num_papers']

ax = sns.barplot(data = papers_authors, x = 'num_papers', y = 'author', 
                 hue = 'institution', dodge = False)
ax.set_ylabel('')
ax.set_title('Prolific Authors')
ax.legend(loc = 'lower right', fontsize = 'small', bbox_to_anchor = (1, 0))
ax.figure.savefig('-NeurIPS-AuthorFreq', dpi = 300, bbox_inches = 'tight')
ax.clear()

# sns barplot number of papers per year
papers_per_year = pd.DataFrame(
    papers_filled.groupby(['year'], as_index = False)['source_id']\
        .count())

ax = sns.barplot(data = papers_per_year, x = 'year', y = 'source_id',
                   palette = 'winter')
ax.set_ylabel('')
ax.invert_xaxis()
ax.tick_params(axis = 'x', rotation = 45)
ax.set_title('Papers per Year')
ax.figure.savefig('-NeurIPS-PapersPerYear', dpi = 300, bbox_inches = 'tight')













