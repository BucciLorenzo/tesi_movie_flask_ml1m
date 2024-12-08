#!/usr/bin/env python
# coding: utf-8

# In[7]:


import csv


# In[8]:


def read_movie_tsv(file_path):
    """
    Legge un file TSV e restituisce un dizionario con ID e titoli dei film.

    Args:
        file_path (str): Il percorso del file TSV.

    Returns:
        dict: Un dizionario con gli ID dei film come chiavi e i titoli come valori.
    """
    movies = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            movie_id = row.get("id")
            title = row.get("Title")  # Modifica per corrispondere a 'Title' con la T maiuscola
            if movie_id and title:
                movies[movie_id] = title
    return movies

