import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import json
import string
import re
from fuzzywuzzy import process
from collections import Counter
from spacy.matcher import Matcher
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def remove_symbols(t):
    bad = {'@', '#'}
    out = []

    for w in t.split():
        w = w.strip()
        if w and w[0].lower() not in bad:
            out.append(w)
    return ' '.join(out)

def get_tweet_data(year):
    p = f'gg{year}.json'
    with open(p, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    texts = [x['text'] for x in raw]
    texts = [remove_symbols(x) for x in texts]
    return pd.DataFrame(texts, columns=['text'])

def get_person_names(texts, nlp):
    counts = {}
    for t in texts:
        name = extract_full_name(nlp(t), nlp)
        if name:
            k = name.lower()
            counts[k] = counts.get(k, 0) + 1
    return counts

def extract_full_name(doc, nlp):
    m = Matcher(nlp.vocab)
    pat = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    try:
        m.add('FULL_NAME', [pat])
    except TypeError:
        m.add('FULL_NAME', None, pat)

    hits = m(doc)
    for _, s, e in hits:
        return doc[s:e].text
    return None



def get_top_percent(d, pct):
    if not d:
        return []
    
    mx = max(d.values())
    th = mx * pct
    return [k for k, v in d.items() if v > th]

def remove_similar_names(names):
    res = sorted(names)
    for n in names:
        sims = process.extract(n, names)
        lead = sims[0][0]

        for cand, score in sims[1:]:
            if cand and lead in res and score >= 60:
                if cand in res:
                    res.remove(cand)
    return res

def get_hosts(df, year):
    nlp = spacy.load("en_core_web_sm")
    df = df.drop_duplicates(subset='text')
    m = df['text'].str.contains("host", na=False)
    host_df = df[m]
    host_df = host_df[~host_df['text'].str.contains("next year", na=False)]


    items = list(host_df['text'])

    people = get_person_names(items, nlp)
    people.pop('golden globes', None)
    people.pop('golden globe', None)

    top = get_top_percent(people, pct=0.2)
    clean = remove_similar_names(top)
    return clean

def run_hosts(year):
    df = get_tweet_data(year)
    return get_hosts(df, year)

