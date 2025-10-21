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

# remove symbols from tweets
def remove_symbols(a_tweet):
    entity_prefixes = ['@','#']
    words = []
    for word in a_tweet.split():
        word = word.strip()
        if word:
            if word[0].lower() not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

# load tweet data
def get_tweet_data(year):
    # if (year == 2020):
    #     return pd.read_json('gg2020.json', lines = True)
    file_string = 'gg' + str(year) + '.json'
    tweets = {}
    with open(file_string, 'r') as f:
        tweets = json.load(f)
    tweets = [tweet['text'] for tweet in tweets] # extract 'text' field from tweets
    # remove @... tokens
    tweets = [remove_symbols(tweet) for tweet in tweets]
    df = pd.DataFrame(tweets, columns = ['text'])
    return df

def get_person_names(list_of_tweets, nlp):
    '''
    Input: a list of strings
    Returns: a dictionary where keys = actor/actress names, values = number of times the key is references in the list of tweets
    '''
    names_dictionary = {}
    for tweet in list_of_tweets:
        name = extract_full_name(nlp(tweet), nlp)
        if name:
            name = name.lower()
            if name in names_dictionary:
                names_dictionary[name] += 1
            else:
                names_dictionary[name] = 1
    return names_dictionary


def extract_full_name(nlp_doc, nlp):
    matcher = Matcher(nlp.vocab)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('FULL_NAME', None, pattern)
    matches = matcher(nlp_doc)
    for match_id, start, end in matches:
        span = nlp_doc[start:end]
        return span.text

def get_top_percent(dictionary, percentile):
    if dictionary == {}:
        return []
    max_val = max(dictionary.values())
    threshold = max_val * percentile
    result = []
    for key in dictionary:
        if dictionary[key] > threshold:
            result.append(key)
    return result

def remove_similar_names(names):
    result = sorted(names)
    for name in names:
        similarities = process.extract(name, names)
        main_person = similarities[0][0]
        for person in similarities[1:]:
            if person[0] and main_person in result:
                if person[1] >= 60:
                    result.remove(person[0])
    return result

# return string name of host(s)
def get_hosts(data, year):

    nlp = spacy.load("en_core_web_sm")

    data = data.drop_duplicates(subset = "text")

    host_data = data[data['text'].str.contains("host")] # get tweets with 'host' in text
    host_df = host_data[host_data["text"].str.contains("next year") == False] # remove tweets with 'next year'

    lst = list(host_df['text'])
    ppl = get_person_names(lst, nlp) # get names of people mentioned
    ppl.pop('golden globes', None)
    ppl.pop('golden globe', None)
    category_nominees = get_top_percent(ppl, percentile=0.20) # get most mentioned people

    names = remove_similar_names(category_nominees) # remove similar names

    return names

# get hosts for specific year
def run_hosts(year):
    data = get_tweet_data(year)
    hosts = get_hosts(data, year)
    return hosts
