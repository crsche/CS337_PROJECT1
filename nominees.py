import json
import re
import string
import spacy
from spacy.matcher import Matcher
from fuzzywuzzy import fuzz
import pandas as pd
from collections import Counter 

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Run the 1st time
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
# install python-Levenshtein

def get_tweet_data(year):
    # load tweet data
    file_string = 'gg' + str(year) + '.json'
    tweets = {}
    with open(file_string, 'r') as f:
        tweets = json.load(f)
    
    # extract 'text' field from tweets
    # tweets = [tweet['text'] for tweet in tweets]
    tweets = [tweet['text'] for tweet in tweets if ' best ' in tweet['text'].lower()]
    
    # remove @,# tokens
    tweets = [remove_symbols(tweet) for tweet in tweets]
    
    return tweets

def remove_symbols(a_tweet):
#     entity_prefixes = ['@','#','(',')']
    entity_prefixes = ['#','(',')']
    words = []
    for word in a_tweet.split():
        word = word.strip()
        if word and word.lower() != 'rt':
            if word[0].lower() not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def get_NNP(tweets_list):
    NNP_dict = {}
    for tweet in tweets_list:
        tokens = nltk.word_tokenize(tweet)
        tags = nltk.pos_tag(tokens)
        for word in tags:
            if word[1] in ['NNP']:
                name = word[0].lower()
                if name in NNP_dict:
                    NNP_dict[name] += 1
                else:
                    NNP_dict[name] = 1
    return NNP_dict

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

def remove_category_tokens(category, pronouns_dict):
    tokens = nltk.word_tokenize(category)
    tokens += ['motion picture', 'golden', 'globe', 'television series', 'tv series', 'mini-', 'rt','⁰']
    
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words] 
    
    deletes = []
    for token in tokens:
        for key in pronouns_dict:
            if token in key:
                deletes.append(key)
    deletes = list(set(deletes)) 
    for key in deletes:
        del pronouns_dict[key]
    return pronouns_dict



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
    names = []
    for match_id, start, end in matches:
        span = nlp_doc[start:end]
        names.append(span.text)
    return names
    
def extract_one_name(nlp_doc, nlp):
    matcher = Matcher(nlp.vocab)
    pattern = [{'POS': 'PROPN'}]
    matcher.add('FULL_NAME', None, pattern)
    matches = matcher(nlp_doc)
    names = []
    for match_id, start, end in matches:
        span = nlp_doc[start:end]
        names.append(span.text)
    return names
    
def get_person(tweet):
    words = [(ent.text, ent.label_) for ent in tweet.ents]
    return ([token[0] for token in list(filter(lambda x: "PERSON" in x, words))])

def test_trans(s):
    table = str.maketrans('', '', string.punctuation)
    return s.translate(table)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    stop_words.append('performance')
    stop_words.append('television')
    stop_words.append('motion')
    stop_words.append('picture')
    stop_words.append('series')
    stopwords_dict = Counter(stop_words)

    text = ' '.join([word for word in text.split() if word not in stopwords_dict])
    return text


def run_nominees(year):
    tweets = get_tweet_data(year)

    OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
    OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']


    if year in [2013, 2015, '2013', '2015']:
        categories = OFFICIAL_AWARDS_1315
    else:
        categories = OFFICIAL_AWARDS_1819

    d = {}
    for a in categories:
        d[a] = {}

    keywords_lst = [['win', 'hope', ' best '], ['nominated for', ' best '], ['should have won', ' best '], ['didn\'t win', ' best '], ['nomin', 'tv', 'mini']]
    # ['nomin', ' best ', 'tv'], ['nomin', ' best act']
    nlp = spacy.load('en_core_web_sm')
    for keywords in keywords_lst:

        count = 0
        trys = []
        for t in tweets:
            if all(x in t.lower() for x in keywords) and 'dress' not in t.lower():
                trys.append(t)
                count += 1

        for t1 in trys:
            t1 = test_trans(t1)
            names = extract_full_name(nlp(t1), nlp)
            if names == []:
                names = extract_one_name(nlp(t1), nlp)
                if names == []:
                    continue

            t2 = t1.lower().split()

            award = []
            start = t2.index('best')
            end_words = ['award', 'drama', 'comedy', 'musical', 'film', 'director', 'series']
            for i in range(start, len(t2)):
                if t2[i] in end_words:
                    award.append(t2[i])
                    break
                if i - start >= 6:
                    break
                award.append(t2[i])

            s = ''
            for x in award:
                s = s + x + ' '
            award = s

            best_score = 0
            best_category = None
            for a in categories:
                search_award = remove_stopwords(a)
                if 'television' in a:
                    search_award += ' tv'

                score = fuzz.partial_ratio(search_award, award)
                if score > best_score:
                    best_score = score
                    best_category = a

            if best_category == None:
                continue

            if names:
                for name in names:
                    if name.lower() in d[best_category]:
                        d[best_category][name.lower()] += 1
                    else:
                        d[best_category][name.lower()] = 1

    for key in d:
        d[key] = remove_category_tokens(key, d[key])

    nominees_dict = {}
    for key in d:
        k = Counter(d[key]) 
        high = k.most_common(4) 
        nominees_dict[key] = [x[0] for x in high]
        while len(nominees_dict[key]) < 4:
            nominees_dict[key].append('L')
    return nominees_dict

