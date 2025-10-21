import json
import re
import string
import spacy
from spacy.matcher import Matcher
from fuzzywuzzy import fuzz
import pandas as pd
import math

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
    tweets = [tweet['text'] for tweet in tweets]
    # tweets = [tweet['text'] for tweet in tweets if 'best' in tweet['text'].lower()]
    
    # remove @,# tokens
    tweets = [remove_symbols(tweet) for tweet in tweets]
    
    return tweets

def remove_symbols(a_tweet):
    entity_prefixes = ['@','#','(',')']
    words = []
    for word in a_tweet.split():
        word = word.strip()
        if word and word.lower() != 'rt':
            if word[0].lower() not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def tweets_contain(category_name, tweets):
    stop_words = set(stopwords.words('english'))
    stop_words.add('-')
    stop_words.add('performance')
    stop_words.add('comedy')
    stop_words.add('television')

    tokens = nltk.word_tokenize(category_name)
    no_stop_words = [w for w in tokens if not w in stop_words] 
    
    regex = r''
    for token in no_stop_words:
        regex += token
        regex += '.*?'
        
    # regex = category_name
    r = re.compile(regex, re.IGNORECASE)
    filtered_list = list(filter(r.search, tweets))
    
    if 'act' not in category_name:
        filtered_list = [x for x in filtered_list if 'act' not in x.lower()]
    
    return filtered_list

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
    tokens += ['motion picture', 'golden', 'globes', 'television series', 'tv series', 'mini-', 'rt', '⁰', '-series']
    
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words] 
    
    for token in tokens:
        for key in pronouns_dict:
            if token in key:
                pronouns_dict[key] = 0
    return pronouns_dict

def get_category_nominees(category, tweets, nlp):
    # filter for tweets which contain the category name regex
    category_tweets = tweets_contain(category, tweets)

    # create a dictionary of person names/film titles
    if 'actor' in category or 'actress' in category:
        pronouns_dictionary = get_person_names(category_tweets, nlp)
        # pronouns_dictionary = get_NNP(category_tweets)
    else:
        # pronouns_dictionary = get_person_names(category_tweets, nlp)
        pronouns_dictionary = get_NNP(category_tweets)

    # remove tokens contained in the category name
    filtered_category_tweets = remove_category_tokens(category, pronouns_dictionary)

    # get pronouns with >some_percentile frequency 
    category_nominees = get_top_percent(filtered_category_tweets, percentile=0.85)
    
    return category_nominees

def run_winners(year):
    tweets = get_tweet_data(year)
    
    seperator = ' '
    nominees = {}
    OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
    OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

    if year in [2013, 2015, '2013', '2015']:
        categories = OFFICIAL_AWARDS_1315
    else:
        categories = OFFICIAL_AWARDS_1819
    
    nlp = spacy.load('en_core_web_sm')
    for category_name in categories:
        category_nominees_list = get_category_nominees(category_name, tweets, nlp)
        category_nominees_string = seperator.join(category_nominees_list)
        # if not any(x in category_name for x in ['act', 'director', 'cecil']):
        #     category_nominees_string = find_movie_title(category_nominees_string, year, category_name, nlp)
        nominees[category_name] = category_nominees_string
        # print(category_name, ':', category_nominees_string)
    return nominees


def get_person_names(list_of_tweets, nlp):
    '''
    Input: a list of strings
    Returns: a dictionary where keys = actor/actress names, values = number of times the key is references in the list of tweets
    '''
    names_dictionary = {}

    matcher = Matcher(nlp.vocab)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('FULL_NAME', None, pattern)

    counter = 0
    l = len(list_of_tweets)
    m = math.floor(l / 1000) + 1

    for tweet in list_of_tweets:
        counter += 1

        if counter % m != 0:
            continue
        
        nlp_doc = nlp(tweet)
        matches = matcher(nlp_doc)
        names = None
        for match_id, start, end in matches:
            span = nlp_doc[start:end]
            names = [span.text]

            name = names[0].lower()
            if name in names_dictionary:
                names_dictionary[name] += 1
            else:
                names_dictionary[name] = 1
            break

    return names_dictionary

def get_person(tweet):
    words = [(ent.text, ent.label_) for ent in tweet.ents]
    return ([token[0] for token in list(filter(lambda x: "PERSON" in x, words))])