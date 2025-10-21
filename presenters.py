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
from fuzzywuzzy import fuzz
from collections import Counter

"""
------------------ functions to load and prep data ------------------
"""
# remove unimportant symbols from tweet data
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
    file_string = 'gg' + str(year) + '.json'
    tweets = {}
    with open(file_string, 'r') as f:
        tweets = json.load(f)
    tweets = [tweet['text'] for tweet in tweets] # extract 'text' field from tweets
    # remove @... tokens
    tweets = [remove_symbols(tweet) for tweet in tweets]
    df = pd.DataFrame(tweets, columns = ['text'])
    return df


"""
-------------------- helper functions to get presenters -----------------
"""


"""
return list of 'PERSON' in tweet"
"""
def get_person(tweet):
    words = [(ent.text, ent.label_) for ent in tweet.ents]
    sub_toks = [tok for tok in tweet if (tok.dep_ == "nsubj") ]
    lst = [token[0] for token in list(filter(lambda x: "PERSON" in x, words))]
    if 'Drama' in lst:
        lst.remove('Drama')
    if 'Jr.' in lst:
        lst.remove('Jr.')
    result = []
    for name in lst:
        if 'Jr.' in name:
            name.strip('Jr.')
        if (len(name.split()) >= 2):
            result.append(name)
    return result

"""
get tweets that have ALL keywords
"""
def get_tweets(keywords, df):
    result = []
    for t in list(df['text']):
        if all(x in t.lower() for x in keywords):
            result.append(t)
    df = pd.DataFrame(result, columns = ['text'])
    return df

"""
get keywords of award
"""
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

#     regex = category_name
    r = re.compile(regex, re.IGNORECASE)
    filtered_list = list(filter(r.search, tweets))

    return filtered_list

"""
remove punctuation
"""
def removePunctuation(string):
    string = string.replace("...", " ")
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string

"""
get index of word in tweet
"""
def index_tweet(tweet, word):
    tweet = removePunctuation(tweet)
    word = removePunctuation(word)
    if word in tweet.lower().split():
        idx = tweet.lower().split().index(word)
        return idx
    else:
        return -1

"""
return 1 if person mentioned before award
"""
def position_of_ppl(tweet, person):
    tweet = removePunctuation(tweet).lower()
    if ('best' in tweet):
        ppl = person[0].lower().split()[0]
        if ppl in tweet.lower().split():
            if 'best' in tweet.lower().split():
                ppl_idx = tweet.lower().split().index(ppl)

                best_idx = tweet.lower().split().index('best')

                if (ppl_idx < best_idx):
                    return 1
                else:
                    return 0
            else:
                return 1
        else:
            return 0
    else:
        return 1

"""
determine position of name relative to verb
"""
# True if looking at cecil award
def get_positions(df, cecil_award = False):
    if (cecil_award):
        df['position'] = 1#df.apply(lambda row: cecil_position(row['text'], row['full names']), axis = 1)
    else:
        df['position'] = df.apply(lambda row: position_of_ppl(row['text'], row['full names']), axis = 1)
    return df[df['position'] == 1]

"""
get keywords of award
"""
def get_keywords_of_award(award):
    a = award
    awards_lst = award.split()

    stopWords = set(stopwords.words('english'))
    # remove punctuation marks
    award = award.translate(str.maketrans('', '', string.punctuation))
    #remove stopwords
    award = word_tokenize(award)
    award = [word for word in award if word not in stopwords.words('english')]
    award = [word.lower() for word in award]

    if 'best' in award: award.remove('best')
    if 'award' in award: award.remove('award')
    if 'performance' in award: award.remove('performance')
    if 'made' in award: award.remove('made') #
    #if 'series' in award: award.remove('series')
    if 'television' in award:
        award.remove('television') #
        award.append('series')
    if 'comedy' in award: award.remove('comedy') #
    if 'musical' in award:
        if 'series' in award:
            award.remove('series')
    if 'feature' in award: award.remove('feature')
    if 'film' in award: award.remove('film')
    if 'role' in award: award.remove('role')
    if 'animated' in award:
        award.remove('animated')
        award.append('animat')
    if 'director' in award:
        if 'motion' in award:
            if 'picture' in award:
                award.remove('motion')
                award.remove('picture')

    if 'mini-series' in award:
        award.remove('mini-series')
        award.append('mini')
        if 'motion' in award:
            if 'picture' in award:
                    award.remove('motion')
                    award.remove('picture')
    if 'miniseries' in award:
        award.remove('miniseries')
        award.append('mini')
        if 'motion' in award:
            if 'picture' in award:
                    award.remove('motion')
                    award.remove('picture')
    if 'series' in award: award.remove('series')
    if 'supporting' in award:
        award.remove('supporting')
        award.append('support')

    # join words
    award = " ".join(award)

    return award


"""
get indices of key verbs in tweet
"""
def get_index(tweet):
    string = tweet.lower().split()
    if 'read' in string:
        return index_tweet(tweet, 'read')
    if 'introduc' in string:
        return index_tweet(tweet, 'introduc')
    if 'introduce' in string:
        return index_tweet(tweet, 'introduce')
    if 'introduced' in string:
        return index_tweet(tweet, 'introduced')
    if 'introduces' in string:
        return index_tweet(tweet, 'introduces')
    if 'reads' in string:
        return index_tweet(tweet, 'reads')
    if 'present' in string:
        return index_tweet(tweet, 'present')
    if 'presents' in string:
        return index_tweet(tweet, 'presents')
    if 'presented' in string:
        return index_tweet(tweet, 'presented')
    if 'presenting' in string:
        return index_tweet(tweet, 'presenting')
    if 'introducing' in string:
        return index_tweet(tweet, 'introducing')
    if 'reading' in string:
        return index_tweet(tweet, 'reading')
    if 'gave' in string:
        return index_tweet(tweet, 'gave')
    if 'gives' in string:
        return index_tweet(tweet, 'gives')
    if 'award' in string:
        return index_tweet(tweet, 'award')
    if 'give' in string:
        return index_tweet(tweet, 'give')
    if 'giving' in string:
        return index_tweet(tweet, 'giving')
    if 'announce' in string:
        return index_tweet(tweet, 'announce')
    if 'announces' in string:
        return index_tweet(tweet, 'announces')
    if 'announced' in string:
        return index_tweet(tweet, 'announced')

"""
get names based on position to key verb
"""
def get_names_after_verb(df):
    result = []
    for index, row in df.iterrows():
        tweet = removePunctuation(row.text)
        string = tweet.lower().split()
        person_lst = row.filtered
        idx = row.verb_index
        if idx:
            for p in person_lst:
                p = removePunctuation(p)
                ppl = p.lower().split()
                if len(ppl) > 0:
                    if ppl[0] in string:
                        person_idx = string.index(ppl[0])
                        if 'wins' in string:
                            wins_idx = string.index('wins')
                            if wins_idx - 2 == person_idx:
                                continue
                        if 'win' in string:
                            win_idx = string.index('win')
                            if win_idx - 2 == person_idx:
                                continue
                        if 'won' in string:
                            won_idx = string.index('won')
                            if won_idx - 2 == person_idx:
                                continue

                        #if 'win' and 'won' not in string.index(person_idx + 1):

                        if (person_idx > -1):
                            if (person_idx < idx):
                                person = p.strip()
                                result.append(person)
                            else:
                                if ('with' in string):
                                    with_idx = string.index('with')
                                    if (with_idx > idx):
                                        person = p.strip()
                                        result.append(person)
            return result
        else:
            return person_lst

"""
filter names baed on punctuation and award words
"""
def filter_names(name_lst):
    result = []
    for name in name_lst:
        if 'RT' in name:
            new_name = name.strip('RT')
        else:
            new_name = name

        if 'Jr.' in new_name:
            new_name = new_name.strip('Jr.')

        n = new_name.lower()
        if 'best' not in n:
            result.append(new_name)
    return result

"""
get tweets with ANY key verb
"""
def get_tweets_with_verb(keywords, df):
    result = []
    for t in list(df['text']):
        if any(x in t.lower() for x in keywords):
            result.append(t)
    df = pd.DataFrame(result, columns = ['text'])
    return df

"""
get mode (top 2) of list
"""
def compute_mode(names):
    result = []
    counts = Counter(names)
    if counts:
        maxcount = max(counts.values())
        for person in counts.items():
            if len(result) >= 2:
                break
            name = person[0]
            count = person[1]
            if count == maxcount:
                result.append(name)
    return result

"""
remove names from list that are similar to each other
"""
def remove_similar_names(names):
    result = sorted(names)
    for name in names:
        similarities = process.extract(name, names)
        main_person = similarities[0][0]
        for person in similarities[1:]:
            if person[0] and main_person in result:
                if person[1] >= 60:
                    if person[0] in result:
                        result.remove(person[0])
    return result

"""
get presenters for specific award
"""
def get_presenters(award, data):

    if (data.shape[0] == 0):
        return "NA"

    # get keywords
    keywords = list(get_keywords_of_award(award).split())

    # get tweets with keywords
    df = get_tweets(keywords, data)

    # get presenting tweets
    tweet_keywords = ['introduc', 'giv', 'gave', 'present', 'read', 'announc']
    df = get_tweets_with_verb(tweet_keywords, df)

    if (df.shape[0] == 0):
        return "NA"

    df = df.drop_duplicates(subset = "text")

    # get list of people in tweets
    nlp = spacy.load("en_core_web_sm")
    df['full names'] = df['text'].apply(lambda x: get_person(nlp(removePunctuation(x))))
    df = df[df['full names'].str.len() != 0] # remove rows with no 'PERSON'

    if (df.shape[0] == 0):
        return "NA"

    df['filtered'] = df['full names'].apply(lambda x: filter_names(x)) #filter names
    df = df[df['filtered'].str.len() != 0]

    if (df.shape[0] == 0):
        return "NA"

    #get indices of verbs
    df['verb_index'] = df['text'].apply(lambda tweet: get_index(tweet))

    # get position of name
    if (award == 'cecil b. demille award'):
        df = get_positions(df, True)
        if (df.shape[0] == 0):
            return "NA"
        df = df[df["text"].str.contains("speech") == False]
        if (df.shape[0] == 0):
            return "NA"
        names_after_verbs = get_names_after_verb(df)
    else:
        df = get_positions(df)
        if (df.shape[0] == 0):
            return "NA"
        names_after_verbs = get_names_after_verb(df)

    if (df.shape[0] == 0):
        return "NA"

    # get person who is talked about most of time
    presenters = compute_mode(names_after_verbs)#get_mode(df['name after verb'])
    if (df.shape[0] == 0):
        return "NA"
    result = ', '.join(remove_similar_names(presenters))

    return result.lower()




"""
---------------------- get presenters for specific year ---------------------
"""

def run_presenters(year):

    data = get_tweet_data(year)

    presenters = {}

    OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']

    OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

    if year in [2013, 2015, '2013', '2015']:
        categories = OFFICIAL_AWARDS_1315
    else:
        categories = OFFICIAL_AWARDS_1819

    for award in categories:
        presenter_lst = get_presenters(award, data)
        presenters[award] = presenter_lst

    return presenters
