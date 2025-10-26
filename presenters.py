import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import json
import string
import re
from fuzzywuzzy import process, fuzz
from collections import Counter
from spacy.matcher import Matcher

# Precompile regex patterns once at module level
URL_RE = re.compile(r'https?://\S+|www\.\S+')
AMP_RE = re.compile(r'&amp;')
EMOJI_RE = re.compile(r'[\u2600-\u27BF\u1F300-\u1F6FF\u1F900-\u1F9FF]+')
SPACE_RE = re.compile(r'\s+')

DROP_SYMBOLS = {'@', '#'}

def remove_symbols(t):
    # use list comprehension for speed
    words = [
        w for w in t.split()
        if w and w.lower() != 'rt' and w[0] not in DROP_SYMBOLS
    ]
    return ' '.join(words)

def normalize_text(t):
    # apply precompiled regexes
    t = URL_RE.sub('', t)
    t = AMP_RE.sub('&', t)
    t = EMOJI_RE.sub(' ', t)
    t = SPACE_RE.sub(' ', t).strip()
    return t


def get_tweet_data(year):
    path = f'gg{year}.json'
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Combine cleaning steps inline for efficiency
    def clean_text(x):
        t = x.get('text', '')
        t = remove_symbols(t)
        return normalize_text(t)

    texts = [clean_text(x) for x in raw]
    return pd.DataFrame({'text': texts})

_PUNCT_TABLE = str.maketrans('', '', r'''!()-[]{};:'"\,<>./?@#$%^&*_~''')

def removePunctuation(s):
    return s.replace("...", " ").translate(_PUNCT_TABLE)


def index_tweet(tweet, word):
    clean_tweet = removePunctuation(tweet).lower().split()
    clean_word = removePunctuation(word).lower()
    try:
        return clean_tweet.index(clean_word)
    except ValueError:
        return -1

BADBITS = {'rt', 'best', 'award', 'golden', 'globes', 'globe'}

def filter_names(name_lst):
    res = []
    for name in name_lst:
        n = name.strip()
        if not n:
            continue
        n = n.replace('Jr.', '').strip()
        words = n.lower().split()
        if any(b in words for b in BADBITS):
            continue
        if len(words) >= 2:
            res.append(n)
    return res

def compute_mode(names, k=2):
    if not names:
        return []
    counts = Counter(names)
    # Take top k directly
    return [n for n, _ in counts.most_common(k)]



def remove_similar_names(names, cutoff=60):
    res = sorted(set(names))
    keep = []
    seen = set()

    for name in res:
        if any(fuzz.ratio(name, k) >= cutoff for k in keep):
            continue
        keep.append(name)
        seen.add(name)
    return keep




def get_keywords_of_award(award):
    s = award.translate(str.maketrans('', '', string.punctuation))
    toks = [w.lower() for w in word_tokenize(s) if w.lower() not in stopwords.words('english')]
    repl = {
        'television': 'series',
        'animated': 'animat',
        'supporting': 'support'
    }

    drop = {
        'best','award','performance','made','feature','film','role','motion','picture','series','comedy','musical'
    }
    out = []
    for w in toks:
        if w in drop:
            continue
        out.append(repl.get(w, w))
    return " ".join(out)

def get_tweets(keywords, df):
    keys = [k.lower() for k in keywords]
    hits = []
    for t in df['text']:
        tl = t.lower()
        if all(k in tl for k in keys):
            hits.append(t)
    return pd.DataFrame(hits, columns=['text'])

def get_tweets_with_verb(keywords, df):
    keys = [k.lower() for k in keywords]
    hits = []
    for t in df['text']:
        tl = t.lower()
        if any(k in tl for k in keys):
            hits.append(t)
    return pd.DataFrame(hits, columns=['text'])





VERB_PATTERNS = [
    # present
    r'\bpresent(?:s|ed|ing)?\b', r'\bpresenter(?:s)?\b', r'presented by',
    # introduce
    r'\bintroduc(?:e|es|ed|ing)?\b', r'introduced by',
    # announce
    r'\bannounc(?:e|es|ed|ing)?\b', r'announced by',
    # give / gave / giving / hands
    r'\bgiv(?:e|es|en|ing)?\b', r'\bgave\b', r'\bhand(?:s|ed|ing)?\b',
    # read / reads / reading
    r'\bread(?:s|ing)?\b',
    # generic phrasing
    r'presented with', r'on stage with', r'joined by'
]

def get_index(tweet):
    t = tweet.lower()
    idxs = []
    for rp in VERB_PATTERNS:
        m = re.search(rp, t)
        if m:
            idxs.append(m.start())
    return min(idxs) if idxs else -1

def position_of_ppl(tweet, persons):
    t = removePunctuation(tweet).lower()
    if not persons:
        return 0
    if 'best' not in t.split():
        return 1
    first = persons[0].lower().split()[0]
    if first not in t.split():
        return 0
    ppl_idx = t.split().index(first)
    best_idx = t.split().index('best') if 'best' in t.split() else len(t.split())
    return 1 if ppl_idx < best_idx else 0

def get_positions(df, cecil_award=False):
    if cecil_award:
        df['position'] = 1
    else:
        df['position'] = df.apply(lambda r: position_of_ppl(r['text'], r['full names']), axis=1)
    return df[df['position'] == 1]

def get_names_after_verb(df):
    out = []
    for _, row in df.iterrows():
        tweet = removePunctuation(row.text).lower().split()
        ppl = row.filtered
        verb_i = row.verb_index
        if verb_i == -1:
            out.extend(ppl)
            continue
        verb_tok_i = max(0, min(len(tweet) - 1, int(round(verb_i / max(1, len(''.join(tweet)) / max(1, len(tweet)))))))
        for p in ppl:
            base = removePunctuation(p).lower().split()
            if not base:
                continue
            try:
                person_idx = tweet.index(base[0])
            except ValueError:
                continue
            if 'wins' in tweet and tweet.index('wins') - 2 == person_idx:
                continue
            if 'win' in tweet and tweet.index('win') - 2 == person_idx:
                continue
            if 'won' in tweet and tweet.index('won') - 2 == person_idx:
                continue
            if person_idx < verb_tok_i:
                out.append(p.strip())
            else:
                if 'with' in tweet:
                    if tweet.index('with') > verb_tok_i:
                        out.append(p.strip())
    return out






def _build_matcher(nlp):
    m = Matcher(nlp.vocab)
    pat = [{'POS': 'PROPN'}, {'POS': 'PROPN', 'OP': '?'}]
    try:
        m.add('FULL_NAME', [pat])
    except TypeError:
        m.add('FULL_NAME', None, pat)
    return m

def get_person(doc, matcher):
    names = set()
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            names.add(ent.text)
    for _, s, e in matcher(doc):
        names.add(doc[s:e].text)
    return [n for n in names if len(n.split()) >= 2]




def get_presenters(award, data, nlp=None, matcher=None):
    if data.shape[0] == 0:
        return "NA"
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    if matcher is None:
        matcher = _build_matcher(nlp)

    keys = get_keywords_of_award(award).split()
    df = get_tweets(keys, data)
    if df.shape[0] == 0:
        return "NA"

    verb_triggers = [
        'introduc', 'introduce', 'introduced', 'introducing',
        'giv', 'give', 'gives', 'gave', 'giving', 'hand', 'hands', 'handed',
        'present', 'presents', 'presented', 'presenting', 'presenter',
        'read', 'reads', 'reading',
        'announc', 'announce', 'announces', 'announced', 'announcing',
        'presented by', 'announced by', 'introduced by', 'on stage with', 'joined by'
    ]




    df = get_tweets_with_verb(verb_triggers, df)
    if df.shape[0] == 0:
        return "NA"

    df = df.drop_duplicates(subset='text')
    df['full names'] = df['text'].apply(lambda x: get_person(nlp(removePunctuation(x)), matcher))
    df = df[df['full names'].str.len() != 0]
    if df.shape[0] == 0:
        return "NA"

    df['filtered'] = df['full names'].apply(filter_names)
    df = df[df['filtered'].str.len() != 0]
    if df.shape[0] == 0:
        return "NA"

    df['verb_index'] = df['text'].apply(get_index)
    df = get_positions(df, award == 'cecil b. demille award')
    if df.shape[0] == 0:
        return "NA"

    if award == 'cecil b. demille award':
        df = df[~df['text'].str.contains("speech", case=False, na=False)]
        if df.shape[0] == 0:
            return "NA"

    names_after = get_names_after_verb(df)
    if not names_after:
        return "NA"

    presenters = compute_mode(names_after, k=2)
    if not presenters:
        return "NA"
    return ', '.join(remove_similar_names(presenters)).lower()




def run_presenters(year):
    data = get_tweet_data(year)
    AWARDS_NAME = [
        'cecil b. demille award', 'best motion picture - drama',
        'best performance by an actress in a motion picture - drama',
        'best performance by an actor in a motion picture - drama',
        'best motion picture - comedy or musical',
        'best performance by an actress in a motion picture - comedy or musical',
        'best performance by an actor in a motion picture - comedy or musical',
        'best animated feature film', 'best foreign language film',
        'best performance by an actress in a supporting role in a motion picture',
        'best performance by an actor in a supporting role in a motion picture',
        'best director - motion picture', 'best screenplay - motion picture',
        'best original score - motion picture', 'best original song - motion picture',
        'best television series - drama',
        'best performance by an actress in a television series - drama',
        'best performance by an actor in a television series - drama',
        'best television series - comedy or musical',
        'best performance by an actress in a television series - comedy or musical',
        'best performance by an actor in a television series - comedy or musical',
        'best mini-series or motion picture made for television',
        'best performance by an actress in a mini-series or motion picture made for television',
        'best performance by an actor in a mini-series or motion picture made for television',
        'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
        'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television'
    ]
    categories = AWARDS_NAME if year in [2013, '2013'] else AWARDS_NAME
    nlp = spacy.load("en_core_web_sm")
    matcher = _build_matcher(nlp)
    out = {}
    for award in categories:
        out[award] = get_presenters(award, data, nlp=nlp, matcher=matcher)
    return out
