import re
import nltk
import numpy as np
import json
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from collections import Counter
from fuzzywuzzy import fuzz

def remove_symbols(a_tweet):
    entity_prefixes = ['@','#']
    words = []
    for word in a_tweet.split():
        word = word.strip()
        if word:
            if word[0].lower() not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def find_special_awards(tweets):
	regexp = "(?<=the)(.*)(?=award)"
	specials = []
	for i in tweets:
		tweet = remove_symbols(i['text'].lower())
		result = re.search(regexp, tweet)
		if result:
			start,end = result.span()
			if end - start > 25 or end - start < 10:
				continue
			specials.append(tweet[start:end])
	specials = list(Counter(specials).most_common(100))
	return specials[0:5]

def run_awards(year):
	with open('gg' + str(year) + '.json') as jsonfile:
		data = json.load(jsonfile)

	regexp = "best"
	stopregexp = ":|-|goes to|goes|\bfor"
	tweets_with_best = []
	whole_tweets_with_best = []

	tweets = data#[80001:100000]
	text = []
	for i in tweets:
		tweet = remove_symbols(i['text'].lower())
		result = re.search(regexp, tweet)
		if result:
			after_best = tweet[result.span()[0]:min(result.span()[1]+50,len(tweet) - 1)]
			stopspot = re.finditer(stopregexp, after_best)
			stopspot = [x for x in stopspot]
			if stopspot:
				stop_point = stopspot[-1]
				after_best = after_best[0:stop_point.span()[0]]
			if len(after_best) > 20:
				tweets_with_best.append(after_best)
			whole_tweets_with_best.append(tweet)

	shorter = list(Counter(tweets_with_best).most_common(100))
	awards = []
	counter = 0
	for i in shorter:
		counter += 1
		need_new = True
		for j in awards:
			if fuzz.token_sort_ratio(j[0],i[0]) > 85 and fuzz.partial_ratio(j[0],i[0]) > 85:
				if len(i[0]) < len(j[0]):
					j[0] = i[0]
				j[1] += i[1]
				need_new = False
		if need_new and len(awards) < 50:
			awards.append([i[0],i[1]])
		elif need_new and len(awards) >= 50:
			awards.sort(key=lambda x: x[1])
			awards[0] = [i[0],1]

	awards.reverse()
	returnawards = []
	special_awards = find_special_awards(tweets)
	ban_words = ['golden','globes','globe']
	for i in special_awards:
		for j in ban_words:
			if j not in i[0]:
				returnawards.append(i[0])

	noiseregex = r"!|@|#|%|&"
	for i in range(30):
		string = awards[i][0]
		result = re.search(noiseregex, string)
		if result:
			continue
		if 'tv' in string:
			string = string.replace('tv','television')
		returnawards.append(string)
		if 'actor' in string:
			string = string.replace('actor','actress')
			returnawards.append(string)
		elif 'actress' in string:
			string = string.replace('actress','actor')
			returnawards.append(string)
	return returnawards

if __name__ == "__main__":
	result = run_awards(2013)







