import re
import nltk
import numpy as np
import json
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from collections import Counter
from fuzzywuzzy import fuzz

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']


def remove_symbols(a_tweet):
    entity_prefixes = ['@','#']
    words = []
    for word in a_tweet.split():
        word = word.strip()
        if word:
            if word[0].lower() not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

with open('gg2013.json') as jsonfile:
	data = json.load(jsonfile)

ggre = "(golden globes)( )*(party|after-party|afterparty|after party)"
regexp = "party|after-party|afterparty|after party"
locationre = "(party|after-party|afterparty|after party)( )*(at|at the)"
stopregexp = ":|-|goes to|goes|\bfor"

places = []

tweets = data
for i in tweets:
	tweet = remove_symbols(i['text'].lower())
	result = re.search(regexp, tweet)
	post_location = False
	ggflag = False
	if result:
		location_descriptor = ""
		tokenized_tweet = pos_tag(word_tokenize(tweet))
		print("TWEET: " + tweet)
		print(tokenized_tweet)
		location = re.search(locationre, tweet)
		if location:
			post_location = True
		golden_globe_mention = re.search(ggre, tweet)
		if golden_globe_mention:
			ggflag = True

		if post_location:
			rest_of_tweet = tweet[location.span()[1]:]
			print("HEY!!!!")
			print(rest_of_tweet)
			start_adding = False
			for j in range(len(tokenized_tweet)):
				if tokenized_tweet[j][0] == 'at':
					start_adding = True
				elif start_adding:
					location_descriptor += " " + tokenized_tweet[j][0]
					if 'NN' in tokenized_tweet[j][1]:
						break
		else:
			if ggflag:
				for j in range(len(tokenized_tweet)):
					if tokenized_tweet[j][0] == 'golden':
						goldenindex = j
						break
				for j in range(goldenindex):
					index = goldenindex - 1 - j
					location_descriptor = tokenized_tweet[index][0] + " " + location_descriptor
					if 'NN' in tokenized_tweet[index][1]:
						break

			else:
				for j in range(len(tokenized_tweet)):
					if 'party' in tokenized_tweet[j][0] or 'after' in tokenized_tweet[j][0]:
						goldenindex = j
						break
				for j in range(goldenindex):
					index = goldenindex - 1 - j
					location_descriptor = tokenized_tweet[index][0] + " " + location_descriptor
					if 'NN' in tokenized_tweet[index][1]:
						break
		print("")
		if len(location_descriptor) > 0:
			if post_location:
				places.append([location_descriptor,'after'])
			else:
				places.append([location_descriptor,'before'])

for i in places:
	print(i)






