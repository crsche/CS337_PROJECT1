import re
import nltk
import numpy as np
import json
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import sys

from collections import Counter
from fuzzywuzzy import fuzz
import spacy

def remove_symbols(a_tweet):
    entity_prefixes = ['@','#']
    words = []
    for word in a_tweet.split():
        word = word.strip()
        if word:
            if word[0].lower() not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def noisefilter(tweet):
	ban_list = ['rt','golden','globes']
	for i in ban_list:
		tweet = tweet.replace(i,'')
	return tweet

def sortbypositivity(element):
	return element[1]

def sortbynegativity(element):
	return element[2]

def sortbypolarity(element):
	return element[3]

def run_redcarpet(year):
	nlp = spacy.load("en_core_web_sm")
	with open('gg' + str(year) + '.json') as jsonfile:
		data = json.load(jsonfile)
	worstre = 'worst dressed'
	bestre = 'best dressed'
	wtweets = []
	btweets = []

	tweets = data
	for i in tweets:
		tweet = remove_symbols(i['text'].lower())
		worstresult = re.search(worstre, tweet)
		bestresult = re.search(bestre, tweet)
		if worstresult and bestresult:
			continue
		elif worstresult:
			wtweets.append(noisefilter(tweet))
		elif bestresult:
			btweets.append(noisefilter(tweet))

	dress_score_dict = {}
	#format for values will be [bestcount, worstcount]
	for i in wtweets:
		result = nlp(i)
		pair_flag = False
		person_name = ''
		for token in result:
			if token.ent_type_ == "PERSON" and pair_flag == False:
				pair_flag = True
				person_name = token.text + ' '
			elif token.ent_type_ == "PERSON" and pair_flag == True:
				pair_flag = False
				person_name += token.text
				if not (person_name.replace(' ','').isalpha()):
					continue
				if person_name not in dress_score_dict.keys():
					dress_score_dict[person_name] = [0,1]
				else:
					dress_score_dict[person_name][1] += 1
	for i in btweets:
		result = nlp(i)
		pair_flag = False
		person_name = ''
		for token in result:
			if token.ent_type_ == "PERSON" and pair_flag == False:
				pair_flag = True
				person_name = token.text + ' '
			elif token.ent_type_ == "PERSON" and pair_flag == True:
				pair_flag = False
				person_name += token.text
				if not (person_name.replace(' ','').isalpha()):
					continue
				if person_name not in dress_score_dict.keys():
					dress_score_dict[person_name] = [1,0]
				else:
					dress_score_dict[person_name][0] += 1

	dress_score_list = []
	for i in dress_score_dict.keys():
		key = i
		polarity = abs(dress_score_dict[i][0]/(dress_score_dict[i][0] + dress_score_dict[i][1]) - 0.5)
		positivity = dress_score_dict[i][0]
		negativity = dress_score_dict[i][1]
		dress_score_list.append([key,positivity,negativity,polarity])

	dress_score_list_copy = dress_score_list

	dress_score_list = []
	for i in dress_score_list_copy:
		if not (i[1] < 5 and i[2] < 5):
			dress_score_list.append(i)
	
	dress_score_list.sort(key=sortbypositivity,reverse=True)
	best_votes = dress_score_list[0][1]
	best_dressed = []
	for i in dress_score_list:
		best_dressed.append(i[0])
		if len(best_dressed) >= 5:
			break

	dress_score_list.sort(key=sortbynegativity,reverse=True)
	worst_votes = dress_score_list[0][2]
	worst_dressed = []
	for i in dress_score_list:
		worst_dressed.append(i[0])
		if len(worst_dressed) >= 5:
			break

	dress_score_list.sort(key=sortbypolarity)
	controversial_ratio = [dress_score_list[0][1],dress_score_list[0][2]]
	most_controversial = []
	for i in dress_score_list:
		most_controversial.append(i[0])
		if len(most_controversial) >= 5:
			break

	print("The five best dressed of the " + year + " Golden Globes were:")
	for i in range(5):
		print(str(i + 1) + ". " + best_dressed[i])
	print("The single best dressed red carpeter was " + best_dressed[0] + " with " + str(best_votes) + " votes for best dressed.")
	print('')

	print("The five worst dressed of the " + year + " Golden Globes were:")
	for i in range(5):
		print(str(i + 1) + ". " + worst_dressed[i])
	print("The single worst dressed red carpeter was " + worst_dressed[0] + " with " + str(worst_votes) + " votes for worst dressed.")
	print('')

	print("The five most controversial red carpeters of the " + year + " Golden Globes were:")
	for i in range(5):
		print(str(i + 1) + ". " + most_controversial[i])
	print("The most controversial red carpeter was " + most_controversial[0] + " with " + str(controversial_ratio[0]) + " votes for best dressed and " + str(controversial_ratio[1]) + " votes for worst dressed.")
	print('')


if __name__ == "__main__":
	result = run_redcarpet(sys.argv[1])
