from textblob import TextBlob 
import json
import sys
import time
import numpy as np

def run_sentiment(year):
	with open('gg' + str(year) + '.json') as jsonfile:
		data = json.load(jsonfile)
	tweets = data
	positive = [0,0]
	very_pos = 0
	negative = [0,0]
	very_neg = 0
	counter = 0
	starttime = time.time()
	num_samples = input("Number of evenly-spaced samples (50000 takes ~20 seconds to run, input ALL to run on everything): ")
	if num_samples == 'ALL':
		indices = np.linspace(0,len(tweets) - 1,len(tweets))
	else:
		indices = np.linspace(0,len(tweets) - 1,int(num_samples))

	for i in indices:
		index = int(i)
		tweet = tweets[index]
		text = tweet['text'].lower()
		blob = TextBlob(text)
		sentiment = blob.sentences[0].sentiment.polarity
		if sentiment > 0:
			positive[0] += 1
			positive[1] += sentiment
			if sentiment >= 0.5:
				very_pos += 1
		elif sentiment < 0:
			negative[0] += 1
			negative[1] += sentiment
			if sentiment <= -0.5:
				very_neg += 1

	possible_sentiments = ['ABSOLUTELY HORRENDOUS','VERY POOR','POOR','ALRIGHT','EXPECTED','DECENT','GOOD','VERY GOOD','GREAT','INCREDIBLE','ASTRONOMICALLY BEAUTIFULLY WONDERFUL']
	print("ANALYSIS OF POSITIVITY")
	print("Total number of positive sentiments: " + str(positive[0]))
	print("Number of highly positive sentiments: " + str(very_pos))
	print("Positivity score: " + str(positive[1]))
	print("The average positivity of all positive tweets was " + str(positive[1]/positive[0]))
	print("The percentage of positive tweets that were VERY positive was " + str(very_pos/positive[0] * 100) + "%")
	print('')
	print("ANALYSIS OF NEGATIVITY")
	print("Total number of negative sentiments: " + str(negative[0]))
	print("Number of highly negative sentiments: " + str(very_neg))
	print("Negativity score: " + str(negative[1]))
	print("The average negativity of all negative tweets was " + str(negative[1]/negative[0]))
	print("The percentage of negative tweets that were VERY negative was " + str(very_neg/negative[0] * 100) + "%")
	print('')
	index = int(positive[0]/(positive[0] + negative[0])*10)
	decided_total_sentiment = possible_sentiments[index]
	print("As a whole, the " + year + " Golden Globes were " + decided_total_sentiment)

if __name__ == "__main__":
	result = run_sentiment(sys.argv[1])