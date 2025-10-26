## CS 337 Project 1
## Group 7
## Members: Weijia Wang / Shuyang Yu / Conar Scheidt

### Introduction: 

In Project 1, we implemented a full NLP-based system to extract structured information about the Golden Globes directly from Twitter data. Each main task (hosts, awards, presenters, nominees, winners, red carpet, and parties) is contained in its own Python file and is coordinated through gg_api.py. The program uses a combination of rule-based text extraction, fuzzy matching, and modern NLP tools such as spaCy and VADER sentiment analysis.

When you run the autograder, five human-readable .txt files are generated with extracted answers and evaluation metrics.

We run this project in Python 3.11.9 

### Functions discriptions: 

### get_hosts.py

For hosts, we filter tweets for the keyword "host", removing irrelevant phrases like "next year". Using spaCy’s named entity recognition, we extract all person names mentioned near hosting-related verbs. The top two most frequently mentioned names are returned as the show hosts.

### get_awards.py

For awards, we look for tweets containing "best", "award", or similar patterns. We extract award phrases following “best” using regex and merge near-duplicates with fuzzy matching (fuzz.partial_ratio). Hashtags such as #BestMotionPicture are also normalized to readable award names (e.g., “best motion picture”). We then clean up text, expand abbreviations (e.g., “TV” → “television”), and return a ranked list of unique awards.

### get_winners.py

For winners, we filter tweets for phrases that include the award name and winning-related verbs like “wins”, “won”, “goes to”. Then we extract person names from these tweets using spaCy and count how often each name is mentioned. The most mentioned name per award is reported as the winner.

### get_nominees.py

For nominees, we filter tweets mentioning an award with nomination-related phrases like “nominated for”, “should have won”, “didn’t win”. We extract proper nouns and names using POS tagging and NER, stop at words like “award”, “film”, or “series”, and count frequency. The top four names per award are returned as the nominees.

### get_presenters.py

For presenters, we focus on verbs that indicate presenting behavior — “presented”, “introduce”, “gave”, “announced”, etc. We identify tweets where a PERSON appears before one of these verbs (since winners tend to appear after). We then merge similar names (e.g., “Tina Fey” vs “Fey”) with fuzzy matching and return the two most frequent names per award.

### redcarpet.py

For the red carpet segment, we identify the best dressed, worst dressed, and most controversial celebrities. Tweets are divided into “best dressed” and “worst dressed” groups. We managet to extract person names using spaCy NER, count how many times each is praised or criticized, and compute a polarity score. he top five best dressed, top five worst dressed, and five most controversial (equal mix of positive/negative) are printed with their counts.

### parties.py

For party analysis, we detect which after-parties were discussed most and how people felt about them. We use spaCy to extract phrases containing “party” or “after-party” and normalize hashtags like #NetflixAfterParty. We group tweets by party name using fuzzy matching, then apply VADER sentiment analysis to compute positive, negative, and neutral sentiment scores. The top 10 most-mentioned parties are printed with sentiment statistics, showing which events were loved, hated, or controversial.

### How to Run
1. Download and unzip the project folder.

2. Create a conda environment with
    conda create -n gg_env python=3.11

3. Activate the environment:
    conda activate gg_env

4. Install extra dependencies:
    pip install -r requirements.txt

5. Ensure tweet data is available as ggYYYY.json (given gg2013.json) in the project directory.

6. To run all modules automatically:
    python autograder.py 2013

7. To run a specific module manually:
    Hosts: python hosts.py 2013
    Awards: python awards.py 2013
    Winners: python winners.py 2013
    Nominees: python nominees.py 2013
    Presenters: python presenters.py 2013
    Red Carpet: python redcarpet.py 2013
    Parties: python parties.py 2013
