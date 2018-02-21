# Golden Globes Parser


Used to parse tweets from a corpus of 750,000 tweets gathered from Twitter during the 2018 Golden Globes using the tracking parameters: track=["goldenglobes", "golden globes", "goldenglobe", "golden globe"]. The program has a global `YEAR` variable that is used to extract the information based on that year if run with different data. 

The program works as follows:

1. Set the `YEAR` variable & path `EXT` variable 
2. Run the `parser.py` file


The program will then go to Wikipedia for the year correlated to the `YEAR` variable and find the associated awards for that ceremony. Then it builds an inverse term frequency dictionary (essentially TF-IDF with the documents being each award name) and populates each award's feature list with their most informative words. This is used to match tweets to specific awards. Each award has frequency dictionaries for possible winners, nominees and presenters. We parse through all the tweets and whenever we match a tweet to an award, we try and find out the subject (winner, nominee or presenter) of the tweet using the keywords dictionary. If the award is referring to a person, we extract the possible names using NLTK & various Regexs; otherwise, if it is not a person, we just use Regexs. From here, we add the extracted names and/or titles from the tweet and add it to the frequency dictionary of the associated award and the associated subject(s). 


We then iterate over the awards and condense the frequency dictionaries of each award's winner, nominee & presenters. This is done by looking at the set of the most popular `Length` words (a parameter set in the function `consolidate_freqs`  defaulted to 10). It then looks in the rest of the frequency dict to find all similar words in the rest of the dict and adding it to their frequency counts. This is done by using a similarity metric which varies by a 	`threshold` variable. The similarity is defined to be the percentage of words in the shorter word found in the larger word. After consolidating the frequency dicts, the program then prints out the top result for the winner, top 5 for nominees, and top 2 for presenters. 
