# -*- coding: utf-8 -*-
"""

This program uses the global variable YEAR to find the Wikipedia page of the associated
golden globes event. From the Wikipedia page, it scrapes the name of all the awards given as well 
as the host(s) of the event. 


Parser to find:

	1. Host(s) 
	2. Award Names
	3. Presenters
	4. Nominees
	5. Winners

	** Host(s) & Award Names are parsed from Wikipedia, all else is deduced from Twitter's tweets **
 

 85% winners
 



To Run:
	set the YEAR variable to the desired year in this file and then 
	>> python parser.py
"""

import time
import os
import re
from operator import itemgetter
from collections import defaultdict 
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk


# GLOBAL DECLARATIONS

YEAR = 2018 			# the year the ceremony we want to find info abotu
FIRST = 1944 			# first ever golden globe
EXT = '../gg2018.json'	# path to twitter data from current file
suffixs = {				# dict used to convert year to word for url 

	1: "st",
	2: "nd",
	3: "rd",
}

regexs = {

	"capitals":     r'([A-Z][a-z]+(?=\s[A-Z])(?:\s[A-Z][a-z]+)+)',						# consecutive capitals
	"name_special": r'([A-Za-z]+\s[A-Z][a-z][A-Za-z]+)', 								# special names like Connor McGregor
	"title":		r'([A-Z][a-z]+(?=\s[A-Z])(?:\s[A-Z][a-z]+)+)(\s[a-z]+\s[A-Za-z]+)',	# consecutive capitals w/ lowercase word (i.e. The Shape of Water)
	"hashtag":		r'(#[A-Za-z]+)',													# #dunkirk
	"apostrophe":   r'([A-Za-z]+\s[A-Za-z]+\'[a-z]\s[A-Za-z]+)',						# apostrophe i.e. The Handmaid's Tale
	"name_proper":  r'([A-Za-z]+\s[A-Z].\s[A-Za-z]+)',									# Sterling K. Brown
	"presented":   r'(presented)(by)([A-Z][a-z]+(?=\s[A-Z])(?:\s[A-Z][a-z]+)+)'
}


keywords = {
	"person":		'(Actor|Actress|direct|cecil|score|screenplay)',
	"nominee":  	'(nom|didnt|not|did not|didn\'t|should|wow|believe)',
	"presenter":	'(pres|intro|announce|gave)',
	"winner": 		'(congr|win|won)',
	"TV":			'(tele)'
}



class Ceremony(object):
	"""
	Represents the whole ceremony
	"""
	remove_person_words = ''

	def __init__(self, year):
		"""
		initalized year to be year and populates the awards list, name string, and
		host string using the scrape_names mehtod of the Ceremony class
		"""
		self.year = year
		self.awards = []
		self.hosts = []
		self.name = ''
		self.scrape_names()
		self.remove_award_words = '(comedy|musical|TV|drama)'
		self.remove_person_words  = '(the|or|of|Limited|Role|Supporting|Series|big|little)'
		self.remove_words = '(Best|Golden|Globe|Actor|Actress|Picture|Motion|Award|Year|Film|Animated|Feature|or|Series|comedy|musical|drama|tv)'

		# print ("setting remove words to be {}".format(self.remove_words.encode('utf-8')))


	def __str__(self):
		"""
		string representation of a ceremony

		will build out later to inclue all the findings 
		"""
		return ("\nThe {} held during {}" .format(self.name, self.year))


	def scrape_names(self):
		"""
		Uses wikipeadia page of the golden globes ceremony associated with the YEAR global variable.
		Populates the ceremony object's awards, host and name attributes.  
		"""

		num_award = YEAR - FIRST + 1

		if (num_award % 10) in suffixs.keys():
			str_format = str(num_award) + suffixs[num_award % 10]
		else:
			str_format = str(num_award) + "th"

		# update ceremony object's name 
		self.name = str_format + " Golden Globe Awards"

		
		# make the web request and get the infromation from wiki page
		url = 'https://en.wikipedia.org/wiki/{}_Golden_Globe_Awards'.format(str_format)
		result = requests.get(url)
		c = result.content

		# store in BeautifulSoup object to parse HTML DOM
		soup = BeautifulSoup(c, "lxml")
		

		#
		#	Find Awards 
		#
		awards_div = soup.find("div", {"aria-labelledby" : "Golden_Globe_Awards"})

		awards = []
		tds = awards_div.find_all("td")
		for td in tds:
			lis = td.find_all("li")

			for li in lis:
				a = li.find("a")
				try:
					title = a["title"]
					award = Award(title)
					awards.append(award)
				except:
					continue


		awards_filtered = filter(lambda award: award.title.startswith("Golden Globe Award"), awards)	 # filter noise from awards list
		awards_filtered = awards_filtered[:-3]
		self.awards = awards_filtered 	# update ceremony object's award attribute
		self.awards.append(Award('Cecil B. DeMille Award'))


		#
		#	Find Host
		#
		hosts = []
		table = soup.find("table", {"class" : "infobox vevent"})
		trs = table.find_all("tr")
		for tr in trs:
			try:
				th = tr.find("th")
				if th.text == "Hosted by":
					a_tags = tr.find_all("a")
					for a in a_tags:
						hosts.append(a.text)
			except:
				continue

		# update ceremony object's host attribute
		self.hosts = hosts

	
	def build_remove_award_words(self):
		"""
		makes a remove words list from all words present in the awards;
		it does not work as well as current implementation
		"""
		begin = '('
		end   = ')'
		keepers = ['the', 'for', '-', 'of', 'or', 'New']
		for award in self.awards:

			begin += '|'.join([word for word in award.title.split(' ') if not(word in keepers)])

		begin += end
		return begin


	def add_remove_person_words(self, addition):
		"""
		append all words in the additions text to the remove words regex
		"""
		add = '|'.join(addition.split(' '))
		self.remove_person_words = self.remove_person_words[:-1] + '|' + add + ')'


	def build_award_features(self):
		"""	
		Creats a list of most uncommon words found in each award based on all words present in all awards.
		Then sets each Award object in self.awards list to have the associated features list. 
		Used to find most important words in each award which is then used for matching tweets to associated awards.
		"""
		award_names = [award.title for award in self.awards]
		phrase_freqs = self.count_phrases(award_names, 1)


		# converts the phrase_freqs dictionary into a sorted list in ascending order
		# it creates each index as a tuple with the val[0] being the phrase
		# and val[1] being the frequency of that phrase
		phrase_freqs_sorted = sorted(phrase_freqs.items(), key=itemgetter(1), reverse=False)


		award_names = [award.title.split(' ') for award in self.awards]
		features_list = [[] for i in range(len(award_names))]
		min_len = np.zeros(len(award_names))
		feature_max = 5

		for val, freq in phrase_freqs_sorted:
			# check if we found a total of feature_max features
			if min(min_len) == feature_max: 
				break

			i = 0
			while i < len(award_names):
				# check if this value is in the award name
				if val.capitalize() in award_names[i]:
					if min_len[i] < feature_max:
						if val == 'Television':
							features_list[i].append('(Television)|(TV)')
						else:
							features_list[i].append(val)
						min_len[i] += 1
				i += 1


		for i, award in enumerate(self.awards):
			# print ("setting {} to have || features: {}".format(award.title.encode('utf-8'), features_list[i]))
			award.features_list = features_list[i]
	

	def remove_stop_words(self, words, stop_words):
		"""
		uses the stop_word string to remove all words represented in the stop_word string.
		the stop_word string is 1-x number of words concatenated together. 
		"""
		clean_words = []
		for word in words:
			if not re.search(word, stop_words, flags=re.I):
				clean_words.append(word)
		return clean_words
		

	def count_phrases(self, data, phrase_length, stop_words=None):
		"""
		Builds a frequency distrobution of all combinations of words of size phrase_length present in the
		given list, if one of the words present is in the stop_words then it will not be considered.
		"""
		phrase_freqs = defaultdict(lambda: 0)
		for i in range(len(data)):
			words = re.findall(r'[a-zA-Z]+', data[i])

			if stop_words != None:
				words = self.remove_stop_words(words, stop_words)

			for j in range(len(words) - phrase_length + 1):
				tmp_phrase = ''
				for k in range(phrase_length):
					tmp_phrase += words[j+k].lower()
					if k != phrase_length - 1:
						tmp_phrase += ' '

				phrase_freqs[tmp_phrase] += 1

		return phrase_freqs


	def clean_words(self, data, mode, min_word_len=3, word_dict=[]):
		"""
		if mode set to 1: filters out all words in each data instance with words less than min_word_len
		if mode set to not 1: filters out all words in each data instance if not present in word_dict
		returns a list of 'clean' data
		"""
		clean_data = []
		for inst in data:

			if mode == 1:
				words = filter(lambda w: len(w) >= min_word_len, inst.split(' '))
			else:
				words = filter(lambda w: w in word_dict, inst.split(' '))

			if len(words):
				clean_data.append(' '.join(words))

		return clean_data


	def keywords_search(self, keywords, data):
		"""
		searches for all keywords in the list of keywords
		returns a subset of the data the has ALL keywords present in it
		the data refers to tweets
		"""
		matches = []
		for inst in data:
			for keyword in keywords:
				if all([re.search(keyword, inst, flags=re.IGNORECASE) for keyword in keywords]):
					matches.append(inst)
		return matches


	def consolidate_freqs(self, freqs, length=10):
		"""
		
		"""
		top_length = freqs[:length]
		rest  = freqs[length:]
		consolidated = {}

		for inst in top_length:
			total = inst[1]

			for sub_inst in rest:
				if self.similar(sub_inst[0], inst[0]):
					total += sub_inst[1]
			consolidated[inst[0]] = total
		consolidated = sorted(consolidated.items(), key=itemgetter(1), reverse=True)

		return consolidated


	def compact_top(self, top, threshold=0.8):
		"""

		"""
		compact = []
		duplicates = []
		for i in range(len(top)):
			val = top[i]
			new_val = val[0]
			total = val[1]
			for j, item in enumerate(top[i+1:]):
				if self.similar(val[0], item[0], threshold=threshold):
					total += item[1]
					if len(new_val) < len(item[0]):
						new_val = item[0]
					duplicates.append(item[0])

			if new_val[0] in duplicates:
				continue

			compact.append((new_val, total))

		s = sorted(compact, key=itemgetter(1), reverse=True)
		# print ('top is {}'.format(top))
		# print ('s is {}'.format(s))
		return s


	def similar(self, phrase1, phrase2, threshold=0.8):
		"""
		checks how similar two phrases are based on the 
		threshold parameter
		"""
		total = 0
		if isinstance(phrase1, tuple):
			phrase1 = str(phrase1[0]) + str(phrase1[1])

		if isinstance(phrase2, tuple):
			phrase2 = str(phrase2[0]) + str(phrase2[1])

		p1_len = len(phrase1.split(' '))
		p2_len = len(phrase2.split(' '))
		ratio = 0.0

		if p1_len < p2_len:
			p2_split = [p.lower() for p in phrase2.split(' ')]
			for p in phrase1.split(' '):
				if p.lower() in p2_split:
					total += 1

			ratio = float(total / p1_len)

		else:
			p1_split = [p.lower() for p in phrase1.split(' ')]
			for p in phrase2.split(' '):
				if p.lower() in p1_split:
					total += 1

			ratio = float(total) / float(p2_len) 

		return ratio >= threshold


	def nltk_name_finder(self, tweet):
		"""
		uses NLTK to find all possible names in a tweet
		"""
		try:
			tokens = word_tokenize(tweet)
			pos = pos_tag(tokens)
			sent = ne_chunk(pos, binary=False)
		except:
			return []
		people = []
		person = []
		name = ""
		for subtree in sent.subtrees(filter=lambda t: t.label() == "PERSON"):
			for leaf in subtree.leaves():
				person.append(leaf[0])
			if len(person) > 1:
				for part in person:
					name += part + " "
				if name[:-1] not in people:
					people.append(name[:-1])
				name = ""
			person = []

		# print ("people are {}".format(people))
		return people 


	def parse_tweets(self, tweets):
		"""
		used to analyze the set of tweets and try to populate the winner, nominees, and presenter attributes of 
		each Award object in the Ceremony's awards attribute. 
		"""

		tweets = list(set(tweets))
		num_feats_enforce = 3



		# iterate over all features for each index
		for award_idx in range(len(self.awards)):
			
			features_list = self.awards[award_idx].features_list

			# set the number of features we want to enforce
			if re.search(keywords["TV"], self.awards[award_idx].title, flags=re.IGNORECASE): 
				features_list = ["(tele|tv)"] + features_list
			# 	num_feats_enforce = 2
			# else:
			# 	num_feats_enforce = 3

			for idx, feat in enumerate(features_list):
				if feat == "(comedy|musical)":
					del features_list[idx]
				if feat == "comedy" or feat == "musical":
					features_list[idx] = "(comedy|musical)"
				if feat == "golden" or feat == "globe":
					del features_list[idx]


			# fins all tweets referencing this award 
			associated_tweets = self.keywords_search(features_list[:num_feats_enforce], tweets)


			person = bool(re.search(keywords["person"], self.awards[award_idx].title, flags=re.IGNORECASE))
			# print ('feats are {} for {}'.format(features_list[:num_feats_enforce], self.awards[award_idx].title.encode('utf-8')))

			for tweet in associated_tweets:

				winner    = False
				nominee   = False
				presenter = False

				# check what subject the tweet was referencing 
				if re.search(keywords["winner"],    tweet, flags=re.IGNORECASE): winner     = True
				if re.search(keywords["nominee"],   tweet, flags=re.IGNORECASE): nominee    = True
				if re.search(keywords["presenter"], tweet, flags=re.IGNORECASE) and re.search("awards_watch", tweet, flags=re.IGNORECASE):
					presenter  = True

				# only want to extract Names & Titles if tweet has possible value
				if (nominee or winner or presenter):

					people 	= self.nltk_name_finder(tweet)
					people.extend(re.findall(regexs["name_special"], tweet))

					if person:
						# we do this to limit noise 
						candidates = filter(lambda person: not bool(re.search(self.remove_person_words, person)), people)
						# candidates = people

					else:
						capitals 		= re.findall(regexs["capitals"], tweet)
						titles 			= re.findall(regexs["title"], tweet)
						hashtags		= re.findall(regexs["hashtag"], tweet)
						presented       = re.findall(regexs["presented"], tweet)
						# apostrophes     = re.findall(regexs["apostrophe"], tweet)
						# names_proper    = re.findall(regexs["name_proper"], tweet)

						# extract information from hashtags
						hashtags_clean = []
						for hashtag in hashtags:
							hashtag = hashtag[1:]
							hashtags_clean.extend(re.findall('[A-Z][^A-Z]*', hashtag))	# Split capital letters DunkirkMovie --> Dunkirk Movie

						# build possible names & titles 
						candidates = []
						# for list_ in [names, names_special, titles, hashtags_clean, apostrophes]:
						for list_ in [titles, hashtags_clean, capitals]:
							if list_ != None:
								candidates.extend(list_)

						# clean all possible names & titles
						clean_cands = []
						for cand in candidates:
							if isinstance(cand, tuple):
								clean_cands.append(cand[0] + cand[1])
							else:
								clean_cands.append(cand)

						candidates = clean_cands

					
					# update awards frequency dictionaries 
					for cand in candidates:
						if winner:    self.awards[award_idx].winner[cand] += 1
						if nominee:   self.awards[award_idx].nominees[cand] += 1

					if presenter:
						for person in people:
							self.awards[award_idx].presenters[person] += 1
						for pres in presented:
							self.awards[award_idx].presenters[pres] += 1
		

	def build_results(self):
		"""
		Use dictionaries for each awards possible  Winners, Presenters and Nominees 
		and extract most likely candidates
		"""
		print ("----------------------------------")
		for award in self.awards:

			print ('For award: {}'.format(award.title.encode('utf-8')))

			person = False
			person = bool(re.search(keywords["person"], award.title, flags=re.IGNORECASE))

			#
			# Winner
			#

			winner = sorted(award.winner.items(), key=itemgetter(1), reverse=True)
			winner = filter(lambda w: not bool(re.search(self.remove_words, w[0], flags=re.IGNORECASE)), winner)
			

			# winner = filter(lambda w: not re.search(remove_award_words, w[0], flags=re.IGNORECASE), winner)
			

			winner = self.consolidate_freqs(winner)
			winner = self.compact_top(winner, threshold=0.5)
			# print ('winner before: {}'.format(winner))
			if person: winner = filter(lambda w: not bool(re.search(self.remove_person_words, w[0], flags=re.IGNORECASE)), winner)
			

			print ('Winner is: {}'.format(winner[0][0]))


			#
			# Nominees 
			#

			nominees = sorted(award.nominees.items(), key=itemgetter(1), reverse=True)
			nominees = filter(lambda w: not re.search(self.remove_words, w[0],  flags=re.IGNORECASE), nominees)

			
			if person: nominees = filter(lambda w: not bool(re.search(self.remove_person_words, w[0], flags=re.IGNORECASE)), nominees)


			nominees = self.consolidate_freqs(nominees)
			if person: 	nominees = self.compact_top(nominees, threshold=0.5)
			else: 		nominees = self.compact_top(nominees)
			
			ns = [n[0] for n in nominees]
			print ('Nominees are: {}'.format(ns[:4]))


			
			#
			# Presenter(s)
			#

			presenters = sorted(award.presenters.items(), key=itemgetter(1), reverse=True)
			presenters = filter(lambda w: not re.search(self.remove_words,  w[0], flags=re.IGNORECASE), presenters)
			

			presenters = filter(lambda w: not bool(re.search(self.remove_person_words, w[0], flags=re.IGNORECASE)), presenters)

			presenters = self.consolidate_freqs(presenters)
			presenters = self.compact_top(presenters, threshold=0.5) 		# presenters are always people

			prs = [pr[0] for pr in presenters]
			print ('Presenters are: {}'.format(prs[:2]))

			
			print ("----------------------------------")
		


class Award(object):
	"""
	Represents a single award
	"""
	def __init__(self, title):
		self.title = title
		self.presenters = defaultdict(lambda: 0)
		self.nominees = defaultdict(lambda: 0)
		self.winner = defaultdict(lambda: 0)
		self.features_list = []


def read_tweets():
	"""
	Reads in all the tweets into a list and returns the list. 
	Uses the global EXT variable to find the data file.
	"""
	cwd = os.getcwd() 
	path = cwd + '/' + EXT
	f = open(path)

	tweets_json = json.load(f)
	tweets = []
	for tweet in tweets_json:
		tweets.append(tweet['text'])
	return tweets


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f mins' % \
                (method.__name__, (float(te - ts)/float(60)))
        return result
    return timed

@timeit
def main():
	"""
	"""
	ceremony = Ceremony(YEAR)
	ceremony.scrape_names()
	ceremony.build_award_features()
	tweets = read_tweets()
	ceremony.parse_tweets(tweets)
	print (ceremony)
	print ("The host is {}".format(ceremony.hosts))
	ceremony.build_results()
	


if __name__ == "__main__":
	main()
