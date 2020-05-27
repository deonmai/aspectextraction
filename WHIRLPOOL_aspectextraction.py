#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import namedtuple, OrderedDict
import json
from collections import defaultdict
import pandas as pd
pd.options.display.max_rows = 10000


# In[3]:


# dataset = xml.dom.minidom.parse("test_datasets/restaurants-trial.xml")
root = ET.parse("test_datasets/laptops-trial.xml").getroot()
dataset = ET.parse("test_datasets/laptops-trial.xml").getroot().findall('sentence')

actual_dataset = ET.parse("test_datasets/laptops-trial.xml").getroot().findall('sentence')

with open("test_datasets/whirlpool.json") as read_file:
    full_dataset = json.load(read_file)
dataset = full_dataset


def validate(dataset):
    elements = dataset
    text = []
    for e in elements:
        sent_id = e['index']
        sent = e['reply']
        text.append((sent, sent_id))
    return text


# In[7]:


text = validate(dataset)



# In[9]:


#LEXICON - subjectivity clues Bing Liu
SubClue = namedtuple('SubClue', ['rel', 'pri_pol', 'stemmed'])
class SubjectivityClues(object):
    # map POS tags in subjectivty clues to spacy tags:
    _pos_map = {
        'noun': 'NOUN',
        'verb': 'VERB',
        'adj': 'ADJ',
        'adverb': 'ADV',
        'anypos': 'ANYPOS',
        }

    def __init__(self, sc_path=
            'lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'):

        # load subjectivity clues
        with open(sc_path, 'r') as fo:
            sub_clues = [c.strip().replace('type=', 'rel=').split()
                         for c in fo]
        sub_clues = [dict([d.split('=') for d in sc if len(d.split('=')) == 2])
                     for sc in sub_clues]

        # convert to dict keyed by (word, part-of-speech) and exclude neutral
        # polarity clues:
        # ISSUE: if stemmed for matches non-stemmed will overwrite (may not
        # have impact as polarity and type remain the same
        self.lex = {(s['word1'], self._pos_map[s['pos1']]): SubClue(s['rel'],
                     s['priorpolarity'],
                     True if s['stemmed1'] == 'y' else False)
                    for s in sub_clues if s['priorpolarity']}
        # sclues_pos and sclues_any are just sets containing all (word, pos)
        # tuples with a proper POS tag and 'ANYPOS' respectively
        self.sclues_pos = set([(w, t) for w, t in self.lex.keys()
                               if t != 'ANYPOS'])
        self.sclues_any = set(self.lex.keys()).difference(self.sclues_pos)

    def lookup(self, token):
        """
        Returns a `SubClue` named tuple for `token` if either `(token.norm_,
        token.pos_)` or `(token.norm_, 'ANYPOS') is in self.lex. Returns an
        empty tuple if no match is found.
        """
        tnp = (token.norm_, token.pos_)
        tna = (token.norm_, 'ANYPOS')
        if tnp in self.lex or tna in self.lex:
            key = tnp if tnp in self.lex else tna
            return self.lex[key]
        else:
            return tuple()


# In[10]:


subjclues = SubjectivityClues()


# In[11]:


import spacy
from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
# from spacy.util import filter_spans
from spacy.lang.en.stop_words import STOP_WORDS
import string
import neuralcoref

punc = string.punctuation
stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_lg')
pronoun_list=['he','she','it','they','them','him','her','his','hers','its','we','us']


# In[12]:


neuralcoref.add_to_pipe(nlp)


# In[13]:



# In[14]:




# In[15]:


# noun_chunks from spacy

from spacy.symbols import NOUN, PROPN, PRON

def noun_chunks(obj):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    labels = [
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
    ]
    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    np_label = doc.vocab.strings.add("NP")
    seen = set()
    for i, word in enumerate(obj):
        # print("i", i)
        # print("word", word)
        if word.pos not in (NOUN, PROPN, PRON):
            continue
        # Prevent nested chunks from being produced
        if word.i in seen:
            continue
        if word.dep in np_deps:
            if any(w.i in seen for w in word.subtree):
                continue
            seen.update(j for j in range(word.left_edge.i, word.i + 1))
            yield word.left_edge.i, word.i + 1, np_label
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                if any(w.i in seen for w in word.subtree):
                    continue
                seen.update(j for j in range(word.left_edge.i, word.i + 1))
                yield word.left_edge.i, word.i + 1, np_label


SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}


# In[16]:


# In[17]:


# In[18]:


new_stopwords = ['time', 'trouble', 'notch', 'the', 'ie', 'eg', 'an', 'thing', 'mouth', 'restaurant', 'bit', 'i', "$", "nsw", "sa", "brisbane", "sydney", "adelaide", "melbourne", "melb", "canberra", "act", "wa", "australia", "Australia"]
for word in new_stopwords:
    stopwords.append(word)

# In[19]:


more_subj = ['deficiency', 'consistently', 'mouth watering', 'all costs']


# In[20]:


def pre_process(sent):
    doc = nlp(sent)
    
    lemmad = ""
    
    #lemmatization
    for token in doc:
        if token.text not in stopwords:
            if token.lemma_ != "-PRON-":
                temp = token.lemma_.lower().strip()
            else:
                temp = token.lower_
            lemmad = lemmad + temp + " "
    lemmad = nlp(lemmad)
    
    #remove stop words and punctuation
    cleaned_tokens = ""
    for token in lemmad:
        if token.text not in stopwords and token.text not in punc:
            cleaned_tokens = cleaned_tokens + token.text + " "
    
    cleaned_tokens = nlp(cleaned_tokens)
    return cleaned_tokens

# ('people', [23673])
# ('property', [10956])
# ('money', [9408])
# ('prices', [9372])
# ('house', [9275])
# ('market', [9153])
# ('rates', [6015])
# ('banks', [5025])
# ('price', [4897])
# ('lot', [4458])
#EXTRACT ASPECTS
my_aspects = []

token_aspects = []
for i in range(len(text)):
	print(i, end = " ")
	current_sent = text[i][0]
	current_id = text[i][1]
	if len(current_sent) > 1000:
		print("skipped")
		continue

	raw_doc = nlp(current_sent)
	pp_doc = pre_process(current_sent)
	temp_aspect = ""
    	
#     list of ents
	ents_list = []
	for ent in raw_doc.ents:
	    ents_list.append(ent.text)

	coref_clusters = raw_doc._.coref_clusters

	for chunk in raw_doc.noun_chunks:
	    #do this at start in case next noun chunk is part of the previous one e.g. okra (bindi)
	    if (chunk.text[0] == "("):
	        temp_aspect = temp_aspect + chunk.text
	        temp_aspect += ") "
	        previous_i = current_i
	        current_i = token.i
	        chunk_span = raw_doc[previous_i:current_i+1]
	        continue
	    
	    if(temp_aspect != ''):
	        temp_aspect = temp_aspect[:-1]
	        my_aspects.append((temp_aspect, current_id))
	        token_aspects.append((chunk_span, current_id))
	    temp_aspect = ""
	    
	    if chunk.text in ents_list:
	        continue
	    
	    for cluster in coref_clusters:
	        if chunk in cluster and chunk != cluster[0]:
	            continue

	    #find index of first key aspect word
	    for token in chunk:
	        if token.lemma_.lower() in pp_doc.text and not subjclues.lookup(token) and token.lemma_.lower() not in more_subj:
	            #remove useless words 'the', 'a', etc before a noun. 
	            if((token.lemma_.lower() == "the" or token.lemma_.lower() == "a") and token.dep_ == "det"):
	                continue
	            current_i = token.i
	            break
	            
	    for token in chunk:
	        #add if its in the pre-processsed doc (meaning its important) and NOT in lexicon.
	        if token.lemma_.lower() in pp_doc.text and not subjclues.lookup(token) and token.lemma_.lower() not in more_subj:
	            #remove useless words 'the', 'a', etc before a noun. 
	            if((token.lemma_.lower() == "the" or token.lemma_.lower() == "a") and token.dep_ == "det"):
	                continue
	            if(token.text in ents_list or token.lemma_.lower() == "i" or token.lemma_.lower() == "and"):
	                continue
	            previous_i = current_i
	            current_i = token.i
	            temp_aspect = temp_aspect + token.text + " "
	            chunk_span = raw_doc[previous_i:current_i+1]    

	#last time to get the last aspect
	if(temp_aspect != ''):
	        temp_aspect = temp_aspect[:-1]
	        my_aspects.append((temp_aspect, current_id))
	        token_aspects.append((chunk_span, current_id))

print()
print("# BEFORE", len(my_aspects))

with open('my_aspectsNOCOREF.txt', 'a') as f:
    for i in range(len(my_aspects)):
        print(my_aspects[i], file = f)

sum_aspects = defaultdict(list)
for aspect in my_aspects:
    if len(sum_aspects[aspect[0]]) == 0:
        sum_aspects[aspect[0]].append(1)
    else:
        sum_aspects[aspect[0]][0] += 1

sum_aspects = sorted(sum_aspects.items(), key=lambda k_v: k_v[1], reverse=True) #2010

for i in range(20):
    print(sum_aspects[i])
    

# already_seen = []
# already_removed = []
# for current_token in token_aspects:
#     if current_token[0].text in already_seen:
#         continue

#     print(current_token[1])
    
#     max_sim = 0
#     sum_sim = 0
#     avg_sim = 0
    
#     for compare_token in token_aspects:
#         if compare_token[0].text in already_removed:
#             continue
            
#         if current_token[1] != compare_token[1]:
#             if(current_token[0].similarity(compare_token[0])):
#                 sim_score = current_token[0].similarity(compare_token[0])
#                 sum_sim += sim_score
#                 if max_sim < sim_score:
#                     max_sim = sim_score

#     # print(current_token[0], avg_sim, max_sim)
#     avg_sim = sum_sim / len(token_aspects)
#     if (avg_sim < 0.17 or max_sim < 0.60) and avg_sim != 0.0:
# #         if((current_token[0].text, current_token[1]) in my_aspects):
#         my_aspects = [aspect for aspect in my_aspects if aspect[0] != current_token[0].text]
    
#         already_removed.append(current_token[0].text)
#         if len(already_removed) > 200:
#             already_removed = []
            
#         # print(current_token[0], avg_sim, max_sim)
#         # print("\tremoved")
        	
#     already_seen.append(current_token[0].text)
#     if len(already_seen) > 200:
#         already_seen = []



print("# AFTER", len(my_aspects))

# In[ ]:


sum_aspects = defaultdict(list)
for aspect in my_aspects:
    if len(sum_aspects[aspect[0]]) == 0:
        sum_aspects[aspect[0]].append(1)
    else:
        sum_aspects[aspect[0]][0] += 1

sum_aspects = sorted(sum_aspects.items(), key=lambda k_v: k_v[1], reverse=True) #2010

for i in range(10):
    print(sum_aspects[i])
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




