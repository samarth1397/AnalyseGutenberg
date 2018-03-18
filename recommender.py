import nltk
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
import random
from scipy.cluster.hierarchy import ward, dendrogram
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim import corpora, models
from collections import OrderedDict
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
import pickle
from collections import OrderedDict, Counter, defaultdict
from nltk.tokenize import sent_tokenize
import string
from sklearn.preprocessing import normalize
from senticnet.senticnet import Senticnet
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

lemmatizer = nltk.wordnet.WordNetLemmatizer()


####################################################################
#
#	Useful functions to load and preprocess the books
#
####################################################################


def loadBook(folder,filename):
    filepath=folder + "/" + filename
    f=open(filepath)
    raw=f.read()
    return(raw)

def preProcess(book):
    book=book.strip()
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(book)
    words=[w.lower() for w in words if w not in stopwords.words('english')]
    words=[lemmatizer.lemmatize(w) for w in words if len(w)>1]
    words=" ".join(words)
    return(words)
    
#######################################################################
#
#	Stylometric features
#
#######################################################################


def mean_word_length(para):
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    total=0
    count=float(0)
    for word in words:
        total=total+len(word)
        count=count+1
    return total/count
   
def average_sent_length(para):
    sents=sent_tokenize(para)
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    return len(words)/float(len(sents))

def average_commas_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count(",")
    return(total/float(len(sents)))    

def average_semicolons_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count(";")
    return(total/float(len(sents)))    


def avg_capital_letters_per_sent(para):
    sents=sent_tokenize(para)
    total=sum(1 for c in para if c.isupper())
    return(total/float(len(sents)))

def average_of_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count("of")
    return(total/float(len(sents))) 
    
def average_and_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count("and")
    return(total/float(len(sents)))

def average_but_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count("but")
    return(total/float(len(sents)))

def average_excl_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count("!")
    return(total/float(len(sents)))    

def average_very_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count("very")
    return(total/float(len(sents)))
    
def average_must_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count("must")
    return(total/float(len(sents)))
    
    
def average_more_per_sent(para):
    sents=sent_tokenize(para)
    total=0
    for s in sents:
        total=total+s.count("more")
    return(total/float(len(sents)))
    
    
def type_token_ratio(para):
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    words=[w.lower() for w in words]
    unique=set(words)
    return len(unique)/float(len(words))
    
#returns a dictionary with normalized count of POS-tags for non-stopwords
def get_posTags_para(para):
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    words=[w.lower() for w in words if w not in stopwords.words('english')]
    tags = nltk.pos_tag(words)
    counts = Counter(tag for word,tag in tags)
    total = sum(counts.values())
    c=dict((word, float(count)/total) for word,count in counts.items())
    return(c)
    
    
#returns a dictionary which conatins the count of various punctuation marks
def get_punct_dist(para):
    freq = {symbol: text.count(symbol) for symbol in string.punctuation}
    return(freq)
    
#returns a dictionary with the normalized count for all the stop words used in the para
def function_words_dist(para):
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    stop=[w.lower() for w in words if w.lower() in stopwords.words('english')]
    stopu=set(stop)
    total=len(words)
    d=dict()
    for i in stopu:
        d[i]=stop.count(i)/float(total)
    return(d)    

# yule's I measure (the inverse of yule's K measure)
# higher number is higher diversity - richer vocabulary
def yules_para(para):
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    token_counter = Counter(tok.lower() for tok in words)
    m1 = sum(token_counter.values())
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    i = (m1*m1) / (m2-m1)
    k = 1/i * 10000
    return (k, i)    

# positive and negative emotion score
sn = Senticnet()
def positiveScore(para):
    posScore=0
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    for i in words:
        try: 
            polarity_intense = float(sn.polarity_intense(i))
            if(polarity_intense>0):
                posScore=posScore+polarity_intense
        except KeyError: 
            continue
    
    return(posScore/len(words))

def negativeScore(para):
    negScore=0
    words=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(para)
    for i in words:
        try: 
            polarity_intense = float(sn.polarity_intense(i))
            if(polarity_intense<0):
                negScore=negScore+polarity_intense
        except KeyError: 
            continue
    
    return(abs(negScore)/len(words))



# Loaded books will be stored in a list sequentially. Titles of books will be stored in the same order. 

print("Do you want to recalculate the distance matrix or load an existing matrix? \n")
print("Press 1 if you want to recalculate the matrix and load the books again from the repository\n")
print("Press 2 if you want to load the matrix with filename as distanceMatrix, booksList, titlesList\n")
print("\n")
x=int(input("Enter: "))
print("\n")

if x==1:

	print("Loading books. Might take a long time depending on the number of books. ")
	folder="./books"

	titles=list()
	books_words=list()
	books=list()

	dictionary_of_names=dict() # A dictionary to maintain the mapping between the titles and the index in the list for easy searching


	i=0
	for file in os.listdir(folder):
	    # if i==100:
	    	# break
	    book=loadBook(folder,file)
	    book=book.replace('\n', ' ')
	    words=preProcess(book)
	    books.append(book)
	    words=preProcess(book)
	    books_words.append(words)
	    dictionary_of_names[file]=i
	    titles.append(file)
	    if i%50==0:
	        print(i)
	    i=i+1

	tfidfVectorizer=TfidfVectorizer(max_df=0.8)
	tfidfMatrix=tfidfVectorizer.fit_transform(books_words)
	sim = cosine_similarity(tfidfMatrix)

	pos_tags_list=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
	numOfFeatures= 16+ len(pos_tags_list) + len(stopwords.words('english'))
	features=np.zeros((len(books),numOfFeatures))

	for i in range(len(books)):
	    features[i,0]=mean_word_length(books[i])
	    features[i,1]=average_sent_length(books[i])
	    features[i,2]=average_commas_per_sent(books[i])
	    features[i,3]=average_semicolons_per_sent(books[i])
	    features[i,4]=avg_capital_letters_per_sent(books[i])
	    features[i,5]=average_of_per_sent(books[i])
	    features[i,6]=average_and_per_sent(books[i])
	    features[i,7]=average_but_per_sent(books[i])
	    features[i,8]=average_very_per_sent(books[i])
	    features[i,9]=average_must_per_sent(books[i])
	    features[i,10]=average_more_per_sent(books[i])
	    features[i,11]=type_token_ratio(books[i])
	    features[i,12]=yules_para(books[i])[0]
	    features[i,13]=positiveScore(books[i])
	    features[i,14]=negativeScore(books[i])
	    features[i,15]=average_excl_per_sent(books[i])
	    
	    #PoS Tags
	    posScore=list()
	    posDict=get_posTags_para(books[i])
	    for tag in pos_tags_list:
	        if tag in posDict.keys():
	            posScore.append(posDict[tag])
	        else:
	            posScore.append(0)
	    for j in range(36):
	        features[i,15+j]=posScore[j]

	    #function words
	    functionScore=list()
	    functDict=function_words_dist(books[i])
	    for funct_word in stopwords.words('english'):
	        if funct_word in functDict.keys():
	            functionScore.append(functDict[funct_word])
	        else:
	            functionScore.append(0)
	            
	    for j in range(len(stopwords.words('english'))):
	        features[i,15+36+j]=functionScore[j]


	features=normalize(features, axis=0, norm='max')
	simStyle = cosine_similarity(features)

	#Dumping to file
	with open("./simMatrix",'wb') as fp:
		pickle.dump(sim,fp)
	with open("./booksList",'wb') as fp:
		pickle.dump(books,fp)
	with open("titlesList",'wb') as fp:
		pickle.dump(titles,fp)
	with open("dictionaryOfNames",'wb') as fp:
		pickle.dump(dictionary_of_names,fp)
	with open("books_words_list",'wb') as fp:
		pickle.dump(books_words,fp)
	with open("simStyle",'wb') as fp:
		pickle.dump(simStyle,fp)
if x==2:
	with open("./simMatrix",'rb') as fp:
		sim=pickle.load(fp)
	with open("./booksList",'rb') as fp:
		books=pickle.load(fp)
	with open("titlesList",'rb') as fp:
		titles=pickle.load(fp)
	with open("dictionaryOfNames",'rb') as fp:
		dictionary_of_names=pickle.load(fp)	
	with open("simStyle",'rb') as fp:
		simStyle=pickle.load(fp)	

print("\n")
bookNumber=int(input("Enter eText number of the book that you want recommendations for: "))

#Load the book
text = strip_headers(load_etext(bookNumber)).strip()
title=set(get_metadata('title', bookNumber))
language=set(get_metadata('language', bookNumber))
lang=language.pop()
tl=title.pop()
tl=str(bookNumber)+"-"+tl


arrayIndex=dictionary_of_names[tl]
row=sim[arrayIndex]
numOfRecs=7
ind = np.argpartition(row, -1*numOfRecs)[-1*numOfRecs:]

print("\n")

print("Content based recommendations for ",tl)
print("\n")
j=1
for i in ind:
    print(j,"----",titles[i])
    j=j+1

print("\n\n")
arrayIndex=dictionary_of_names[tl]
row=simStyle[arrayIndex]
numOfRecs=7
ind = np.argpartition(row, -1*numOfRecs)[-1*numOfRecs:]

print("\n")

print("Style based recommendations for ",tl)
print("\n")
j=1
for i in ind:
    print(j,"----",titles[i])
    j=j+1


