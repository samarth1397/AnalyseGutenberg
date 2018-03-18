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
	books=list()

	dictionary_of_names=dict() # A dictionary to maintain the mapping between the titles and the index in the list for easy searching


	i=0
	for file in os.listdir(folder):
	    # if i==100:
	    	# break
	    book=loadBook(folder,file)
	    words=preProcess(book)
	    books.append(book)
	    dictionary_of_names[file]=i
	    titles.append(file)
	    if i%50==0:
	        print(i)
	    i=i+1

	tfidfVectorizer=TfidfVectorizer(max_df=0.8)
	tfidfMatrix=tfidfVectorizer.fit_transform(books)
	dist = 1 - cosine_similarity(tfidfMatrix)

	#Dumping to file
	with open("./distanceMatrix",'wb') as fp:
		pickle.dump(dist,fp)
	with open("./booksList",'wb') as fp:
		pickle.dump(books,fp)
	with open("titlesList",'wb') as fp:
		pickle.dump(titles,fp)
	with open("dictionaryOfNames",'wb') as fp:
		pickle.dump(dictionary_of_names,fp)

if x==2:
	with open("./distanceMatrix",'rb') as fp:
		dist=pickle.load(fp)
	with open("./booksList",'rb') as fp:
		books=pickle.load(fp)
	with open("titlesList",'rb') as fp:
		titles=pickle.load(fp)
	with open("dictionaryOfNames",'rb') as fp:
		dictionary_of_names=pickle.load(fp)	

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
row=dist[arrayIndex]
numOfRecs=7
ind = np.argpartition(row, -1*numOfRecs)[-1*numOfRecs:]

print("\n")

print("Content based recommendations for ",tl)
print("\n")
j=1
for i in ind:
    print(j,"----",titles[i])
    j=j+1





