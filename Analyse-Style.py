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
from collections import OrderedDict, Counter, defaultdict
from nltk.tokenize import sent_tokenize
import string
import numpy as np
from sklearn.preprocessing import normalize
from senticnet.senticnet import Senticnet

#########################################
# FEW USEFUL FUNCTIONS
#######################################

def loadBook(folder,filename):
    filepath=folder + "/" + filename
    f=open(filepath)
    raw=f.read()
    return(raw)
    
def preProcess(book):
    book=book.replace('\n', ' ')
    sents=sent_tokenize(book)
    return(sents)
    
def createParagraphs(sents,filename,text_list,id_list,numOfSents=25):
    i=0
    count=0
    while(i<len(sents)):
        if((i+numOfSents)<len(sents)):
            para=sents[i:i+numOfSents]
            para=" ".join(para)
        else:
            para=sents[i:len(sents)]
            para=" ".join(para)
        i=i+numOfSents
        count=count+1
        text_list.append(para)
        id_list.append(filename + "-" + str(count))
    return(text_list,id_list)
    


#################################################################
# STYLOMETRIC FEATURES
#################################################################


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


###########################################

#    STYLOMETRIC FEATURES OVER. 

##########################################   

author1_books=[]
author1_id=[]
folder="author1/en/"

print("Loading works of author 1...............\n\n\n")

i=1
dictionaryOfAuthor1Books=OrderedDict()
for file in os.listdir(folder):
    #print(file)
    book=loadBook(folder,file)
    sents=preProcess(book)
    filename="A1-"+"B"+str(i)
    createParagraphs(sents,filename,author1_books,author1_id)
    dictionaryOfAuthor1Books["B"+str(i)]=file
    i=i+1

print("Loading complete.......")
for i in dictionaryOfAuthor1Books.keys():
    print(i,"----",dictionaryOfAuthor1Books[i])
    
print("\n\n\n")


author2_books=[]
author2_id=[]
folder="author2/en/"
print("Loading works of author 2...............\n\n\n")

i=1
dictionaryOfAuthor2Books=OrderedDict()
for file in os.listdir(folder):
    #print(file)
    book=loadBook(folder,file)
    sents=preProcess(book)
    filename="A2-"+"B"+str(i)
    createParagraphs(sents,filename,author2_books,author2_id)
    dictionaryOfAuthor2Books["B"+str(i)]=file
    i=i+1


print("Loading complete.......")
for i in dictionaryOfAuthor2Books.keys():
    print(i,"----",dictionaryOfAuthor2Books[i])


books=author1_books+author2_books
ids=author1_id+author2_id


#removing any small chunks that have less than 15 sentences
i=0
while(i<len(books)):
    s=sent_tokenize(books[i])
    if(len(s)<15):
        prev_l=len(sent_tokenize(books[i-1]))
        books[i-1]=" ".join(books[i-1:i+1])
        new_l=len(sent_tokenize(books[i-1]))
        #print(ids[i],len(s),prev_l,new_l)
        del books[i]
        del ids[i]
    i=i+1



pos_tags_list=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

numOfFeatures= 16+ len(pos_tags_list) + len(stopwords.words('english'))

features=np.zeros((len(books),numOfFeatures))

print("\n\n\nGenerating features............. Will take a bit of time.....\n\n\n")

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
        features[i,13+j]=posScore[j]

    #function words
    functionScore=list()
    functDict=function_words_dist(books[i])
    for funct_word in stopwords.words('english'):
        if funct_word in functDict.keys():
            functionScore.append(functDict[funct_word])
        else:
            functionScore.append(0)
            
    for j in range(len(stopwords.words('english'))):
        features[i,13+36+j]=functionScore[j]


features=normalize(features, axis=0, norm='max')

print("Feature generation complete........... Preparing for plotting........\n\n")
#preparing for plotting

author=[" "]*len(books)
num_author1=0
num_author2=0
for i in range(len(ids)):
    if (ids[i][0:2]=="A1"):
        author[i]="Author1"
        num_author1=num_author1+1
    else:
        author[i]="Author2"
        num_author2=num_author2+1


dist = 1 - cosine_similarity(features)

num_of_clusters=25
kmeans=KMeans(n_clusters=num_of_clusters)
kmeans.fit(features)
clusters=kmeans.labels_.tolist()

documents = { 'title': ids,  'text': books, 'cluster': clusters, 'author':author }
documentFrame = pd.DataFrame(documents, index = [clusters])

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=7)
transformed = mds.fit_transform(dist)
x=transformed[:, 0]
y=transformed[:, 1]

x_author1=transformed[:num_author1,0]
y_author1=transformed[:num_author1,1]

x_author2=transformed[num_author1:,0]
y_author2=transformed[num_author1:,1]

num_colours=num_of_clusters
col=mpl.colors.cnames.items()
colours={}
random.seed(5)
i=0
while (i<num_of_clusters):
    c=random.choice(list(col))
    if(c[1] not in colours.keys()):
        colours[i]=c[1]
        i=i+1


cluster_names={}
for i in range(num_of_clusters):
    cluster_names[i]="Cluster-"+str(i)

df = pd.DataFrame(dict(x=x, y=y, label=clusters, title=ids,author=author)) 
groups = df.groupby('label')

df_author1=pd.DataFrame(dict(x=x_author1, y=y_author1, label=clusters[:num_author1], title=ids[:num_author1],author=author[:num_author1]))
groups_author1 = df_author1.groupby('label')

df_author2=pd.DataFrame(dict(x=x_author2, y=y_author2, label=clusters[:num_author2], title=ids[:num_author2],author=author[:num_author2]))
groups_author2 = df_author2.groupby('label')


fig, ax = plt.subplots(figsize=(20, 20)) # set size
#Marx's books are marked by circles
for name, group in groups_author1:
    ax.plot(group.x, group.y, marker='D', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')
#Hegel's books are marked by diamonds
for name, group in groups_author2:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')

ax.legend(numpoints=1)  #show legend 

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=3)

plt.show()



















