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
lemmatizer = nltk.wordnet.WordNetLemmatizer()


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
    return(words)
    

def createChunks(words,filename,text_list,id_list):
    i=0
    count=0
    while(i<len(words)):
        if((i+1000)<len(words)):
            chunk=words[i:i+1000]
            chunk=" ".join(chunk)
        else:
            chunk=words[i:len(words)]
            chunk=" ".join(chunk)
        i=i+1000
        count=count+1
        text_list.append(chunk)
        id_list.append(filename + "-" + str(count))
    return(text_list,id_list)
    
    
author1_books=[]
author1_id=[]
folder="author1/en/"

print("Loading English books for author 1")

i=1
dictionaryOfAuthor1Books=OrderedDict()
for file in os.listdir(folder):
    #print(file)
    book=loadBook(folder,file)
    words=preProcess(book)
    filename="A1-"+"B"+str(i)
    createChunks(words,filename,author1_books,author1_id)
    dictionaryOfAuthor1Books["B"+str(i)]=file
    i=i+1
    
for i in dictionaryOfAuthor1Books.keys():
    print(i,"----",dictionaryOfAuthor1Books[i])
   
print("\n\n\nLoading English books for author 2")

author2_books=[]
author2_id=[]
folder="author2/en/"

i=1
dictionaryOfAuthor2Books=OrderedDict()
for file in os.listdir(folder):
    #print(file)
    book=loadBook(folder,file)
    words=preProcess(book)
    filename="A2-"+"B"+str(i)
    createChunks(words,filename,author2_books,author2_id)
    dictionaryOfAuthor2Books["B"+str(i)]=file
    i=i+1
    
for i in dictionaryOfAuthor2Books.keys():
    print(i,"----",dictionaryOfAuthor2Books[i])


books=author1_books+author2_books
ids=author1_id+author2_id

###### Merge the smaller chunks ###############
i=0
while(i<len(books)):
    b=nltk.word_tokenize(books[i])
    if(len(b)<700):
        prev_l=len(nltk.word_tokenize(books[i-1]))
        books[i-1]=" ".join(books[i-1:i+1])
        new_l=len(nltk.word_tokenize(books[i-1]))
        #print(ids[i],len(b),prev_l,new_l)
        del books[i]
        del ids[i]
    i=i+1
    
print("\n\n\n")
#print("Number of chunks of text written by author 1: ",len(author1_id))
#print("Number of chunks of text written by author 2: ",len(author2_id))

######################################################## TF-IDF based Clustering #############################################################
print("TfIDF based clustering........")

#TF IDF Matrix
tfidfVectorizer=TfidfVectorizer(max_df=0.8)
tfidfMatrix=tfidfVectorizer.fit_transform(books)
#Distance matrix
dist = 1 - cosine_similarity(tfidfMatrix)

#K-Means clustering
num_of_clusters=25
kmeans=KMeans(n_clusters=num_of_clusters)
kmeans.fit(tfidfMatrix)
clusters=kmeans.labels_.tolist()


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
        
documents = { 'title': ids,  'text': books, 'cluster': clusters, 'author':author }
documentFrame = pd.DataFrame(documents, index = [clusters])

#Convert to 2D for plotting
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=7)
transformed = mds.fit_transform(dist)
x=transformed[:, 0]
y=transformed[:, 1]

x_author1=transformed[:num_author1,0]
y_author1=transformed[:num_author1,1]

x_author2=transformed[num_author1:,0]
y_author2=transformed[num_author1:,1]

#Generate colours for plotting
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
    

#Preparing for plotting    
df = pd.DataFrame(dict(x=x, y=y, label=clusters, title=ids,author=author)) 
groups = df.groupby('label')

df_author1=pd.DataFrame(dict(x=x_author1, y=y_author1, label=clusters[:num_author1], title=ids[:num_author1],author=author[:num_author1]))
groups_author1 = df_author1.groupby('label')

df_author2=pd.DataFrame(dict(x=x_author2, y=y_author2, label=clusters[:num_author2], title=ids[:num_author2],author=author[:num_author2]))
groups_author2 = df_author2.groupby('label')


print("Author 1 is represented by Diamonds")
print("Author 2 is represented by circles")

#Plot using Matplotlib
fig, ax = plt.subplots(figsize=(20, 10)) # set size

#Author 1 books are marked by circles
for name, group in groups_author1:
    ax.plot(group.x, group.y, marker='D', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')

#Author 2 books are marked by diamonds
for name, group in groups_author2:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')

ax.legend(numpoints=1)  #show legend 

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=3)

plt.show()

print("TFIdf Based clustering is now complete.")


########################################################### Doc2Vec based clustering ############################################################
print("\n\n\n\n")
print("Generating new features using Gensim's Doc2Vec")

#Converting documents to tagged document format
taggeddoc=[]
text=[]

for i in range(len(books)):
    t=nltk.word_tokenize(books[i])
    text.append(t)
    td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(t))).split(),tags=[u'Doc_{:d}'.format(i)])
    taggeddoc.append(td)

#Training the Gensim model

numOfFeatures=200 #change if required
model = gensim.models.Doc2Vec(taggeddoc,alpha=0.025, size= numOfFeatures, min_alpha=0.025, min_count=0)
for epoch in range(20):
    if(epoch%5==0):
        print('Now training epoch %s'%epoch)
    model.train(taggeddoc,total_examples=model.corpus_count,epochs=model.iter)
    model.alpha -= 0.002 
    model.min_alpha = model.alpha
    
#Generating features
features=np.zeros((len(books),numOfFeatures))
for i in range(len(books)):
    features[i]=model.docvecs[u'Doc_{:d}'.format(i)]
dist = 1 - cosine_similarity(features)


#Clustering

num_of_clusters=25
kmeans=KMeans(n_clusters=num_of_clusters)
kmeans.fit(features)
clusters=kmeans.labels_.tolist()

documents = { 'title': ids,  'text': books, 'cluster': clusters, 'author':author }
documentFrame = pd.DataFrame(documents, index = [clusters])

#Converting to 2D

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=7)
transformed = mds.fit_transform(dist)
x=transformed[:, 0]
y=transformed[:, 1]

x_author1=transformed[:num_author1,0]
y_author1=transformed[:num_author1,1]

x_author2=transformed[num_author1:,0]
y_author2=transformed[num_author1:,1]


#Generating colours
num_colours=num_of_clusters
col=mpl.colors.cnames.items()
colours={}
random.seed(7)
i=0
while (i<num_of_clusters):
    c=random.choice(list(col))
    if(c[1] not in colours.keys()):
        colours[i]=c[1]
        i=i+1


cluster_names={}
for i in range(num_of_clusters):
    cluster_names[i]="Cluster-"+str(i)


#Prepration for plotting
df = pd.DataFrame(dict(x=x, y=y, label=clusters, title=ids,author=author)) 
groups = df.groupby('label')

df_author1=pd.DataFrame(dict(x=x_author1, y=y_author1, label=clusters[:num_author1], title=ids[:num_author1],author=author[:num_author1]))
groups_author1 = df_author1.groupby('label')

df_author2=pd.DataFrame(dict(x=x_author2, y=y_author2, label=clusters[:num_author2], title=ids[:num_author2],author=author[:num_author2]))
groups_author2 = df_author2.groupby('label')

#Plotting

fig, ax = plt.subplots(figsize=(20, 10)) # set size

for name, group in groups_author1:
    ax.plot(group.x, group.y, marker='D', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')

for name, group in groups_author2:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')

ax.legend(numpoints=1)  #show legend 

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=3)

plt.show()


print("Doc2Vec based clustering is now over.")


################################################## Using Topic Modelling to create Content based features#################################################### 
print("\n\n\n\n")
print("Extracting a document-topic matrix using LDA. Weights in this matrix will be used as features for further clustering.")

#Preparing a corpus

def prepareCorpus(docs):
    text=list()
    for i in range(len(docs)):
        words=nltk.word_tokenize(docs[i])
        text.append(words)
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(t) for t in text]
    return(dictionary,corpus)

dic,corp=prepareCorpus(books)

print("Training LDA. Will take a while...")
numOfTopics=50
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corp, num_topics=numOfTopics, id2word = dic, passes=20)

docsTopicMatrix=np.zeros((len(books),numOfTopics))
for i in range(len(books)):
	topicsList=ldamodel.get_document_topics(corp[i],minimum_probability=0)
	for j in range(len(topicsList)):
		docsTopicMatrix[i,j]=topicsList[j][1]

dist = 1 - cosine_similarity(docsTopicMatrix)

#Clustering

num_of_clusters=25
kmeans=KMeans(n_clusters=num_of_clusters)
kmeans.fit(features)
clusters=kmeans.labels_.tolist()

documents = { 'title': ids,  'text': books, 'cluster': clusters, 'author':author }
documentFrame = pd.DataFrame(documents, index = [clusters])

#Converting to 2D

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=7)
transformed = mds.fit_transform(dist)
x=transformed[:, 0]
y=transformed[:, 1]

x_author1=transformed[:num_author1,0]
y_author1=transformed[:num_author1,1]

x_author2=transformed[num_author1:,0]
y_author2=transformed[num_author1:,1]


#Generating colours
num_colours=num_of_clusters
col=mpl.colors.cnames.items()
colours={}
random.seed(7)
i=0
while (i<num_of_clusters):
    c=random.choice(list(col))
    if(c[1] not in colours.keys()):
        colours[i]=c[1]
        i=i+1


cluster_names={}
for i in range(num_of_clusters):
    cluster_names[i]="Cluster-"+str(i)


#Prepration for plotting
df = pd.DataFrame(dict(x=x, y=y, label=clusters, title=ids,author=author)) 
groups = df.groupby('label')

df_author1=pd.DataFrame(dict(x=x_author1, y=y_author1, label=clusters[:num_author1], title=ids[:num_author1],author=author[:num_author1]))
groups_author1 = df_author1.groupby('label')

df_author2=pd.DataFrame(dict(x=x_author2, y=y_author2, label=clusters[:num_author2], title=ids[:num_author2],author=author[:num_author2]))
groups_author2 = df_author2.groupby('label')

#Plotting

fig, ax = plt.subplots(figsize=(20, 10)) # set size

for name, group in groups_author1:
    ax.plot(group.x, group.y, marker='D', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')

for name, group in groups_author2:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
        label=cluster_names[name], color=colours[name], 
        mec='none')

ax.legend(numpoints=1)  #show legend 

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=3)

plt.show()






















