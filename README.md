# AnalyseGutenberg

### About
This repository might be of use to you if you're a humanities researcher or looking for booking recommendations on Project Gutenberg. This repository does the following:
It allows you to download the works of various authors from Project Gutenberg and stores them into individual directories. Sub-directories for the language of the text are created. This downloading is done using the Python Gutenberg package (https://github.com/c-w/Gutenberg). After this, it performs some basic text clustering on their works based on content as well as style. Currently, you can download the works of multiple authors at a time but analyse only two at a time. Also, the analysis just now is only for English texts. I will change this soon. 

It also allows you to get the recommendations for a book. 

### Usage

**Download.py** will help you download all the works of multiple authors and store them into relevant folders. 

**Analyse-Content.py** will perform the following actions:
It begins by loading the english texts of each author and then breaking up these texts into smaller documents of 1000 words each.It then creates a tf-idf matrix using the works of both the authors. Then a distance matrix is computed (1-cosine similarity) which is used for K-Means clustering. The clusters are then plotted using matplotlib. The next form of analysis is based on gensim's doc2vec, where a feature vector is generated for each document. This is followed by K-Mean clustering. The final step in the content similarity based analysis is to use Topic-Modelling. Using Gensim's LDA, a Document - Topic matrix is constructed, where the i,jth entry in the matrix represents the weight of topic j in document i. These weights associated with each document are used as features for K-means clustering. 

**Analyse-Style.py** will do the following: It loads the english texts of each author and then splits them into small paragraphs or chunks of 25 sentences each. A large number of stylometric features are then extracted from each features which are used for K-Means clustering. Each feature is a separate function in the analyse-style file. Further, a Support Vector Machine classifier is trained on about 70% of the works and tested on the remaining 30%. A large number of incorrect classifications might be an indicator of stylistic similarity. 

I've added a short shell script called execute that runs Download.py, Analyse-Content.py and Analyse-Style.py and deletes the downloaded texts after that. 

The two folders author1 and author2 are how your texts would get downloaded and stored. Delete them before you run the script. 

If you're using the Gutenberg package for the first time, run the inital.py script before you start working. This will take a while. Took about an hour on my computer. For more details on the Gutenberg package you can refer to https://github.com/c-w/Gutenberg .

**recommender.py** will load an existing distance matrix from your computer or calculate a new one if you're using the script for the first time. It will then suggest 7 books which are similar in "content". Before running recommender.py for the first time, you will have to run **downloadBooks.py**. This script downloads all the books from Project Gutenberg and stores them in a folder called "books" in your machine. Currently, the script downloads 56740 books. 

###  Requirements
To use the scripts you would require the following:
1. Python3
2. Gutenberg (python package: https://pypi.python.org/pypi/Gutenberg)
3. Gensim
4. NLTK
5. Numpy
6. Matplotlib
7. Pandas
8. Scikit learn 
9. Sentic Net

This project uses Sentic Net in some of the Stylometric features. For more information on Sentic Net, please refer to: 
*E. Cambria, S. Poria, R. Bajpai, and B. Schuller. SenticNet 4: A semantic resource for sentiment analysis based on conceptual primitives. In: COLING, pp. 2666-2677, Osaka (2016)*
