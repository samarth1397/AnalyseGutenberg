from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
from collections import OrderedDict
import os


print("Hello. Welcome to the Gutenberg Analyser.")
print("We begin by downloading the relevant texts for each author.")
print("\n\n\n")


print("Enter the number of authors whose works you want to download: ")
n=int(input())

for j in range(n):
	print("Enter the name of the author. Please make sure that string that you enter matches the author name on Project Gutenberg exactly")
	author=input()
	print("Name entered by you is: ",author)

	print("Loading books.....")
	originalList=(get_etexts('author', author))
	dictionaryOfNames=OrderedDict() #contains names of the books and language of the book 
	listOfTexts=[]  #contains book number
	
	for i in originalList:
		try: 
		    text = strip_headers(load_etext(i)).strip()
		    title=set(get_metadata('title', i))
		    lanugage=set(get_metadata('language', i)) 
		    dictionaryOfNames[title.pop()]=lanugage.pop()
		    listOfTexts.append(i)
		except:
		    pass
		    #print("error found in download number",i)

	
	print("Writing books to text files.......")
	
	
	keys=list(dictionaryOfNames.keys())
	for i in range(len(keys)):
		#get file name
		key=keys[i]
		#create directory
		file_path = "author"+str(j+1)+"/"+dictionaryOfNames[key]+"/"+key            #authorN/language/filename
		directory = os.path.dirname(file_path)
		if not os.path.exists(directory):
		    os.makedirs(directory)
		#write to file
		text = strip_headers(load_etext(listOfTexts[i])).strip()
		f=open("author"+str(j+1)+"/"+dictionaryOfNames[key]+"/"+"A"+str(j+1)+"-"+key,"w+")
		f.write(text)
		f.close()
		
		
	print("Process completed for author: ",j+1)

