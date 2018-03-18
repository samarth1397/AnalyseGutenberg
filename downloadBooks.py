from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata

# Script that downloads all the books from Project Gutenberg onto your local machine
# All the files will be stored in the ./books repository. The format of the file will be etextNumber-Title

for i in range(1,56740):
	try:
		if i%50==0:
			print(i)
		# try:
		text = strip_headers(load_etext(i)).strip()
		title=set(get_metadata('title', i))
		language=set(get_metadata('language', i))
		lang=language.pop()
		if lang=='en':
			tl=title.pop()
			f=open("./books/"+str(i)+"-"+tl,"w+") 
			f.write(text)
			f.close()
	except:
		pass	
