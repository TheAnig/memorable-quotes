import re
import requests
from bs4 import BeautifulSoup

links_expr = 'quote_title_[a-z][a-z]?\.php'
links_pattr = re.compile(links_expr)

books_expr = 'quote_title_resp\.php\?TName=[A-z 0-9 % , \' ! \- & \. ( ) \? \$]+'
books_pattr = re.compile(books_expr)

remove_end_expr = 'Share this Quote'
remove_end_pattr = re.compile(remove_end_expr)

quotes_count = 0
books_count = 0
letters_count = 1

def find_quotes(qpage):
	global quotes_count
	soup = BeautifulSoup(qpage.text, 'html.parser')
	quotes = soup.find_all('div', class_='share')
	fname = 'litquotes/'+re.sub('\- LitQuotes','',soup.title.get_text()).strip()+'.txt'
	f = open(fname, 'w')
	for quote in quotes:
		quote_string = quote.parent.get_text()
		f.write(remove_end_pattr.sub('',quote_string).encode('utf-8').strip()+'\n')
		quotes_count+=1
	f.close()
		

def find_books(spage):
	global books_count
	books = books_pattr.findall(spage.text)
	for book in books:
		book_page = requests.get(url+book)
		find_quotes(book_page)
		books_count+=1

url = 'http://www.litquotes.com/'
init = 'quote_title.php'
#init2 = 'quote_title_resp.php?TName=20,000%20Leagues%20Under%20The%20Sea'



page = requests.get(url+init)

find_books(page)

links = links_pattr.findall(page.text)

for link in links:
	cur_page = requests.get(url+link)
	find_books(cur_page)
	letters_count+=1

print "Letters Scanned: "+str(letters_count)

print "Books Scanned: "+str(books_count)

print "Quotes Saved: "+str(quotes_count)
