import re
import requests
from bs4 import BeautifulSoup

links_expr = '="/subjects\/([A-Za-z])\w+\/"'
links_pattr = re.compile(links_expr)


quotes_count = 0
subjects_count = 0

qlist = []

def get_quotes(qpage):
    global quotes_count
    global qlist
    soup = BeautifulSoup(qpage.text, 'html.parser')
    quotes = soup.find_all('dt', class_='quote')
    for quote in quotes:
        qlist.append(quote.get_text())
        quotes_count += 1
    print(quotes_count)
    
url = 'http://www.quotationspage.com/subjects/'

page = requests.get(url)
#linkslist = BeautifulSoup(page.text, parseOnlyThese=SoupStrainer('a'))
soup = BeautifulSoup(page.text, 'html.parser')

sections = soup.find_all('div', class_='subjects')
#linkslist = subjects.findAll('a')

subjects = []

for section in sections:
    linkslist = section.findAll('a')
    for linkitem in linkslist:
        subjects.append(linkitem.text)

subjects.pop()

for subject in subjects:
    sub_page = requests.get(url+subject)
    get_quotes(sub_page)
    
with open('quotes.txt', 'w') as outfile:
    for quote in qlist:
        outfile.write(quote + '\n')
    
