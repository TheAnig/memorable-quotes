# Deep Learning Methods for Quotable Text
The repository for the "Deep Learning Methods for Quotable Text" article published in my blog [here]().

## Repository Structure
```
The base directory contains the model and the code required to train it
data/ contains the quotes, and GloVe embeddings
scraping/ contains all the code for acquiring the relevant dataset
reference-papers/ contains a copy of the papers and some additional reading materials I had referenced in the post
```

## Data Sources

1. [LitQuotes.com - Over 2800 Literary Quotes website](http://litquotes.com/)
2. [QuotationsPage.com - Quotes and Famous Sayings](http://www.quotationspage.com/)
3. [You had me at hello: How phrasing affects memorability, Cristian Danescu-Niculescu-Mizil, Justin Cheng, Jon Kleinberg and Lillian Lee, Proceedings of ACL, 2012](https://www.cs.cornell.edu/~cristian/memorability.html)
4. ["Echoes of Persuasion: The Effect of Euphony in Persuasive Communication" by Guerini, M., Gozde, O., & Strapparava, C. HLT-NAACL, page 1483-1493. The Association for Computational Linguistics (2015).](https://github.com/marcoguerini/paired_datasets_for_persuasion)
5. [News Aggregator Dataset](https://www.kaggle.com/uciml/news-aggregator-dataset/data)
6. [Project Gutenberg](https://www.gutenberg.org)

## Dependencies

```
keras
tensorflow (CUDA and tensorflow-gpu for GPU training)
nltk/spacy (for Dataset preprocessing and sentence pairing)
BeautifulSoup (for scraping the websites for quotes)
```

## Usage


### Data Collection
```
python litquotes_scraper.py
python quotationspage_scraper.py

cd litquotes/
copy /b *.txt ../quotes.txt
cd ../quotationspage
copy /b *.txt ../quotes.txt
```

### Training
Configure the parameters in configuration.py

then run
```
python main.py
```
