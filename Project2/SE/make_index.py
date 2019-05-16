import os.path
import json
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, NUMERIC

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.stem.snowball import SnowballStemmer

def should_preserve_token(token: str) -> bool:
    word = token[0]
    if not word.isalpha():
        return False

    pos_tag = token[1]
    if not pos_tag.isalpha(): return False
    if pos_tag in ['IN', 'CD', 'DT', 'CC', 'TO', 'MD', 'EX', 'UH']: return False
    if pos_tag.startswith('W'): return False
    if pos_tag.startswith('P'): return False
    return True


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    elif pos_tag.startswith('FW') or pos_tag.startswith('SYM'):
        return None
    else:
        return None

def lemmatize(pos_tag, lemmatizer):
    wordnet_pos = get_wordnet_pos(pos_tag[1])
    if wordnet_pos != None:
        return lemmatizer.lemmatize(pos_tag[0], wordnet_pos)
    else:
        return pos_tag[0]


def process_doc(doc: str):
    tokens = nltk.word_tokenize(doc.lower(), preserve_line=False)

    pos_tag = nltk.pos_tag(tokens)
    pos_tag = list(filter(lambda x: should_preserve_token(x), pos_tag))
    # print(pos_tag)

    # lemmatized_tokens = map(lambda x: lemmatize(x, lemmatizer), pos_tag)
    stemmed_tokens = map(lambda x: stemmer.stem(x[0]), pos_tag)
    # stemmed_tokens = map(lambda x: stemmer.stem(x), lemmatized_tokens)
    processed_doc = ' '.join(list(stemmed_tokens))
    # print(processed_doc)

    return processed_doc

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language = 'english')

schema = Schema(docID=NUMERIC(stored=True), contents=TEXT)
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

# ix = open_dir(index_dir)
# print(len(list(ix.searcher().documents())))
# doc = list(filter(lambda d: d['docID'] == 9501, ix.searcher().documents()))[0]
# print(doc['contents'])
# exit()

ix = create_in(index_dir, schema)

writer = ix.writer()
with open('doc/document.json') as doc_data:
    doc_list = json.load(doc_data)

    # processed_doc = process_doc(doc_list['9470'])
    # print(processed_doc)
    # exit()

    for doc_idx, doc in doc_list.items():
        docID = int(doc_idx)
        processed_doc = process_doc(doc)
        print(processed_doc)
        writer.add_document(docID=docID, contents=processed_doc)

writer.commit()