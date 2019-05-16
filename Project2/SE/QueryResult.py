import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring
from nltk.corpus import stopwords

import math
import nltk
from nltk.stem import WordNetLemmatizer
import nltk.tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet


ix = index.open_dir("index")
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')
searcher = ix.searcher(weighting=scoring.ScoringFunction())
parser = QueryParser("contents", schema=ix.schema, group=OrGroup)
stopWords = set(stopwords.words('english'))

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def getSearchEngineResult(query_dict):
    result_dict = {}

    for qid, q in query_dict.items():
        result_dict[qid] = getSingleSearchEngineResult(q, False)
        print()

    return result_dict

def getSingleSearchEngineResult(q, verbose):
    print(q)

    words = list(filter(lambda x: x not in stopWords and x.isalpha(), map(lambda x: x.lower(), q.split(' '))))
    new_q = process_query(words, lemmatizer)
    print(new_q)

    q = parser.parse(new_q.lower())
    results = searcher.search(q, limit=None)

    if verbose:
        for result in results:
            print(result.fields()['docID'], result.score)

    return [(result.fields()['docID'], result.score) for result in results]

def process_query(words: list, lemmatizer) -> str:
    # 만약 쿼리가 그렇게 길지 않다면, synonym은 찾지 않는다.
    if False:
        q = ' '.join(map(lambda x: stemmer.stem(x), words)) + ' '
    else:
        q = ''
        tokens = {}
        pos_tag = nltk.pos_tag(words)
        i = 0
        for word in words:
            i += 1
            # word가 너무 짧으면 무시한다. bigram에서 걸리기를...
            if len(word) < 3:
                continue

            wn_pos = get_wordnet_pos(pos_tag[i - 1][1])
            lemma_word = word
            if not wn_pos == '' and wn_pos is not None:
                lemma_word = lemmatizer.lemmatize(word, pos=wn_pos)
            stemmed_word = stemmer.stem(word)

            freq = max(get_freq(lemma_word), get_freq(stemmed_word), get_freq(word))

            boost_org = 20 / math.pow(freq + 1, 0.28)
            if boost_org < 0:
                continue

            if stemmed_word in tokens:
                tokens[stemmed_word] = max(tokens[stemmed_word], boost_org)
            else:
                tokens[stemmed_word] = boost_org

            if boost_org > 14:
                definition = get_definition(word)
                if len(definition) > 0:
                    def_boost = 8 / len(definition)
                    for def_word in definition:
                        def_freq = get_freq(def_word)
                        stemmed_def_word = stemmer.stem(def_word)
                        def_word_boost = def_boost
                        if def_freq > 0:
                            def_word_boost = def_boost / math.pow(def_freq, 0.25)
                        if def_word_boost > 1:
                            stemmed_def = stemmer.stem(stemmed_def_word)
                            if stemmed_def in tokens:
                                tokens[stemmed_def] = max(tokens[stemmed_def], def_word_boost)
                            else:
                                tokens[stemmed_def] = def_word_boost

        for (token, boost) in tokens.items():
            q += token + '^' + str(boost) + ' '

    if True:
        swords = list(map(lambda x: stemmer.stem(x), words))

        # bigram을 쿼리에 추가한다.
        for i in range(len(swords) - 1):
            boost = 1
            frame = 5
            q += f'"{swords[i]} {swords[i + 1]}"^{boost}~{frame} '

        # trigram을 쿼리에 추가한다.
        for i in range(len(swords) - 2):
            boost = 1
            frame = 5
            q += f'"{swords[i]} {swords[i + 1]} {swords[i + 2]}"^{boost}~{frame} '

        # 4-gram을 쿼리에 추가한다.
        for i in range(len(swords) - 3):
            boost = 1
            frame = 5
            q += f'"{swords[i]} {swords[i + 1]} {swords[i+ 2]} {swords[i + 3]}"^{boost}~{frame} '

    return q


def get_freq(word: str):
    synsets = wordnet.synsets(word)
    if len(synsets) == 0:
        return 0
    return synsets[0].lemmas()[0].count()


def get_definition(word: str):
    synsets = wordnet.synsets(word)
    if len(synsets) == 0:
        return []
    return process_doc(synsets[0].definition())


def get_synonyms(word: str):
    synonyms = []
    synsets = wordnet.synsets(word)
    for synset in synsets:
        synonyms += map(lambda x: stemmer.stem(x), filter(lambda x: '_' not in x, synset.lemma_names()))
    return synonyms


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
    pos_tag = list(filter(lambda x: x[1].startswith('N') or x[1].startswith('V'), pos_tag))

    lemmatized_tokens = map(lambda x: lemmatize(x, lemmatizer), pos_tag)
    lemmatized_tokens = filter(lambda x: x not in ['something', 'someone', 'somebody'], lemmatized_tokens)
    return list(lemmatized_tokens)

