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
from xgoogle.search import GoogleSearch

ix = index.open_dir("index")
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')
searcher = ix.searcher(weighting=scoring.ScoringFunction())
parser = QueryParser("contents", schema=ix.schema, group=OrGroup)
stopWords = set(stopwords.words('english'))


def getSearchEngineResult(query_dict):
    result_dict = {}

    for qid, q in query_dict.items():
        result_dict[qid] = getSingleSearchEngineResult(q, False)
        print()

    return result_dict


def getSingleSearchEngineResult(q, verbose):
    print(q)

    words = list(filter(lambda x: x not in stopWords and x.isalpha(), map(lambda x: x.lower(), q.split(' '))))
    new_q = process_query(words)
    print(new_q)

    q = parser.parse(new_q.lower())
    results = searcher.search(q, limit=None)

    if verbose:
        for result in results:
            print(result.fields()['docID'], result.score)

    return [(result.fields()['docID'], result.score) for result in results]


def process_query(words: list) -> str:
    # 만약 쿼리가 그렇게 길지 않다면, synonym은 찾지 않는다.
    q = ''
    pos_tag = nltk.pos_tag(words)
    query_words = {}

    def add_query_word(stem_: str, boost_: float):
        if stem_ in query_words:
            query_words[stem_] = max(query_words[stem_], boost_)
        else:
            query_words[stem_] = boost_

    for i in range(len(words)):
        # word가 너무 짧으면 무시한다. n-gram에서 걸리기를...
        word = words[i]
        if len(word) < 3:
            continue

        wn_pos = get_wn_pos(pos_tag[i - 1][1])
        word_set = create_word_set(word, wn_pos)
        word_boost = calculate_query_word_boost(word_set)
        if word_boost <= 0:
            continue

        stem = word_set.stem
        add_query_word(stem, word_boost)

        if word_boost > 14:
            def_boosts = calculate_definition_boost(word_set.lemma)
            for (stem, boost) in def_boosts.items():
                add_query_word(stem, boost)

    for (stem, boost) in query_words.items():
        q += stem + '^' + str(boost) + ' '

    # n-gram을 쿼리에 추가한다.
    stemms = list(map(lambda x: stemmer.stem(x), words))
    q += generate_ngram_boost(stemms, 2)
    q += generate_ngram_boost(stemms, 3)
    q += generate_ngram_boost(stemms, 4)

    return q


class WordSet(object):
    __slots__ = "word", "lemma", "stem", "wn_pos"

    word: str
    lemma: str
    stem: str
    wn_pos: str

    def __init__(self, word: str, lemma: str, stem: str, wn_pos: str):
        self.word = word
        self.lemma = lemma
        self.stem = stem
        self.wn_pos = wn_pos


def get_wn_pos(pos_tag: str) -> str:
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def create_word_set(word: str, wn_pos: str) -> WordSet:
    lemma = word
    if wn_pos is not None and wn_pos != '':
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
    stem = stemmer.stem(word)
    # TODO
    # stem = stemmer.stem(lemma)
    return WordSet(word, lemma, stem, wn_pos)


def get_freq(word: str):
    synsets = wordnet.synsets(word)
    if len(synsets) == 0:
        return 0
    return synsets[0].lemmas()[0].count()


def get_adjusted_freq(word_set: WordSet):
    return max(get_freq(word_set.word), get_freq(word_set.lemma), get_freq(word_set.stem))


def preprocess_text(text: str) -> list:
    def should_preserve_token(pos_tag: str) -> bool:
        if not pos_tag.isalpha(): return False
        if pos_tag in ['IN', 'CD', 'DT', 'CC', 'TO', 'MD', 'EX', 'UH']: return False
        if pos_tag.startswith('W'): return False
        if pos_tag.startswith('P'): return False
        return True

    tokens = nltk.word_tokenize(text.lower(), preserve_line=False)
    pos_tags = nltk.pos_tag(tokens)
    word_sets = []
    for (word, pos_tag) in pos_tags:
        # if not (pos_tag.startswith('N') or pos_tag.startswith('V')):
        #     continue
        if not should_preserve_token(pos_tag):
            continue
        wn_pos = get_wn_pos(pos_tag)
        word_set = create_word_set(word, wn_pos)
        word_sets.append(word_set)
    return word_sets


def calculate_query_word_boost(word_set: WordSet):
    freq = get_adjusted_freq(word_set)
    boost = 20 / math.pow(freq + 1, 0.28)
    return max(boost, 0)


def calculate_definition_boost(lemma: str) -> dict:
    # 정의를 얻어옵니다.
    synsets = wordnet.synsets(lemma)
    if len(synsets) == 0:
        return {}
    definition_text = synsets[0].definition()

    definition = preprocess_text(definition_text)
    # 정의에 자주 등장하는 something, someone, somebody를 제거합니다.
    definition = list(filter(lambda x: x.word not in ['something', 'someone', 'somebody'], definition))

    if len(definition) == 0:
        return {}

    result = {}
    boost_base = 8 / len(definition)
    for word_set in definition:
        freq = get_freq(word_set.word)
        word_boost = boost_base
        if freq > 0:
            word_boost = boost_base / math.pow(freq, 0.25)
        if word_boost < 1:
            continue

        stem = word_set.stem
        if stem in result:
            result[stem] = max(result[stem], word_boost)
        else:
            result[stem] = word_boost

    return result


def generate_ngram_boost(tokens: list, count: int) -> str:
    q = ''
    boost = 1
    frame = 5
    for i in range(len(tokens) - count + 1):
        ngram = '"'
        for j in range(count):
            ngram += tokens[i + j] + ' '
        q += ngram[:-1] + f'"^{boost}~{frame} '
    return q


def get_google_result(q: str):
    lemmas = []
    try:
        gs = GoogleSearch(q)
        gs.results_per_page = 25
        results = gs.get_results()
        for res in results:
            lemmas += preprocess_text(res.title)
            if res.desc and res.desc.text:
                lemmas += preprocess_text(res.desc.text)
    except Exception as e:
        print(e)
    return lemmas
