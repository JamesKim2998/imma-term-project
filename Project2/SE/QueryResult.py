import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring
from nltk.corpus import stopwords

import math
import nltk
from nltk.stem import WordNetLemmatizer
import nltk.tokenize
from nltk.corpus import wordnet
from xgoogle.search import GoogleSearch

ix = index.open_dir("index")
lemmatizer = WordNetLemmatizer()
searcher = ix.searcher(weighting=scoring.ScoringFunction())
parser = QueryParser("contents", schema=ix.schema, group=OrGroup)
stopWords = set(stopwords.words('english'))


def getSearchEngineResult(query_dict):
    result_dict = {}

    for qid, q in query_dict.items():
        result_dict[qid] = getSingleSearchEngineResult(q, False)
        print()

    return result_dict


def getSingleSearchEngineResult(query, verbose):
    print(query)
    processed_query = process_query(query)
    print(processed_query)

    q = parser.parse(processed_query)
    results = searcher.search(q, limit=None)

    if verbose:
        for result in results:
            print(result.fields()['docID'], result.score)

    return [(result.fields()['docID'], result.score) for result in results]


def process_query(query: str) -> str:
    query_words = {}

    def add_query_word(lemma_: str, boost_: float):
        if lemma_ in query_words:
            query_words[lemma_] = max(query_words[lemma_], boost_)
        else:
            query_words[lemma_] = boost_

    def plus_query_word(lemma_: str, boost_: float):
        if lemma_ in query_words:
            query_words[lemma_] += boost_
        else:
            query_words[lemma_] = boost_

    tokens = map(lambda x: x.lower(), query.split(' '))
    words = list(filter(lambda x: x not in stopWords and x.isalpha(), tokens))
    wn_poses = list(map(lambda x: get_wn_pos(x[1]), nltk.pos_tag(words)))

    word_sets = []
    for i in range(len(words)):
        word_sets.append(create_word_set(words[i], wn_poses[i]))

    for word_set in word_sets:
        # word가 너무 짧으면 무시한다. n-gram에서 걸리기를...
        word = word_set.word
        if len(word) < 3:
            continue

        word_boost = calculate_query_word_boost(word_set)
        lemma = word_set.lemma
        add_query_word(lemma, word_boost)

        if word_boost > 15:
            def_boosts = calculate_definition_boost(word_set.lemma)
            for (def_lemma, def_boost) in def_boosts.items():
                add_query_word(def_lemma, def_boost)

    # 구글 검색결과를 이용해서 query expansion을 한다.
    if True:
    # if False:
        try:
            google_result = get_google_result(query)
            google_boost = calculate_google_boost(google_result)
            for (lemma, boost) in google_boost.items():
                add_query_word(lemma, boost)
        except Exception as e:
            print(e)

    q = ''
    for (lemma, boost) in query_words.items():
        q += lemma + '^' + str(boost) + ' '

    # n-gram을 쿼리에 추가한다.
    q += generate_ngram_boost(word_sets, 2)
    q += generate_ngram_boost(word_sets, 3)
    q += generate_ngram_boost(word_sets, 4)

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
    # stem = stemmer.stem(word)
    return WordSet(word, lemma, None, wn_pos)


freq_cache = {}


def get_freq(lemma: str):
    if lemma in freq_cache:
        return freq_cache[lemma]

    synsets = wordnet.synsets(lemma)
    if len(synsets) == 0:
        return 0.5  # 0.8번은 등장한 것으로 취급.

    count = 0
    for synset in synsets:
        synset_lemmas = synset.lemmas()
        for synset_lemma in synset_lemmas:
            count += synset_lemma.count()
    if count == 0:
        return 0.5
    freq_cache[lemma] = count
    return count


def preprocess_text(text: str, filter_pos: bool) -> list:
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
        if word in stopWords:
            continue
        if not word.isalpha():
            continue
        if filter_pos:
            if not should_preserve_token(pos_tag):
                continue
        wn_pos = get_wn_pos(pos_tag)
        word_set = create_word_set(word, wn_pos)
        word_sets.append(word_set)
    return word_sets


def calculate_query_word_boost(word_set: WordSet):
    freq = get_freq(word_set.lemma)
    return 7 / math.pow(freq, 0.28)


def calculate_definition_boost(lemma: str) -> dict:
    # 정의를 얻어옵니다.
    synsets = wordnet.synsets(lemma)
    if len(synsets) == 0:
        return {}
    definition_text = synsets[0].definition()

    definition = preprocess_text(definition_text, True)
    # 정의에 자주 등장하는 something, someone, somebody를 제거합니다.
    definition = list(filter(lambda x: x.word not in ['something', 'someone', 'somebody'], definition))

    if len(definition) == 0:
        return {}

    result = {}
    boost_base = 4 / len(definition)
    for word_set in definition:
        freq = get_freq(word_set.lemma)
        word_boost = boost_base / math.pow(freq, 0.5)
        if word_boost < 1:
            continue

        lemma = word_set.lemma
        if lemma in result:
            result[lemma] += word_boost
        else:
            result[lemma] = word_boost

    return result


def get_google_result(q: str):
    word_sets = []
    try:
        gs = GoogleSearch(q)
        gs.results_per_page = 10
        results = gs.get_results()
        for res in results:
            # word_sets += preprocess_text(res.title, True)
            if res.desc and res.desc.text:
                word_sets += preprocess_text(res.desc.text, True)
    except Exception as e:
        print(e)
    return word_sets


def calculate_google_boost(word_sets: list) -> dict:
    if len(word_sets) == 0:
        return {}

    word_count = {}
    for word_set in word_sets:
        lemma = word_set.lemma
        if lemma in word_count:
            word_count[lemma] += 1
        else:
            word_count[lemma] = 1

    results = {}
    boost_base = 2
    for (lemma, count) in word_count.items():
        freq = get_freq(lemma)
        if count / freq < 0.04:
            continue
        word_boost = boost_base * math.pow(count / freq, 0.28)
        results[lemma] = word_boost

    return results


def generate_ngram_boost(word_sets: list, count: int) -> str:
    q = ''
    boost = 1
    frame = 5
    for i in range(len(word_sets) - count + 1):
        ngram = '"'
        for j in range(count):
            ngram += word_sets[i + j].lemma + ' '
        q += ngram[:-1] + f'"^{boost}~{frame} '
    return q
