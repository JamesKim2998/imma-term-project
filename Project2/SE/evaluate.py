import numpy as np
import json
from QueryResult import getSearchEngineResult, getSingleSearchEngineResult


def readQueryFile(filename):
    # query json(dict)
    #   key : query id
    #   value : query content
    with open(filename) as query_data:
        query = json.load(query_data)
    return query


def readRelevanceFile(filename):
    # relevance json(dict)
    #   key : query id
    #   value : relevant document index list
    with open(filename) as relevance_data:
        relevance = json.load(relevance_data)
    return relevance


def evaluate(query_dict, relevent_dict):
    BPREF = []

    for queryID in query_dict.keys():
        score = evaluateSingleQuery(query_dict[queryID], relevent_dict[queryID], False)
        print(f'queryID: {queryID} score: {score}')
        BPREF.append(score)

    print(np.mean(BPREF))


def evaluateSingleQuery(query, relevent_docs, verbose):
    relevantCount = 0
    nonRelevantCount = 0
    score = 0
    results = getSingleSearchEngineResult(query, False)
    relDocCount = len(relevent_docs)

    for (document, documentScore) in results:
        if document in relevent_docs:
            relevantCount += 1
            if nonRelevantCount >= relDocCount:
                if verbose:
                    print(f'TP: document={document} score={documentScore} but no BPREF')
                score += 0
            else:
                if verbose:
                    print(f'TP: document={document} score={documentScore}')
                score += (1 - nonRelevantCount / relDocCount)
        else:
            if verbose:
                print(f'FP: document={document} score={documentScore}')
            nonRelevantCount += 1
        if relevantCount == relDocCount:
            break
    score = score / relDocCount
    return score


def main():
    query_dict = readQueryFile('doc/query.json')
    relevant_dict = readRelevanceFile('doc/relevance.json')

    hard_query = None
    # hard_query = 'methods control type ii diabetes'
    # hard_query = 'information pre 1500 history inuit eskimo people'
    # hard_query = 'regulations considerations concerning registering freighter country'

    single_query = None
    # single_query = 701

    if hard_query is not None:
        results = getSingleSearchEngineResult(hard_query, False)
        for result in results[:20]:
            print(result)
    elif single_query is None:
        evaluate(query_dict, relevant_dict)
    else:
        query = query_dict[str(single_query)]
        relevant_doc = relevant_dict[str(single_query)]
        score = evaluateSingleQuery(query, relevant_doc, False)
        print(f'queryID: {single_query} score: {score}')


if __name__ == '__main__':
    main()
