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
    # single_query = 704
    # single_query = 743  # "regulations considerations concerning registering freighter country"
    # single_query = 847  # portugal
    # single_query = 837  # eskimo, inuit -> 8893, 8894
    # queryID: 701 score: 0.13541666666666669
    # queryID: 712 score: 0.0625
    # queryID: 716 score: 0.24131944444444445
    # queryID: 727 score: 0.29733882030178316
    # queryID: 740 score: 0.19753086419753085
    # queryID: 743 score: 0.11111111111111112
    # queryID: 750 score: 0.0
    # queryID: 763 score: 0.15625
    # queryID: 811 score: 0.25925925925925924
    # queryID: 822 score: 0.1875
    # queryID: 825 score: 0.15976331360946747

    # queryID: 729 score: 0
    # single_query = 702
    # single_query = 712

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
