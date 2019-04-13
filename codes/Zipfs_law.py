from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import os
import nltk


def zipf_law(tokenized_words):
    total_word_count = len(tokenized_words)
    word_array = np.array(tokenized_words)
    word_unique, word_count = np.unique(word_array, return_counts=True)
    word_count_dict = dict(zip(word_unique, word_count))

    word_rank_pair_list = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    word_rank_list = list(map(lambda x: x[0], word_rank_pair_list))
    # col_head = "Word".ljust(20) + " Freq".ljust(11) + " r".ljust(7) + "Pr".ljust(20) + "r*Pr"
    # print(col_head)

    freq_list = []
    prob_list = []
    rank_list = []

    rank = 1 # 현재 rank 값.
    for i in range(len(word_count_dict)):
        word = word_rank_list[i]
        freq = word_count_dict[word]
        ocur_prob = freq / total_word_count
        # freq가 같은 경우 rank도 같아야 하지만 현재 word_rank_list의 index는 1씩 증가하며 모두 다른 값을 가지므로
        # 바로 rank = i + 1을 적용할 수 없음
        # 만약 이전 rank의 단어가 현재 단어와 동일한 freq를 같는다면, 이전 rank를 그대로 유지한다.
        if i > 0 and freq_list[i - 1] == freq:
            pass
        else:
            rank = i + 1

        # if using Unigram
        # if use_unigram:
        #     print(word.ljust(20), str(freq).ljust(10), str(rank).ljust(5), '%.17f' % ocur_prob, '%.6f' % (rank * ocur_prob))
        # if using Bi, Tri-gram
        # else:
        #     print(word, str(freq), str(rank), '%.17f' % ocur_prob, '%.6f' % (rank * ocur_prob))

        freq_list.append(freq)
        prob_list.append(ocur_prob)
        rank_list.append(rank)

    return freq_list, prob_list, rank_list


if __name__ == "__main__":
    with open("bible.txt", 'r') as f:
        text = f.read()
    s_list = ['.', ',', '?', '!', ';', ':', '\'s', '(', ')', '“', '”', '’', '=', '>', '+', '<', '&', '#', '·', '&', '←']
    for c in s_list:
        text = text.replace(c, '')
    # 소문자로 변환
    text = text.lower()

    # Setup environments.
    if not os.path.isdir('output'):
        os.mkdir('output')

    # Configure pyplot.
    plt.xlabel('Log(Rank)')
    plt.ylabel('Log(Frequency)')


    """3-1"""
    # zipf's law plot (Represent X,Y axis name!)
    tokenized_words = word_tokenize(text)           # 다른 tokenizer 사용 가능
    freq_list, _, rank_list = zipf_law(tokenized_words)
    plt.plot(np.log(rank_list), np.log(freq_list), label='Unigram')
    plt.savefig('output/unigram.png')


    """3-2"""
    bigram_list = list(nltk.bigrams(tokenized_words))
    freq_list, _, rank_list = zipf_law(bigram_list)
    plt.plot(np.log(rank_list), np.log(freq_list), label='Bigram')

    trigram_list = list(nltk.trigrams(tokenized_words))
    freq_list, _, rank_list = zipf_law(trigram_list)
    plt.plot(np.log(rank_list), np.log(freq_list), label='Trigram')

    plt.legend()
    plt.savefig('output/benchmark.png')
