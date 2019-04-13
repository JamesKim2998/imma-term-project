import Zipfs_law as zl


def test_zipf_law01():
    (freq_list, prob_list, rank_list) = zl.zipf_law(
        ['the', 'all', 'some', 'the', 'abc'])
    assert len(freq_list) == 4
    assert len(prob_list) == 4
    assert len(rank_list) == 4
    assert freq_list[0] == 2
    assert freq_list[1] == 1
    assert freq_list[2] == 1
    assert freq_list[3] == 1
    assert prob_list[0] == 2 / 5
    assert prob_list[1] == 1 / 5
    assert rank_list[0] == 1
    assert rank_list[1] == 2
    assert rank_list[2] == 2
    assert rank_list[3] == 2


def test_zipf_law02():
    (freq_list, prob_list, rank_list) = zl.zipf_law(
        ['the', 'all', 'some', 'the', 'abc', 'i', 'i', 'am', 'i', 'am'])
    assert len(freq_list) == 6
    assert len(prob_list) == 6
    assert len(rank_list) == 6
    assert freq_list[0] == 3
    assert freq_list[1] == 2
    assert freq_list[2] == 2
    assert freq_list[3] == 1
    assert freq_list[4] == 1
    assert freq_list[5] == 1
    # assert prob_list[0] == 2 / 5
    # assert prob_list[1] == 1 / 5
    # assert rank_list[0] == 1
    # assert rank_list[1] == 2
    # assert rank_list[2] == 2
    # assert rank_list[3] == 2


def test_plot01():
    rank_list = [1, 2, 3]
    freq_list = [5, 3, 1]

    import numpy as np
    import matplotlib.pyplot as plt
    plt.plot(np.log(rank_list), np.log(freq_list), label='Plot1')
    plt.plot(np.log(rank_list), np.log(freq_list), label='Plot2')
    plt.legend()
    plt.show()
