import math
from contextlib import closing

import numpy as np
from operator import itemgetter

from inverted_index_gcp import MultiFileReader


def get_candidate_documents(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    list of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = []
    for term in np.unique(query_to_search):
        if term in words:
            current_list = (pls[words.index(term)])
            candidates += current_list
    candidates = [i[0] for i in candidates]
    return np.unique(candidates)


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, DL, index_type, page_rank, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.bucketName = ""
        self.index_type = index_type
        self.page_rank = page_rank
        self.DL = DL
        #self.words, self.pls = zip(*self.index.posting_lists_iter(index_type))

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, queries, N=3):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        unique_terms_in_queries = []
        for l in queries.values():
            unique_terms_in_queries += l
        idf = self.calc_idf(list(set(unique_terms_in_queries)))
        self.idf = idf
        scores = {}
        term_frequencies_dict = {}
        for query in queries.items():
            words_pls = self.get_posting_gen(query[1])
            words = tuple(words_pls.keys())
            pls = tuple(words_pls.values())
            for term in query[1]:
                if term in self.index.df:
                    term_frequencies_dict[term] = dict(pls[words.index(term)])
            candidate = get_candidate_documents(query[1], self.index, words, pls)
            scores[query[0]] = sorted([(doc_id, self._score(query[1], doc_id, term_frequencies_dict) + (self.page_rank[doc_id])) for doc_id in candidate],
                                      key=itemgetter(1), reverse=True)[:N]
        return scores

    def _score(self, query, doc_id, term_frequencies_dict):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        # if doc_id not in DL.keys():
        #     return -math.inf
        doc_len = self.DL[doc_id]
        for term in query:
            if doc_id in term_frequencies_dict[term].keys():
                freq = term_frequencies_dict[term][doc_id]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + (self.b * doc_len / self.AVGDL))
                score += (numerator / denominator)
        return score

    def read_posting_list(self, index, w):
        TUPLE_SIZE = 6
        TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
        with closing(MultiFileReader()) as reader:
            locs = index.posting_locs[w]
            b = reader.read(locs, index.df[w] * TUPLE_SIZE, self.index_type)
            posting_list = []
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                if(tf/self.DL[doc_id] > 1/200):
                    posting_list.append((doc_id, tf))
        return posting_list

    def get_posting_gen(self, query):
        """
        This function returning the generator working with posting list.

        Parameters:
        ----------
        index: inverted index
        """
        w_pls_dict = {}
        for term in query:
            temp_pls = self.read_posting_list(self.index, term)
            w_pls_dict[term] = temp_pls
        # words, pls = zip(*index.posting_lists_iter("title"))
        return w_pls_dict


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE

    mergeDict = {}
    for query in title_scores.keys():
        res = []
        for title_pair in title_scores[query]:
            found = False
            doc_id = title_pair[0]
            score = title_weight * title_pair[1]
            for body_pair in body_scores[query]:
                if (body_pair[0] == doc_id):
                    found = True
                    score += text_weight * body_pair[1]
                    res.append((doc_id, score))
            if (found == False):
                res.append((doc_id, score))

        temp_id = [res[i][0] for i in range(len(res))]
        if (query in body_scores):
            for body_pair in body_scores[query]:
                score = 0
                doc_id2 = body_pair[0]
                if (doc_id2 not in temp_id):
                    score = text_weight * body_pair[1]
                    res.append((doc_id2, score))
        res = sorted(res, key=lambda x: x[1], reverse=True)[:N]
        mergeDict[query] = res
        # mergeDict[query] = [(int(doc_id), IdTitle[doc_id]) for doc_id, score in mergeDict[query]]
    # queries_id = [i for i in mergeDict.keys()]
    # for query1 in body_scores.keys():
    #     res1 = []
    #     score1 = 0
    #     if (query1 not in queries_id):
    #         for body_pair1 in body_scores[query1]:
    #             score = 0
    #             doc_id1 = body_pair1[0]
    #             score = text_weight * body_pair1[1]
    #             res1.append((doc_id1, score))
    #         res1 = sorted(res1, key=lambda x: x[1], reverse=True)[:N]
    #         mergeDict[query1] = res1
    # return [(str(doc_id), IdTitle[doc_id]) for doc_id, score in list(mergeDict.values())[0]]
    return mergeDict