import concurrent.futures
import csv
import json
import math
import pickle
import re
import time

import nltk
from nltk.corpus import stopwords

import searchBody
import pandas as pd

nltk.download('stopwords')
from nltk.stem import PorterStemmer
import search_backend
import evaluations
from flask import Flask, request, jsonify


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        # storage_client = storage.Client.from_service_account_json(
        #     r"C:\Users\liron\OneDrive\Desktop\לימודים\שנה ג\אחזור מידע\פרוייקט\ir_proj\assignment-3-334118-bba4d4bd1179.json")
        # blobs = storage_client.list_blobs("ir-course-ass-3", prefix="ir_proj/s/postings_gcp_small/")
        # bucket = storage_client.get_bucket("ir-course-ass-3")
        # for b in blobs:
        #     print(b)
        # file = bucket.blob("ir_proj/s/postings_gcp_small/index.pkl")
        # file = file.download_as_string()
        # file = pickle.load(file)
        # print(pickle.load(open(file.download_as_file(), 'rb')))
        # with open(r"postings_gcp_body\index.pkl", "rb") as f:
        #     self.indexBody = pickle.load(f)
        # with open(r"postings_gcp_body_stem/index.pkl", "rb") as f:
        #     self.indexSmallBody = pickle.load(f)
        # with open(r"postings_gcp_title_stem/index.pkl", "rb") as f:
        #     self.indexSmallTitle = pickle.load(f)
        with open(r"postings_gcp_anchor\index.pkl", "rb") as f:
            self.indexAnchor = pickle.load(f)
        # with open(r"postings_gcp_title\index.pkl", "rb") as f:
        #     self.indexTitle = pickle.load(f)
        # with open(r"DL_body.pkl", "rb") as f:
        #     self.DL_body = pickle.load(f)
        # with open(r"DL_title.pkl", "rb") as f:
        #     self.DL_title = pickle.load(f)
        with open(r"IdTitle.pkl", "rb") as f:
            self.IdTitle = pickle.load(f)
        # with open(r"pageviews.pkl", "rb") as f:
        #     self.pageviews = pickle.load(f)
        # with open('pageRank.csv', mode='r') as infile:
        #     reader = csv.reader(infile)
        #     self.page_rank = {int(rows[0]): math.log(float(rows[1]), 10) for rows in reader}
        # self.page_rank = pd.read_csv('pageRank.csv', header=None)
        # self.page_rank.columns = ["id", "pageRank"]
        # self.page_rank.set_index("id", inplace=True)
        # self.bm25_title = search_backend.BM25_from_index(self.indexSmallTitle, self.DL_title, "title_stem",
        #                                                  self.page_rank, k1=1.2, b=0.75)
        # self.bm25_text = search_backend.BM25_from_index(self.indexSmallBody, self.DL_body, "body_stem", self.page_rank,
        #                                                 k1=1.2, b=0.75)
        self.stemmer = PorterStemmer()
        self.executor = concurrent.futures.ThreadPoolExecutor(2)
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    # YOUR CODE HERE
    tokens = [tok for tok in tokens if tok not in english_stopwords and tok not in corpus_stopwords]
    return (tokens)


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    timeS = time.time()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query = tokenize(query)
    query = [app.stemmer.stem(term) for term in query]
    # print(query)
    future_body = app.executor.submit(app.bm25_text.search, {1: query}, 100)
    future_title = app.executor.submit(app.bm25_title.search, {1: query}, 100)
    res = search_backend.merge_results(future_body.result(), future_title.result(), N=100)
    # bm25_queries_score_test_title = app.bm25_title.search({1: query}, N=100)
    # bm25_queries_score_test_text = app.bm25_text.search({1: query}, N=100)
    # res = search_backend.merge_results(bm25_queries_score_test_title, bm25_queries_score_test_text,
    #                                    title_weight=0.5,
    #                                    text_weight=0.5, N=100)
    print(time.time() - timeS)
    return jsonify([(int(doc_id), app.IdTitle[doc_id]) for doc_id, score in res[1]])


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    res = searchBody.get_topN_score_for_queries({1: tokenize(query)}, app.indexSmall, app.DL_body, app.IdTitle, "small",
                                                N=100)
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    res = searchBody.generate_relevant_dict_title(tokenize(query), app.indexTitle, app.IdTitle, "title")
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    res = searchBody.generate_relevant_dict_anchor(tokenize(query), app.indexAnchor, app.IdTitle, "anchor")
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    for doc_id in wiki_ids:
        res.append(app.page_rank.loc[doc_id])
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    for id in wiki_ids:
        res.append(app.pageviews[id])
    return jsonify(res)


@app.route("/evaluate")
def get_evaluate():
    # parameters_list = []
    # weights_list = []
    # for k in [round(x * 0.1, 1) for x in range(5, 21)]:
    #     for b in [round(y * 0.1, 1) for y in range(3, 10)]:
    #         parameters_list.append((k, b))
    # # for weight_body in [round(z * 0.05, 2) for z in range(101)]:
    # #     weights_list.append((weight_body, round(1-weight_body, 2)))
    # for weight_body in [round(z * 0.1, 1) for z in range(11)]:
    #     weights_list.append((weight_body, round(1 - weight_body, 1)))
    parameters_list = [(1.8, 0.3), (1.2, 0.75), (1.3, 0.8), (1.2, 0.8), (0.9, 0.3), (1.4, 0.75), (1.7, 0.85)]
    weights_list = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.35, 0.65), (0.4, 0.6), (0.3, 0.7)]
    N = 40
    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)
    query_dict = {}
    for i in enumerate(queries.keys()):
        query_dict[i[0]] = [app.stemmer.stem(j) for j in tokenize(i[1])]
    relevance_dict = dict(enumerate(queries.values()))
    grid_models = evaluations.grid_search_models(query_dict, relevance_dict, app.DL_body, "body_stem", app.DL_title,
                                                 "title_stem", parameters_list, weights_list, N, app.indexSmallTitle,
                                                 app.indexSmallBody, app.IdTitle, app.page_rank)
    print(grid_models)
    ret = {}
    k1, b, title_w, body_w = max(grid_models, key=grid_models.get)
    for k1, b, titlw_w, body_w in grid_models.keys():
        bm25_title = search_backend.BM25_from_index(app.indexSmallTitle, app.DL_title, "title_stem", app.page_rank, k1=k1,
                                                    b=b)
        bm25_text = search_backend.BM25_from_index(app.indexSmallBody, app.DL_body, "body_stem", app.page_rank, k1=k1, b=b)
        bm25_queries_score_test_title = bm25_title.search(query_dict, N=N)
        bm25_queries_score_test_text = bm25_text.search(query_dict, N=N)
        merge_res = search_backend.merge_results(bm25_queries_score_test_title, bm25_queries_score_test_text, app.IdTitle,
                                                 title_weight=title_w,
                                                 text_weight=body_w, N=N)
        test_metrices = evaluations.evaluate(dict(enumerate(queries.values())), merge_res, k=N, print_scores=False)
        ret[(k1, b, titlw_w, body_w)] =  test_metrices['MAP@k']
        print((k1, b, titlw_w, body_w), test_metrices['MAP@k'])
    # k_values = [1, 3, 5, 10, 20, 30, 50, 70, 100]
    # metrices_names = ['MAP@k']
    # statistics_of_differnt_k = evaluations.plot_metric_with_differnt_k_values(dict(enumerate(queries.values())), merge_res,
    #                                                               metrices_names, k_values=k_values)
    return jsonify(ret)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
