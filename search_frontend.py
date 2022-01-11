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
from google.cloud import storage
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import search_backend
import evaluations
from flask import Flask, request, jsonify


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        bucket_name = "ir-course-ass-3"
        client = storage.Client()
        self.bucket = client.bucket(bucket_name)
        blobs = client.list_blobs(bucket_name)
        for blob in blobs:
            if blob.name == "postings_gcp_body/index.pkl":
                with blob.open("rb") as f:
                    self.indexBody = pickle.load(f)
            if blob.name == "postings_gcp_title/index.pkl":
                with blob.open("rb") as f:
                    self.indexTitle = pickle.load(f)
            if blob.name == "postings_gcp_anchor/index.pkl":
                with blob.open("rb") as f:
                    self.indexAnchor = pickle.load(f)
            if blob.name == "postings_gcp_body_stem/postings_gcp/index.pkl":
                with blob.open("rb") as f:
                    self.indexBodyStem = pickle.load(f)
            if blob.name == "postings_gcp_title_stem/index.pkl":
                with blob.open("rb") as f:
                    self.indexTitleStem = pickle.load(f)
            if blob.name == "docVecLen.pkl":
                with blob.open("rb") as f:
                    self.docVecLen = pickle.load(f)
            if blob.name == "DLBody.pkl":
                with blob.open("rb") as f:
                    self.DLBody = pickle.load(f)
            if blob.name == "DLTitle.pkl":
                with blob.open("rb") as f:
                    self.DLTitle = pickle.load(f)
            if blob.name == "pageviews.pkl":
                with blob.open("rb") as f:
                    self.pageViews = pickle.load(f)
            if blob.name == "IdTitle/IdTitle.pkl":
                with blob.open("rb") as f:
                    self.IdTitle = pickle.load(f)
            if blob.name == "pageRank.csv":
                with blob.open("r") as f:
                    reader = csv.reader(f)
                    self.page_rank = {int(rows[0]): math.log(float(rows[1]), 10) for rows in reader}
        self.bm25_title = search_backend.BM25_from_index(self.indexTitleStem, self.DLTitle, "title_stem",
                                                         self.page_rank, k1=1.4, b=0.75)
        self.bm25_text = search_backend.BM25_from_index(self.indexBodyStem, self.DLBody, "body_stem/postings_gcp", self.page_rank,
                                                        k1=1.4, b=0.75)
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
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query = tokenize(query)
    query = [app.stemmer.stem(term) for term in query]
    future_body = app.executor.submit(app.bm25_text.search, {1: query}, N=50)
    future_title = app.executor.submit(app.bm25_title.search, {1: query}, N=50)
    res = search_backend.merge_results(future_body.result(), future_title.result(),title_weight=0.3, text_weight=0.7, N=50)
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
    res = searchBody.get_topN_score_for_queries({1: tokenize(query)}, app.indexBody, app.DL_body, "body", app.docVecLen,
                                                N=40)
    return jsonify([(int(doc_id), app.IdTitle[doc_id]) for doc_id, score in res])

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
        res.append(math.pow(app.page_rank.loc[doc_id]),10)
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
    parameters_list = [(1.4 , 0.75)]
    weights_list = [(0.3, 0.7)]
    N = 100
    blob = app.bucket.get_blob("queries_train.json")
    with blob.open("rt") as f:
        queries = json.load(f)
    query_dict = {}
    for i in enumerate(queries.keys()):
        query_dict[i[0]] = [j for j in tokenize(i[1])]
        query_dict[i[0]] = [app.stemmer.stem(j) for j in query_dict[i[0]]]
        query_dict[i[0]] = list(set(query_dict[i[0]]))
    relevance_dict = dict(enumerate(queries.values()))
    grid_models = evaluations.grid_search_models(query_dict, relevance_dict, app.DLBody, "body_stem/postings_gcp", app.DLTitle, "title_stem", parameters_list, weights_list, N, app.indexTitleStem, app.indexBodyStem, app.page_rank)
    print(grid_models)
    ret = {}
    for k1, b, title_w, body_w in grid_models.keys():
        bm25_title = search_backend.BM25_from_index(app.indexTitleStem, app.DLTitle, "title_stem", app.page_rank, k1=k1, b=b)
        bm25_text = search_backend.BM25_from_index(app.indexBodyStem, app.DLBody, "body_stem/postings_gcp", app.page_rank, k1=k1, b=b)
        timeS = time.time()
        bm25_queries_score_test_title = bm25_title.search(query_dict, N=N)
        bm25_queries_score_test_text = bm25_text.search(query_dict, N=N)
        merge_res = search_backend.merge_results(bm25_queries_score_test_title, bm25_queries_score_test_text, title_weight=title_w,text_weight=body_w, N=N)
        print(time.time() - timeS)
        test_metrices = evaluations.evaluate(dict(enumerate(queries.values())), merge_res, k=40, print_scores=False)
        ret[(k1, b, title_w, body_w)] =  test_metrices['MAP@k']
        print((k1, b, title_w, body_w), test_metrices['MAP@k'])
    k_values = [1, 3, 5, 10, 20, 30, 40, 50]
    metrices_names = ['MAP@k']
    statistics_of_differnt_k = evaluations.plot_metric_with_differnt_k_values(dict(enumerate(queries.values())), merge_res,
                                                                  metrices_names, k_values=k_values)
    return jsonify(list(ret))


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
