from search_backend import merge_results, BM25_from_index
import matplotlib.pyplot as plt


def intersection(l1,l2):
    """
    This function perform an intersection between two lists.

    Parameters
    ----------
    l1: list of documents. Each element is a doc_id.
    l2: list of documents. Each element is a doc_id.

    Returns:
    ----------
    list with the intersection (without duplicates) of l1 and l2
    """
    return list(set(l1)&set(l2))

def precision_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the precision@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    return (round(len(intersection(true_list, predicted_list[:k])) / k, 3))


def average_precision(true_list, predicted_list, k=40):
    """
    This function calculate the average_precision@k metric.(i.e., precision in every recall point).

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, average precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    bool_list = [True if (doc_id in true_list) else False for doc_id in predicted_list[:k]]
    rel = sum([1 for i in bool_list if i])
    if (rel == 0):
        return (0)
    # inter = intersection(true_list, predicted_list[:k])
    count = 0
    avg = 0
    for b in range(1, len(bool_list) + 1):
        if (bool_list[b - 1]):
            avg += precision_at_k(true_list, predicted_list, b)
            count += 1
        if (count == rel):
            return (round(avg / rel, 3))
    return (0)


def evaluate(true_relevancy, predicted_relevancy, k, print_scores=False):
    """
    This function calculates multiple metrics and returns a dictionary with metrics scores across different queries.
    Parameters
    -----------
    true_relevancy: list of tuples indicating the relevancy score for a query. Each element corresponds to a query.
    Example of a single element in the list:
                                            (3, {'question': ' what problems of heat conduction in composite slabs have been solved so far . ',
                                            'relevance_assessments': [(5, 3), (6, 3), (90, 3), (91, 3), (119, 3), (144, 3), (181, 3), (399, 3), (485, 1)]})

    predicted_relevancy: a dictionary of the list. Each key represents the query_id. The value of the dictionary is a sorted list of relevant documents and their scores.
                         The list is sorted by the score.
    Example:
            key: 1
            value: [(13, 17.256625), (486, 13.539465), (12, 9.957595), (746, 9.599499999999999), (51, 9.171265), .....]

    k: integer, a number to slice the length of the predicted_list

    print_scores: boolean, enable/disable a print of the mean value of each metric.

    Returns:
    -----------
    a dictionary of metrics scores as follows:
                                                        key: metric name
                                                        value: list of metric scores. Each element corresponds to a given query.
    """

    avg_precision_lst = []

    metrices = {
        'MAP@k': avg_precision_lst,
    }

    for query_id, query_info in true_relevancy.items():
        predicted = [doc_id for doc_id, score in predicted_relevancy[query_id]]
        ground_true = [doc_id for doc_id in query_info]
        avg_precision_lst.append(average_precision(ground_true, predicted, k=k))

    if print_scores:
        for name, values in metrices.items():
            print(name, sum(values) / len(values))

    return metrices


def grid_search_models(data, true_relevancy, DL_body, index_type_body, DL_title, index_type_title, bm25_param_list, w_list, N, idx_title, idx_body, page_rank):
    """
      This function is performing a grid search upon different combination of parameters.
    The parameters can be BM25 parameters (i.e., bm25_param_list) or different weights (i.e., w_list).

    Parameters
    ----------
    data: dictionary as follows:
                            key: query_id
                            value: list of tokens

    true_relevancy: list of tuples indicating the relevancy score for a query. Each element corresponds to a query.
    Example of a single element in the list:
                                            (3, {'question': ' what problems of heat conduction in composite slabs have been solved so far . ',
                                            'relevance_assessments': [(5, 3), (6, 3), (90, 3), (91, 3), (119, 3), (144, 3), (181, 3), (399, 3), (485, 1)]})

    bm25_param_list: list of tuples. Each tuple represent (k,b1) values.

    w_list: list of tuples. Each tuple represent (title_weight,body_weight).
    N: Integer. How many document to retrieve.

    idx_title: index build upon titles
    idx_body:  index build upon bodies
    ----------
    return: dictionary as follows:
                            key: tuple indiciating the parameters examined in the model (k1,b,title_weight,body_weight)
                            value: MAP@N score
    """
    models = {}
    for i in range(len(bm25_param_list)):
        bm25_title = BM25_from_index(idx_title, DL_title, index_type_title, page_rank, bm25_param_list[i][0], bm25_param_list[i][1])
        bm25_body = BM25_from_index(idx_body, DL_body, index_type_body, page_rank, bm25_param_list[i][0], bm25_param_list[i][1])
        bm25_queries_score_title = bm25_title.search(data, N)
        bm25_queries_score_body = bm25_body.search(data, N)
        for w in range(len(w_list)):
            bm25_queries_score_merged = merge_results(bm25_queries_score_title, bm25_queries_score_body, w_list[w][0],
                                                      w_list[w][1], N)
            models[(bm25_param_list[i][0], bm25_param_list[i][1], w_list[w][0], w_list[w][1])] = 0
            for j in range(len(bm25_queries_score_merged)):
                models[(bm25_param_list[i][0], bm25_param_list[i][1], w_list[w][0], w_list[w][1])] += average_precision(
                    [doc_id for doc_id in true_relevancy[j]],
                    [doc_id for doc_id, score in bm25_queries_score_merged[j]], N)
    return models


def plot_metric_with_differnt_k_values(true_relevancy, predicted_relevancy, metrices_names, k_values):
    """
    This function plot a for each given metric its value depands on k_values as line chart.
    This function does not return any value.

    Parameters
    ----------
    true_relevancy: list of tuples indicating the relevancy score for a query. Each element corresponds to a query.
    Example of a single element in the list:
                                            (3, {'question': ' what problems of heat conduction in composite slabs have been solved so far . ',
                                            'relevance_assessments': [(5, 3), (6, 3), (90, 3), (91, 3), (119, 3), (144, 3), (181, 3), (399, 3), (485, 1)]})

    predicted_relevancy: a dictionary of the list. Each key represents the query_id. The value of the dictionary is a sorted list of relevant documents and their scores.
                         The list is sorted by the score.
    Example:
            key: 1
            value: [(13, 17.256625), (486, 13.539465), (12, 9.957595), (746, 9.599499999999999), (51, 9.171265), .....]

    metrices_names: list of string representing the metrices to plot. For example: ['precision@k','recall@k','f_score@k']

    k_values: list of integer of different k values. For example [1,3,5]

    returns:
    plot values in format :

    statistics[metric_name] = [values , k_values]

    """
    statistics = {}
    metrices = {}
    for k in k_values:
        metrices[k] = evaluate(true_relevancy, predicted_relevancy, k, print_scores=False)
    for metric_name in metrices_names:
        # YOUR CODE HERE
        values = []
        for k in k_values:
            values.append(round(sum(metrices[k][metric_name]) / len(metrices[k][metric_name]), 3))
        statistics[metric_name] = [values, k_values]
        plt.plot(k_values, values)
        plt.title(metric_name)
        plt.show()
    return statistics
#
#

