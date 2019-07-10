from pyspark import SparkContext
import argparse
from operator import add

def read_graph(sc, graph):
    """
    Read a file containing pairs representing the edges of a graph and save
    it inside an rdd. 

        Inputs are:
           -sc: SparkContext instance
           -graph: path of the file containing pairs representing the edges of a graph 

        The return value is
           - an rdd saves the graph
    """
    return sc.textFile(graph)\
             .map(eval)

def compute_degree(graph):
    """
    Compute the degree of rdd graph and save it as (node, degree) pair

        Inputs are:
           -graph: rdd containing nodes pairs representing the edges of a graph 

        The return value is
           - an rdd saves the (node, degree) pairs
    """
    return graph.flatMap(lambda (u, v) : [(u, 1),(v, 1)])\
                .reduceByKey(add)

def save_rdd(rdd, save_path):
    """
    Save rdd as text file, each line is an element in RDD

        Inputs are:
            -rdd: rdd to be saved
            -save_path: the path where rdd is saved
    """
    rdd.saveAsTextFile(save_path)

def get_color_distribution(color):
    """
    Compute distribution of a color RDD

        Inputs are:
            -color: rdd to of the format (node, color)
        
        The return value is an rdd contains the sorted number of appearances 
        of each color
    """
    return color.mapValues(lambda c: (c, 1))\
                .values()\
                .reduceByKey(add)\
                .sortBy(lambda (c, n): n)\
                .values()

def compute_WL(graph):
    """
    Compute WL algorithm of a graph

        Inputs are:
            -graph: rdd containing nodes pairs representing the edges of a graph
        
        The return value is an rdd contains the final (node, color) pairs
    """
    color = graph.flatMap(lambda (u, v) : [u, v])\
                     .distinct()\
                     .map(lambda u: (u, 1))\
                     .cache()
    while(True):
        color_new = graph.flatMap(lambda (u, v) : [(u, v), (v, u)])\
                             .join(color)\
                             .values()\
                             .mapValues(lambda color : [color])\
                             .reduceByKey(add)\
                             .mapValues(lambda clist : hash(str(sorted(clist)))).cache()
        # reannotate the colors
        color_num = color.values().distinct().count()
        color_new_num = color_new.values().distinct().count()
        if color_num == color_new_num:
            break
        color = color_new

    return color_new

def get_num_nodes(graph):
    """
    Compute number of nodes in a graph rdd

        Inputs are:
            -graph: rdd containing nodes pairs representing the edges of a graph

        The return value is the number of nodes in the graph
    """
    return graph.flatMap(lambda (u, v) : [u, v])\
                     .distinct()\
                     .count()

def compare_WL(graph1, graph2, save_path):
    """
    Compare if 2 graphs are isomorphic or maybe or not

        Inputs are:
            -graph1: rdd containing nodes pairs representing the edges of the 1st graph
            -graph2: rdd containing nodes pairs representing the edges of the 2nd graph
            -save_path: where to save the mapping if 2 graphs are isomorphic or maybe

        The return value is string shows if 2 graphs are isomorphic or maybe or not
    """
    n1 = get_num_nodes(graph1)
    n2 = get_num_nodes(graph2)
    if n1 != n2:
        return "not isomorphic"
    n = n1
    color1 = compute_WL(graph1).cache()
    color2 = compute_WL(graph2).cache()
    dist1 = get_color_distribution(color1).collect()
    dist2 = get_color_distribution(color2).collect()

    if dist1 == dist2 and len(dist1) == n:
        color1_reverse = color1.map(lambda (u, c) : (c, u))
        color2_reverse = color2.map(lambda (u, c) : (c, u))
        mapping = color1_reverse.join(color2_reverse)\
                                .values()

        save_rdd(mapping, save_path)
        return "isomorphic"
    elif dist1 == dist2 and len(dist1) < n:
        color1_reverse = color1.map(lambda (u, c) : (c, [u]))\
                               .reduceByKey(add)
        color2_reverse = color2.map(lambda (u, c) : (c, [u]))\
                               .reduceByKey(add)
        mapping = color1_reverse.join(color2_reverse)\
                                .values()
        save_rdd(mapping, save_path)
        return "maybe isomorphic"

    else:
        return "not isomorphic"












if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Graph Matching',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('question',type=int, default=1, help='Choose which question you are answering')
    parser.add_argument('-g1', '--graph1', default=None, help='Path to first graph')
    parser.add_argument('-g2', '--graph2', default=None, help='Path to second graph')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    sc = SparkContext('local[20]', appName='Graph Matching')

    if not args.verbose :
        sc.setLogLevel("ERROR")

    if args.question == 1:
        graph = read_graph(sc, args.graph1)
        degree = compute_degree(graph)
        print(degree.collect())
        save_rdd(degree, 'question1')

    elif args.question == 2:
        graph = read_graph(sc, args.graph1)
        colored_graph = compute_WL(graph)
        print(colored_graph.collect())
        save_rdd(colored_graph, 'question2')

    elif args.question == 3 or args.question == 4:
        graph1 = read_graph(sc, args.graph1)
        graph2 = read_graph(sc, args.graph2)
        save_path = 'question' + str(args.question)
        print(compare_WL(graph1, graph2, save_path))



