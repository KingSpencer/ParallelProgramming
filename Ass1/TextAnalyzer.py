import sys
import argparse
import numpy as np
import math
from pyspark import SparkContext


def toLowerCase(s):
    """ Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')


    if args.mode=='TF':
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed
        input_rdd = sc.textFile(args.input)
        input_rdd.flatMap(lambda x:x.split()) \
                 .map(lambda x:stripNonAlpha(toLowerCase(x))) \
                 .filter(lambda x:x != '') \
                 .map(lambda x:(x, 1)) \
                 .reduceByKey(lambda x, y: x+y) \
                 .saveAsTextFile(args.output)







    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL), 
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        input_rdd = sc.textFile(args.input)
        '''fre = input_rdd.map(lambda x: (x[1:-1].split(',')[0][2:], int(x[1:-1].split(',')[1]))) \
                    .sortBy(lambda (x,y):y, ascending=False) \
                    .take(20)'''
        top20_tuples = input_rdd.map(eval) \
                       .sortBy(lambda (x,y):y, ascending=False) \
                       .take(20)
        with open(args.output, 'w') as f:
            for top20_tuple in top20_tuples:
                f.write(str(top20_tuple))
                f.write('\n')
        





       
    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        allFiles = sc.wholeTextFiles(args.input)
        num_of_files = allFiles.count()
        allFiles.flatMapValues(lambda x:x.split()) \
                .mapValues(lambda x:stripNonAlpha(toLowerCase(x))) \
                .filter(lambda (key,val):val!='') \
                .distinct() \
                .map(lambda (key, val): (val, 1)) \
                .reduceByKey(lambda x, y: x+y) \
                .mapValues(lambda x: math.log(float(num_of_files) / x)) \
                .saveAsTextFile(args.output)







    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        tf_scores = sc.textFile(args.input).map(eval)
        idf_scores = sc.textFile(args.idfvalues).map(eval)
        tf_scores.join(idf_scores) \
                 .mapValues(lambda (v1, v2):v1 * v2) \
                 .sortBy(lambda (x,y):y, ascending=False) \
                 .saveAsTextFile(args.output)






        
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        tfidf_f1 = sc.textFile(args.input).map(eval)
        tfidf_f2 = sc.textFile(args.other).map(eval)
        numerator = tfidf_f1.join(tfidf_f2) \
                            .mapValues(lambda (v1, v2):v1 * v2) \
                            .map((lambda (key, value): value)) \
                            .reduce(lambda x, y: x+y)
        denom1 = tfidf_f1.map(lambda (key,val): val*val) \
                         .reduce(lambda x, y: x+y)
        denom2 = tfidf_f2.map(lambda (key,val): val*val) \
                         .reduce(lambda x, y: x+y)
        denominator = math.sqrt(denom1*denom2)
        sim = numerator/denominator
        with open(args.output, 'w') as f:
            f.write(str(sim))

        




