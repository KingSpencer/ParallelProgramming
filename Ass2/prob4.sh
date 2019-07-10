#!/bin/bash
for i in {0..20..1}
do 
spark-submit --master local[40] --driver-memory 100G ParallelRegression.py \
--train data/small.train --test data/small.test \
--beta beta_small_${i} --lam ${i} --silent
done