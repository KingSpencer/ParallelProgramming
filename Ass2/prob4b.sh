#!/bin/bash
for i in {0..10..1}
do 
spark-submit --master local[40] --driver-memory 100G ParallelRegression.py \
--train data/big.train --test data/big.test \
--beta beta_big_${i} --lam ${i} --silent
done