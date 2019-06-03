#!/bin/bash

# First argument is the name of the results file

echo -e 'Year\tAccuracy@1\tAccuracy@5\tAccuracy@10\tAccuracy@1_new\tAccuracy@5_new\tAccuracy@10_new\tOOV\tNew_pairs' > ${1}

for i in {2010..2016}
do
    echo '===='
    echo ${i}
    echo '===='
    # /projects/ltg/python3/bin/python3 transform_diachronic.py --modelfile models/incremental/${i}_0.model --reference 2017_dataset/${i}_single.tsv >> ${1}
    python3 transform_diachronic.py --modelfile models/NoW/now_${i}_incremental.model --reference 2019_dataset/${i}_single.tsv >> ${1}
done
