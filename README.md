# Diachronic Armed Conflicts
Diachronic armed conflicts prediction with news texts and word embeddings

Code and data for the paper:

`One-to-X analogical reasoning on word embeddings: a case for diachronic armed conflict prediction from news texts`
by Andrey Kutuzov, Erik Velldal and Lilja Øvrelid

## Embeddings models
- CBOW incremental embeddings for Gigaword (1994-2010) can be found at...
- CBOW embeddings for NoW (2010-2018) can be found at...

## Running
`python3 transform_diachronic.py --modelfile PATH_TO_FIRST_EMBEDDING --reference PATH_TO_FIRST_GOLD_DATA`

For example:

`python3 transform_diachronic.py --modelfile 2000_0.model --reference 2000_single.tsv`

will learn the transformatiom matrix on the embeddings and gold data from 2000 and test it on the next year (2001).

`test_all_2017_way.sh` will test all the years and output the results to the `results.tsv` file.
