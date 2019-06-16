# Diachronic Armed Conflicts
Diachronic armed conflicts prediction with news texts and word embeddings

Code and data for the paper:

*`One-to-X analogical reasoning on word embeddings: a case for diachronic armed conflict prediction from news texts`*
by Andrey Kutuzov, Erik Velldal and Lilja Øvrelid

## Embeddings models
Word embeddings we trained can be found at the [NLPL Vectors repository](http://vectors.nlpl.eu/repository/):
- [CBOW incremental embeddings for Gigaword (1995-2010)](http://vectors.nlpl.eu/repository/11/191.zip)
- [CBOW incremental embeddings for News on the Web (2010-2017)](http://vectors.nlpl.eu/repository/11/192.zip)

## Running
`python3 transform_diachronic.py --modelfile PATH_TO_FIRST_EMBEDDING --reference PATH_TO_FIRST_GOLD_DATA`

For example:

`python3 transform_diachronic.py --modelfile 2000.bin --reference 2000_single.tsv`

will learn the transformatiom matrix on the embeddings and gold data from 2000 and test it on the next year (2001).

`test_all_2017_way.sh` will test all the years and output the results to the `results.tsv` file.
