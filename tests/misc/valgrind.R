# Test script for memory errors on linux
# 
# sudo apt-get install valgrind
# 
# R -d "valgrind --tool=memcheck --leak-check=full" --vanilla < tests/misc/valgrind.R
# 
# See https://cran.r-project.org/doc/manuals/r-devel/R-exts.html#Using-Valgrind
#--------------------------------------------------------------------------------

library(quanteda)
library(wordvector)

toks <- tokens(data_corpus_news2014, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE) %>% 
    tokens_tolower()

cb1 <- word2vec(toks, dim = 10, type = "cbow", use_ns = TRUE, iter = 5, verbose = TRUE, ns_size = 20)

cb2 <- word2vec(toks, dim = 10, type = "cbow", use_ns = FALSE, iter = 5, verbose = TRUE)

sg1 <- word2vec(toks, dim = 10, type = "skip-gram", use_ns = TRUE, iter = 5, verbose = TRUE)

sg2 <- word2vec(toks, dim = 10, type = "skip-gram", use_ns = FALSE, iter = 5, verbose = TRUE)
