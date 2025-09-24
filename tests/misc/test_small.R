library(quanteda)
library(wordvector)
options(wordvector_threads = 8)

corp <- data_corpus_news2014
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>%
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                 padding = TRUE) %>%
    tokens_tolower()

wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow2", min_count = 5, verbose = TRUE, iter = 10)

for (i in 1:10) {
    cat(i, "\n")
    wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE, iter = 30)
}

# word2vec -------------------------------------
wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, iter = 5, 
                          verbose = TRUE)

similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()
similarity(wdv, analogy(~ berlin - germany + france)) %>% 
    head()


# LSA -------------------------------------

lsa <- textmodel_lsa(toks, dim = 50, min_count = 0, verbose = TRUE)
similarity(wdv, analogy(~ berlin - germany + france)) %>% 
    head()
