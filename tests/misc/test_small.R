library(quanteda)
library(wordvector)
options(wordvector_threads = 8)

corp <- data_corpus_news2014
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>%
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                 padding = TRUE) %>%
    tokens_tolower()

wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE, iter = 10)
similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()

dov <- textmodel_doc2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE, iter = 10)

similarity(dov, analogy(~ washington - america + france)) %>% 
    head()

sim <- proxyC::simil(
    dov$values$doc,
    dov$values$doc["4263794",, drop = FALSE]
)
sim <- proxyC::simil(
    dov$values$doc,
    dov$values$doc["3016236",, drop = FALSE]
)
sim <- proxyC::simil(
    dov$values$doc,
    dov$values$doc["3555430",, drop = FALSE]
)

tail(sort(s <- rowSums(sim)))
print(tail(toks[order(s)]), max_ntoken = -1)

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
