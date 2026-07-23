library(quanteda)
library(wordvector)
options(wordvector_threads = 8)
quanteda_options(verbose = TRUE)

corp <- data_corpus_news2014
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>%
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                 padding = TRUE) %>%
    tokens_tolower()

wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, iter = 10)
similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()

wdv2 <- textmodel_word2vec(toks[1:1000], dim = 50, type = "cbow", min_count = 2, iter = 10, 
                           model = wdv, update_weights = TRUE, verbose = TRUE)
wdv3 <- textmodel_word2vec(toks[1:1000], dim = 50, type = "cbow", min_count = 2, iter = 10, 
                           model = wdv, update_weights = FALSE, verbose = TRUE)


similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()
similarity(wdv2, analogy(~ washington - america + france)) %>% 
    head()
similarity(wdv3, analogy(~ washington - america + france)) %>% 
    head()

wdv$weights["america",]
wdv2$weights["america",]
wdv3$weights["america",]

dov <- textmodel_doc2vec(toks, dim = 100, type = "dm", min_count = 5, iter = 10, verbose = TRUE)
dov2 <- textmodel_doc2vec(toks, dim = 100, type = "dm", min_count = 5, iter = 20, verbose = TRUE)

sim <- proxyC::simil(
    dov$values$doc,
    dov$values$doc["3555430",, drop = FALSE]
)
tail(sort(s <- rowSums(sim)))

sim2 <- proxyC::simil(
    dov2$values$doc,
    dov2$values$doc["3555430",, drop = FALSE]
)
tail(sort(s2 <- rowSums(sim2)))

identical(s, s2)
cor(s, s2)

sim <- proxyC::simil(
    dov2$values$doc,
    dov2$values$doc["4263794",, drop = FALSE]
)
sim <- proxyC::simil(
    dov2$values$doc,
    dov2$values$doc["3016236",, drop = FALSE]
)
sim <- proxyC::simil(
    dov2$values$doc,
    dov2$values$doc["3555430",, drop = FALSE]
)
tail(sort(s <- rowSums(sim)))
print(tail(toks[order(s)]), max_ntoken = -1)

for (i in 1:10) {
    cat(i, "\n")
    wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, iter = 30, verbose = TRUE)
}

# word2vec -------------------------------------
wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, iter = 5, 
                          verbose = TRUE)

similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()
similarity(wdv, analogy(~ berlin - germany + france)) %>% 
    head()

# -----------------------------

dov <- textmodel_doc2vec(toks, dim = 50, type = "skip-gram", min_count = 5, iter = 10, verbose = TRUE)
seed <- LSX::seedwords("sentiment")
doc <- probability(dov, seed, layer = "documents")[,1]
print(toks[head(doc)], -1, -1)
print(toks[tail(doc)], -1, -1)

# LSA -------------------------------------

lsa <- textmodel_lsa(toks, dim = 50, min_count = 0, verbose = TRUE)
similarity(wdv, analogy(~ berlin - germany + france)) %>% 
    head()
