library(quanteda)
library(wordvector)
library(word2vec)

jaccard <- function(a, b) {
    mapply(function(x, y) {length(intersect(x, y)) / length(union(x, y))},
           as.data.frame.matrix(a), as.data.frame.matrix(b)
    )
}

correlation <- function(a, b) {
    mapply(function(x, y) {cor(x, y)},
           as.data.frame.matrix(a), as.data.frame.matrix(b)
    )
}

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()
toks_grp <- tokens_group(toks)

lis <- as.list(toks)
lis_grp <- as.list(toks_grp)

#feat <- names(topfeatures(dfm(toks), 20))
feat <- c("people", "government", "us", "world", "country")
doc <- tail(docnames(toks_grp))

test_that("Skip-gram models are similar", {
    
    wdv <- wordvector::word2vec(toks, dim = 50, iter = 5, min_count = 100, type = "skip-gram",
                                verbose = FALSE, threads = 1, sample = 0)
    w2v <- word2vec::word2vec(lis, dim = 50, iter = 5, min_count = 100, type = "skip-gram",
                              verbose = FALSE, threads = 1, sample = 0)
    
    expect_true(all(
        jaccard(synonyms(wdv, feat, 100),
                synonyms(w2v, feat, 100)) > 0.9
    ))
    
    dov <- wordvector::doc2vec(toks_grp, wdv)
    d2v <- word2vec::doc2vec(w2v, newdata = sapply(lis_grp, paste, collapse = " "))
    
    expect_true(all(
        correlation(proxyC::simil(dov[doc,]), 
                    proxyC::simil(d2v[doc,])) > 0.8
    ))
})

test_that("CBOW models are similar", {
    
    wdv <- wordvector::word2vec(toks, dim = 100, iter = 10, min_count = 100, type = "cbow",
                                verbose = FALSE, threads = 1, sample = 0)
    w2v <- word2vec::word2vec(lis, dim = 100, iter = 10, min_count = 100, type = "cbow",
                              verbose = FALSE, threads = 1, sample = 0)
    
    expect_true(all(
        jaccard(synonyms(wdv, feat, 100),
                synonyms(w2v, feat, 100)) > 0.9
    ))
    
    dov <- wordvector::doc2vec(toks_grp, wdv)
    d2v <- word2vec::doc2vec(w2v, newdata = sapply(lis_grp, paste, collapse = " "))
    
    expect_true(all(
        correlation(proxyC::simil(dov[doc,]), 
                    proxyC::simil(d2v[doc,])) > 0.8
    ))
})
