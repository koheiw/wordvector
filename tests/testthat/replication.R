library(quanteda)
library(wordvector)
library(word2vec)
options(wordvector_threads = 2)

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

corp <- head(data_corpus_inaugural, 59) %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()
toks_grp <- tokens_group(toks)

lis <- as.list(toks)
lis_grp <- as.list(toks_grp)

feat <- names(topfeatures(dfm(toks), 20))
#feat <- c("people", "government", "us", "world", "country")
doc <- tail(docnames(toks_grp), 20)

test_that("Skip-gram models are similar", {
    skip_on_cran()
    set.seed(1234)
    
    wdv <- textmodel_word2vec(toks, dim = 100, iter = 20, min_count = 0, type = "skip-gram",
                              verbose = FALSE, sample = 0)
    w2v <- word2vec(lis, dim = 100, iter = 20, min_count = 0, type = "skip-gram",
                    verbose = FALSE, threads = 2, sample = 0)
    
    expect_true(all(
        correlation(proxyC::simil(as.matrix(wdv)[feat,]),
                    proxyC::simil(as.matrix(w2v)[feat,])) > 0.8
    ))
    
    dov <- as.matrix(textmodel_doc2vec(toks_grp, wdv))
    d2v <- doc2vec(w2v, newdata = sapply(lis_grp, paste, collapse = " "))
    
    expect_true(all(
        correlation(proxyC::simil(dov[doc,]), 
                    proxyC::simil(d2v[doc,])) > 0.8
    ))
})

test_that("CBOW models are similar", {
    skip_on_cran()
    set.seed(1234)
    
    wdv <- textmodel_word2vec(toks, dim = 100, iter = 20, min_count = 0, type = "cbow",
                              verbose = FALSE, sample = 0)
    w2v <- word2vec(lis, dim = 100, iter = 20, min_count = 0, type = "cbow",
                              verbose = FALSE, threads = 2, sample = 0)
    expect_true(all(
        correlation(proxyC::simil(as.matrix(wdv)[feat,]),
                    proxyC::simil(as.matrix(w2v)[feat,])) > 0.8
    ))
    
    dov <- as.matrix(textmodel_doc2vec(toks_grp, wdv))
    d2v <- doc2vec(w2v, newdata = sapply(lis_grp, paste, collapse = " "))
    
    expect_true(all(
        correlation(proxyC::simil(dov[doc,]), 
                    proxyC::simil(d2v[doc,])) > 0.8
    ))
})
