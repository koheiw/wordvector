require(quanteda)
require(wordvector)

corp <- data_corpus_news2014
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
   tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>%
   tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                 padding = TRUE) %>%
   tokens_tolower()

toks0 <- head(toks, 5000)
toks1 <- tail(toks, 5000)

test_that("model works", {
    
    skip_on_cran()
    
    wov0 <- textmodel_word2vec(toks0, dim = 50, type = "cbow")
    wov1 <- textmodel_word2vec(toks1, dim = 50, type = "cbow")
    wov2 <- textmodel_word2vec(toks1, dim = 50, type = "cbow", model = wov0)
    
    # without model
    f1 <- intersect(rownames(wov0$values), rownames(wov1$values))
    sim1 <- Matrix::diag(proxyC::simil(wov0$values[f1,], wov1$values[f1,], diag = TRUE))
    expect_true(
        median(sim1) < 0.20
    )
    
    # with model
    f2 <- intersect(rownames(wov0$values), rownames(wov2$values))
    sim2 <- Matrix::diag(proxyC::simil(wov0$values[f2,], wov2$values[f2,], diag = TRUE))
    expect_true(
        median(sim2) > 0.70
    )
    
    expect_error(
        textmodel_word2vec(toks1, dim = 50, type = "cbow", model = list()),
        "model must be a trained textmodel_wordvector"
    )
    
    expect_error(
        textmodel_word2vec(toks1, dim = 10, type = "cbow", model = wov0),
        "model must be trained with dim = 10"
    )
})

