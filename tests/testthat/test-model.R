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

test_that("textmodel_word2vec() works", {
    
    skip_on_cran()
    
    wov0 <- textmodel_word2vec(toks0, dim = 50, type = "cbow", alpha = 0.5)
    wov1 <- textmodel_word2vec(toks1, dim = 50, type = "cbow", alpha = 0.5)
    wov2 <- textmodel_word2vec(toks1, dim = 50, type = "cbow", model = wov0)
    
    expect_false(identical(rownames(wov1$values$word), 
                           rownames(wov0$values$word)))
    expect_false(identical(rownames(wov2$values$word), 
                           rownames(wov0$values$word)))
    expect_true(identical(rownames(wov1$values$word), 
                          rownames(wov2$values$word)))
    
    # without model
    f1 <- intersect(rownames(wov0$values$word), 
                    rownames(wov1$values$word))
    sim1 <- Matrix::diag(proxyC::simil(wov0$values$word[f1,], 
                                       wov1$values$word[f1,], diag = TRUE))
    expect_lt(
        median(sim1), 0.20
    )
    
    # with model
    f2 <- intersect(rownames(wov0$values$word), 
                    rownames(wov2$values$word))
    sim2 <- Matrix::diag(proxyC::simil(wov0$values$word[f2,], 
                                       wov2$values$word[f2,], diag = TRUE))
    expect_gt(
        median(sim2), 0.70
    )
    
    expect_error(
        textmodel_word2vec(toks1, dim = 50, type = "cbow", model = list()),
        "model must be a trained textmodel_word2vec"
    )
    
    expect_error(
        textmodel_word2vec(toks1, dim = 50, type = "dbow", model = wov0),
        "'arg' should be one of \"cbow\", \"sg\", \"dm\""
    )
    
    expect_warning(
        textmodel_word2vec(toks1, dim = 25, type = "cbow", model = wov0),
        "dim, type and use_na are overwritten by the pre-trained model"
    )
    
    expect_warning(
        textmodel_word2vec(toks1, dim = 25, type = "sg", model = wov0),
        "dim, type and use_na are overwritten by the pre-trained model"
    )
})


test_that("textmodel_doc2vec() works", {
    
    skip_on_cran()
    
    dov0 <- textmodel_doc2vec(toks0, dim = 50, type = "dm", alpha = 0.5)
    dov1 <- textmodel_doc2vec(toks1, dim = 50, type = "dm", alpha = 0.5)
    dov2 <- textmodel_doc2vec(toks1, dim = 50, type = "dm", model = dov0)
    
    # word layer
    expect_false(identical(rownames(dov1$values$word), 
                           rownames(dov0$values$word)))
    expect_false(identical(rownames(dov2$values$word), 
                           rownames(dov0$values$word)))
    expect_true(identical(rownames(dov1$values$word), 
                          rownames(dov2$values$word)))
    
    # document layer
    expect_false(identical(rownames(dov1$values$doc), 
                           rownames(dov0$values$doc)))
    expect_false(identical(rownames(dov2$values$doc), 
                           rownames(dov0$values$doc)))
    expect_true(identical(rownames(dov1$values$doc), 
                          rownames(dov2$values$doc)))
    
    # without model
    f1 <- intersect(rownames(dov0$values$word), 
                    rownames(dov1$values$word))
    sim1 <- Matrix::diag(proxyC::simil(dov0$values$word[f1,], 
                                       dov1$values$word[f1,], diag = TRUE))
    
    sim1 <- Matrix::diag(proxyC::simil(as.matrix(dov0, layer = "word")[f1,], 
                                       as.matrix(dov1, layer = "word")[f1,], diag = TRUE))
    
    expect_lt(
        median(sim1), 0.20
    )
    
    # with model
    f2 <- intersect(rownames(dov0$values$word), 
                    rownames(dov2$values$word))
    sim2 <- Matrix::diag(proxyC::simil(dov0$values$word[f2,], 
                                       dov2$values$word[f2,], diag = TRUE))
    expect_gt(
        median(sim2), 0.70
    )
    
    expect_error(
        textmodel_doc2vec(toks1, dim = 50, type = "dm", model = list()),
        "model must be a trained textmodel_word2vec or textmodel_doc2vec"
    )
    
    expect_warning(
        textmodel_doc2vec(toks1, dim = 10, type = "dm", model = dov0),
        "dim, type and use_na are overwritten by the pre-trained model"
    )
    
    expect_warning(
        textmodel_doc2vec(toks1, dim = 50, type = "dbow", model = dov0),
        "dim, type and use_na are overwritten by the pre-trained model"
    )
})

