library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

corp <- head(data_corpus_inaugural, 59) %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) 

test_that("textmodel_word2vec works", {
    
    skip_on_cran()
    
    # CBOW
    expect_output(
        wov1 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1, verbose = TRUE),
        "Training continuous BOW model with 50 dimensions"
    )
    expect_equal(
        class(wov1), 
        c("textmodel_word2vec", "textmodel_wordvector")
    )
    expect_true(
        wov1$use_ns
    )
    expect_identical(
        wov1$ns_size, 5L
    )
    expect_identical(
        wov1$window, 5L
    )
    expect_identical(
        dim(wov1$values$word), c(5360L, 50L)
    )
    expect_identical(
        dim(wov1$weights), c(5360L, 50L)
    )
    expect_identical(
        wov1$sample, 1.0
    )
    expect_equal(
        wov1$min_count, 2L
    )
    expect_false(
        wov1$normalize
    )
    expect_identical(
        featfreq(dfm_trim(dfm(toks), 2)),
        wov1$frequency
    )
    
    expect_output(
        print(wov1),
        paste(
            "",
            "Call:",
            "textmodel_word2vec(x = toks, dim = 50, min_count = 2, iter = 10, ",
            "    sample = 1, verbose = TRUE)",
            "",
            "50 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(expect_output(print(wov1))), 
        class(wov1)
    )
    
    expect_equal(
        rownames(probability(wov1, c("good", "bad"), layer = "words", mode = "numeric")),
        rownames(wov1$values$word)
    )
    
    expect_error(
        probability(wov1, c("good", "bad"), layer = "documents", mode = "numeric"),
        "textmodel_word2vec does not have the layer for documents"
    )
    
    # SG
    expect_output(
        wov2 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                                   type = "sg", verbose = TRUE),
        "Training skip-gram model with 50 dimensions"
    )
    expect_equal(
        class(wov2), 
        c("textmodel_word2vec", "textmodel_wordvector")
    )
    expect_true(
        wov2$use_ns
    )
    expect_identical(
        wov2$ns_size, 5L
    )
    expect_identical(
        wov2$window, 10L
    )
    expect_identical(
        dim(wov2$values$word), c(5360L, 50L)
    )
    expect_identical(
        dim(wov2$weights), c(5360L, 50L)
    )
    expect_identical(
        wov2$sample, 1.0
    )
    expect_equal(
        wov2$min_count, 2L
    )
    expect_false(
        wov2$normalize
    )
    expect_identical(
        featfreq(dfm_trim(dfm(toks), 2)),
        wov2$frequency
    )
    
    expect_output(
        print(wov2),
        paste(
            "",
            "Call:",
            "textmodel_word2vec(x = toks, dim = 50, type = \"sg\", min_count = 2, ",
            "    iter = 10, sample = 1, verbose = TRUE)",
            "",
            "50 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(expect_output(print(wov2))), 
        class(wov2)
    )
    
    expect_equal(
        rownames(probability(wov2, c("good", "bad"), layer = "words", mode = "numeric")),
        rownames(wov2$values$word)
    )
    
    expect_error(
        probability(wov2, c("good", "bad"), layer = "documents", mode = "numeric"),
        "textmodel_word2vec does not have the layer for documents"
    )
    
})


test_that("textmodel_word2vec works hierachical softmax", {
    
    skip_on_cran()
    
    # CBOW
    wov1 <- textmodel_word2vec(head(toks, 1000), type = "cbow", dim = 10, use_ns = FALSE)
    expect_equal(
        class(wov1), 
        c("textmodel_word2vec", "textmodel_wordvector")
    )
    expect_false(
        wov1$use_ns
    )
    expect_equal(
        wov1$type, 
        "cbow"
    )
    
    # SG
    
    wov2 <- textmodel_word2vec(head(toks, 1000), dim = 10, type = "sg", use_ns = FALSE)
    expect_equal(
        class(wov2), 
        c("textmodel_word2vec", "textmodel_wordvector")
    )
    expect_false(
        wov2$use_ns
    )
    expect_equal(
        wov2$type, 
        "sg"
    )
})

test_that("works with old names of type", {
    
    expect_output(
        wov <- textmodel_word2vec(head(toks, 10), dim = 50, iter = 10, 
                                   type = "skip-gram", verbose = TRUE),
        "Training skip-gram model with 50 dimensions"
    )
    expect_equal(
        wov$window,
        10L
    )
    expect_equal(
        wov$type, 
        "sg"
    )
})

test_that("textmodel_word2vec works with include_data", {
    
    skip_on_cran()
    wov0 <- textmodel_word2vec(toks, dim = 10, iter = 1, min_count = 10, 
                               include_data = TRUE)
    expect_identical(wov0$data, toks)
    
    wov1 <- textmodel_word2vec(as.tokens_xptr(toks), dim = 10, iter = 1, min_count = 10, 
                               include_data = TRUE)
    expect_identical(wov1$data, toks)
    
})

test_that("normalize is defunct", {
    
    skip_on_cran()
    
    expect_error({
        textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                                   normalize = TRUE)
        "'normalize' is defunct."
    })

})

test_that("tolower is working", {
    
    skip_on_cran()
    
    wov0 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                               tolower = FALSE)
    expect_equal(dim(wov0$values$word),
                 c(5556L, 50L))
                 
    
    wov1 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                               tolower = TRUE)
    expect_equal(dim(wov1$values$word),
                 c(5360L, 50L))
    
})

test_that("tokens and tokens_xptr produce the same result", {
    
    skip_on_cran()
    
    set.seed(1234)
    wov0 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)

    set.seed(1234)
    xtoks <- as.tokens_xptr(toks)
    wov1 <- textmodel_word2vec(xtoks, dim = 50, iter = 10, min_count = 2, sample = 1)
    
    expect_equal(
        names(wov0),
        names(wov1)
    )
    
    expect_equal(
        dimnames(wov0$values$word), 
        dimnames(wov1$values$word) 
    )

})

test_that("textmodel_word2vec is robust", {
    
    expect_s3_class(
        textmodel_word2vec(head(toks, 1), dim = 50, iter = 10, min_count = 1),
        c("textmodel_word2vec", "textmodel_wordvector")
    )
    
    expect_error(
        suppressWarnings(
            textmodel_word2vec(head(toks, 0), dim = 50, iter = 10, min_count = 1)
        ),
        "Failed to train word2vec"
    )
    
    expect_error(
        suppressWarnings(
            textmodel_word2vec(toks, dim = 0, iter = 10, min_count = 1)
        ),
        "The value of dim must be between 2 and Inf"
    )
    
    expect_error(
        suppressWarnings(
            textmodel_word2vec(toks, dim = 50, iter = 0, min_count = 1)
        ),
        "The value of iter must be between 1 and Inf"
    )
  
})  
