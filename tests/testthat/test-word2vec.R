library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

corp <- head(data_corpus_inaugural, 59) %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) 

set.seed(1234)
wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)
dov <- textmodel_doc2vec(toks, wov)

test_that("textmodel_word2vec works", {
    
    # wordvector
    expect_equal(
        class(wov), "textmodel_wordvector"
    )
    expect_true(
        wov$use_ns
    )
    expect_identical(
        wov$ns_size, 5L
    )
    expect_identical(
        wov$window, 5L
    )
    expect_identical(
        dim(wov$values), c(5360L, 50L)
    )
    expect_identical(
        dim(wov$weights), c(5360L, 50L)
    )
    expect_identical(
        wov$sample, 1.0
    )
    expect_equal(
        wov$min_count, 2L
    )
    
    expect_identical(
        featfreq(dfm_trim(dfm(toks), 2)),
        wov$frequency
    )
    
    expect_output(
        print(wov),
        paste(
            "",
            "Call:",
            "textmodel_word2vec(x = toks, dim = 50, min_count = 2, iter = 10, ",
            "    sample = 1)",
            "",
            "50 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(print(wov)), "textmodel_wordvector"
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

test_that("normalize is working", {
    
    skip_on_cran()
    
    wov0 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                               normalize = FALSE)
    expect_false(wov0$normalize)
    
    wov1 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                               normalize = TRUE)
    expect_true(wov1$normalize)
    
})

test_that("tolower is working", {
    
    skip_on_cran()
    
    wov0 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                               tolower = FALSE)
    expect_equal(dim(wov0$values),
                 c(5556L, 50L))
                 
    
    wov1 <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                               tolower = TRUE)
    expect_equal(dim(wov1$values),
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
        dimnames(wov0$values), dimnames(wov1$values) 
    )

})

test_that("textmodel_word2vec is robust", {
    
    expect_s3_class(
        textmodel_word2vec(head(toks, 1), dim = 50, iter = 10, min_count = 1),
        "textmodel_wordvector"
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
