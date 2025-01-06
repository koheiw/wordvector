library(quanteda)
library(wordvector)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()

wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)

test_that("analogy works", {
    
    expect_equal(
        analogy(~ us),
        c("us" = 1)
    )
    
    expect_equal(
        analogy(~ us),
        c("us" = 1)
    )
    
    expect_equal(
        analogy(~ people - us),
        c("people" = 1, "us" = -1)
    )
    
    expect_error(
        analogy("people"),
        "formula must be a formula object"
    )
})

test_that("similarity works", {
    
    sim1 <- similarity(wov, "us", mode = "values")
    expect_true(is.matrix(sim1))
    expect_identical(
        dimnames(sim1),
        list(names(wov$frequency), "us")
    )
    
    sim2 <- similarity(wov, c("us", "people"), mode = "values")
    expect_true(is.matrix(sim2))
    expect_identical(
        dimnames(sim2),
        list(names(wov$frequency), c("us", "people"))
    )
    
    sim3 <- similarity(wov, "us", mode = "words")
    expect_true(is.matrix(sim3))
    expect_identical(
        sim3[1,],
        c("us" = "us")
    )
    expect_identical(
        dim(sim3),
        c(length(wov$frequency), 1L)
    )
    
    sim4 <- similarity(wov, c("us", "people"), mode = "words")
    expect_true(is.matrix(sim4))
    expect_identical(
        sim4[1,],
        c("us" = "us", "people" = "people")
    )
    expect_identical(
        dim(sim4),
        c(length(wov$frequency), 2L)
    )
    expect_warning(
        similarity(wov, c("xx", "yyy", "us"), mode = "values"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
        is.matrix(similarity(wov, c("xx", "yyy"), mode = "values"))
        )
    )
    expect_warning(
        similarity(wov, c("xx", "yyy", "us"), mode = "words"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
            is.matrix(similarity(wov, c("xx", "yyy"), mode = "words"))
        )
    )
    
    sim5 <- similarity(wov, c("us" = 1, "people" = -1), mode = "values")
    expect_equal(ncol(sim5), 1)
    expect_true(is.matrix(sim5))
    expect_identical(
        dimnames(sim5),
        list(names(wov$frequency), NULL)
    )
    
    sim6 <- similarity(wov, c("us" = 1, "people" = -1), mode = "words")
    expect_equal(ncol(sim6), 1)
    expect_true(is.matrix(sim6))
    expect_identical(
        dimnames(sim6),
        NULL
    )
    
    expect_error(
        similarity(wov, c(1, -1), mode = "words"),
        "words must be named"
    )
})

test_that("get_threads are working", {
    
    options("wordvector_threads" = "abc")
    expect_error(
        suppressWarnings(wordvector:::get_threads()),
        "wordvector_threads must be an integer"
    )
    options("wordvector_threads" = NA)
    expect_error(
        wordvector:::get_threads(),
        "wordvector_threads must be an integer"
    )
    
    ## respect other settings
    options("wordvector_threads" = NULL)
    
    Sys.setenv("OMP_THREAD_LIMIT" = 2)
    expect_equal(
        wordvector:::get_threads(), 2
    )
    Sys.unsetenv("OMP_THREAD_LIMIT")
    
    Sys.setenv("RCPP_PARALLEL_NUM_THREADS" = 3)
    expect_equal(
        wordvector:::get_threads(), 3
    )
    Sys.unsetenv("RCPP_PARALLEL_NUM_THREADS")
    
    options("wordvector_threads" = NULL)
})
