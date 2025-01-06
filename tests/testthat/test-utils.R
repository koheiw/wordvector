library(quanteda)
library(wordvector)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()

wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)

test_that("analogy works", {
    
    ana1 <- analogy(wov, ~ us, exclude = FALSE)
    expect_true(ana1$word[1] == "us")
    expect_true(ana1$similarity[1] == 1.0)
    expect_identical(attr(ana1, "weight"), 
                     c("us" = 1))
    
    ana2 <- analogy(wov, ~ people - us, exclude = FALSE)
    expect_true(ana2$word[1] != "us")
    expect_true(ana2$similarity[1] < 1.0)
    expect_identical(attr(ana2, "weight"), 
                     c("people" = 1, "us" = -1))
    
    ana3 <- analogy(wov, ~ us, exclude = TRUE)
    expect_true(ana3$word[1] != "us")
    expect_true(ana3$similarity[1] < 1.0)
    expect_identical(attr(ana3, "weight"), 
                     c("us" = 1))
    
    expect_equivalent(
        analogy(wov, ~ people, exclude = FALSE),
        analogy(wov, "people", exclude = FALSE)
    )
    
    expect_equivalent(
        analogy(wov, ~ people - us, exclude = FALSE),
        analogy(wov, c("people" = 1, "us" = -1), exclude = FALSE)
    )
    
    expect_error(
        analogy(wov, c(1, -1), exclude = FALSE),
        "formula must be named"
    )
    
    expect_warning(
        analogy(wov, ~ xxxx, exclude = FALSE),
        '"xxxx" is not found'
    )
    expect_true(
        suppressWarnings(
            is.data.frame(analogy(wov, ~ xxxx, exclude = FALSE))
        )
    )
    expect_warning(
        analogy(wov, ~ xxxx, exclude = FALSE),
        '"xxxx" is not found'
    )
    expect_true(
        suppressWarnings(
            is.data.frame(analogy(wov, ~ xxxx, exclude = FALSE))
        )
    )
    
    # different formulas
    expect_equal(
        attr(analogy(wov, ~ us, exclude = FALSE), "weight"),
        c("us" = 1.0)
    )
    expect_equal(
        attr(analogy(wov, ~ -us+people, exclude = FALSE), "weight"),
        c("us" = -1.0, "people" = 1.0)
    )
    expect_equal(
        attr(analogy(wov, ~ +people - us, exclude = FALSE), "weight"),
        c("people" = 1.0, "us" = -1.0)
    )
})

test_that("similarity works", {
    
    
    sim1 <- similarity(wov, "us", mode = "value")
    expect_true(is.matrix(sim1))
    expect_identical(
        dimnames(sim1),
        list(names(wov$frequency), "us")
    )
    
    sim2 <- similarity(wov, c("us", "people"), mode = "value")
    expect_true(is.matrix(sim2))
    expect_identical(
        dimnames(sim2),
        list(names(wov$frequency), c("us", "people"))
    )
    
    sim3 <- similarity(wov, "us", mode = "word")
    expect_true(is.matrix(sim3))
    expect_identical(
        sim3[1,],
        c("us" = "us")
    )
    expect_identical(
        dim(sim3),
        c(length(wov$frequency), 1L)
    )
    
    sim4 <- similarity(wov, c("us", "people"), mode = "word")
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
        similarity(wov, c("xx", "yyy", "us"), mode = "value"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
        is.matrix(similarity(wov, c("xx", "yyy"), mode = "value"))
        )
    )
    expect_warning(
        similarity(wov, c("xx", "yyy", "us"), mode = "word"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
            is.matrix(similarity(wov, c("xx", "yyy"), mode = "word"))
        )
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
