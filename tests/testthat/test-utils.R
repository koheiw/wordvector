library(quanteda)
library(wordvector)

corp <- head(data_corpus_inaugural, 59) %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()

dfmt <- dfm(toks, remove_padding = TRUE)

set.seed(1234)
wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                          normalize = FALSE)
dov <- as.textmodel_doc2vec(dfmt, model = wov)

test_that("as.matrix works", {
    
    # word2vec
    expect_setequal(rownames(as.matrix(wov)), 
                    types(tokens_trim(tokens_tolower(toks), min_termfreq = 2)))
    expect_error(
        as.matrix(wov, layer = "documents"),
        "'arg' should be \"words\""
    )
    
    expect_false(
        identical(as.matrix(wov, normalize = TRUE), wov$values$word)
    )
    expect_true(
        identical(as.matrix(wov, normalize = FALSE), wov$values$word)
    )
    
    # doc2vec
    expect_setequal(rownames(as.matrix(dov)), 
                    docnames(dfmt))
    expect_setequal(rownames(as.matrix(dov, layer = "words")), 
                    featnames(dfm_trim(dfmt, min_termfreq = 2)))
    
    expect_false(
        identical(as.matrix(dov, normalize = TRUE), dov$values$doc)
    )
    expect_true(
        identical(as.matrix(dov, normalize = FALSE), dov$values$doc)
    )
    
})

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
    
    # word2vec
    sim1 <- similarity(wov, "us", mode = "numeric")
    expect_true(is.matrix(sim1))
    expect_identical(
        dimnames(sim1),
        list(names(wov$frequency), "us")
    )
    
    sim2 <- similarity(wov, c("us", "people"), mode = "numeric")
    expect_true(is.matrix(sim2))
    expect_identical(
        dimnames(sim2),
        list(names(wov$frequency), c("us", "people"))
    )
    
    sim3 <- similarity(wov, "us", mode = "character")
    expect_true(is.matrix(sim3))
    expect_identical(
        sim3[1,],
        c("us" = "us")
    )
    expect_identical(
        dim(sim3),
        c(length(wov$frequency), 1L)
    )
    
    sim4 <- similarity(wov, c("us", "people"), mode = "character")
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
        similarity(wov, c("xx", "yyy", "us"), mode = "numeric"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
        is.matrix(similarity(wov, c("xx", "yyy"), mode = "numeric"))
        )
    )
    expect_warning(
        similarity(wov, "xx", mode = "character"),
        '"xx" is not found'
    )
    expect_warning(
        similarity(wov, c("xx", "yyy", "us"), mode = "character"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
            is.matrix(similarity(wov, c("xx", "yyy"), mode = "character"))
        )
    )
    
    sim5 <- similarity(wov, c("us" = 1, "people" = -1), mode = "numeric")
    expect_equal(ncol(sim5), 1)
    expect_true(is.matrix(sim5))
    expect_identical(
        dimnames(sim5),
        list(names(wov$frequency), NULL)
    )
    
    sim6 <- similarity(wov, c("us" = 1, "people" = -1), mode = "character")
    expect_equal(ncol(sim6), 1)
    expect_true(is.matrix(sim6))
    expect_identical(
        dimnames(sim6),
        NULL
    )
    expect_error(
        similarity(wov, c(TRUE, FALSE), mode = "character"),
        "targets must be a character vector or a named numeric vector"
    )
    expect_error(
        similarity(wov, c(1, -1), mode = "character"),
        "targets must be named"
    )
    
    # doc2vec
    sim10 <- similarity(dov, c("us" = 1, "people" = -1), mode = "numeric")
    expect_equal(ncol(sim10), 1)
    expect_true(is.matrix(sim10))
    expect_identical(
        dimnames(sim10),
        list(names(wov$frequency), NULL)
    )
    
    sim11 <- similarity(dov, c("us" = 1, "people" = -1), mode = "character")
    expect_equal(ncol(sim11), 1)
    expect_true(is.matrix(sim11))
    expect_identical(
        dimnames(sim11),
        NULL
    )
    
    sim12 <- similarity(dov, c("2009-Obama.1", "2017-Trump.1"), 
                        layer = "documents", mode = "numeric")
    expect_equal(ncol(sim12), 2)
    expect_true(is.matrix(sim12))
    expect_identical(
        dimnames(sim12),
        list(docnames(toks), c("2009-Obama.1", "2017-Trump.1"))
    )
    
    sim13 <- similarity(dov, c("2009-Obama.1", "2017-Trump.1"), 
                        layer = "documents", mode = "character")
    expect_equal(ncol(sim13), 2)
    expect_true(is.matrix(sim13))
    expect_identical(
        dimnames(sim13),
        list(NULL, c("2009-Obama.1", "2017-Trump.1"))
    )
    expect_error(
        similarity(dov, c(TRUE, FALSE), mode = "character"),
        "targets must be a character vector or a named numeric vector"
    )
    expect_error(
        similarity(list(), c("us" = 1, "people" = -1)),
        "x must be a textmodel_wordvector object"
    )
})

test_that("probability works", {
    
    skip_on_cran()
    skip_on_os("mac") # for github action

    # word2vec
    prob1 <- probability(wov, "us", mode = "numeric")
    expect_true(all(prob1 <= 1.0))
    expect_true(all(prob1 >= 0.0))
    expect_true(is.matrix(prob1))
    expect_identical(
        dimnames(prob1),
        list(names(wov$frequency), "us")
    )
    
    prob2 <- probability(wov, c("us", "people"), mode = "numeric")
    expect_true(all(prob2 <= 1.0))
    expect_true(all(prob2 >= 0.0))
    expect_true(is.matrix(prob2))
    expect_identical(
        dimnames(prob2),
        list(names(wov$frequency), c("us", "people"))
    )
    
    prob3 <- probability(wov, "us", mode = "character")
    expect_true(is.matrix(prob3))
    expect_identical(
        prob3[1,],
        c("us" = "let")
    )
    expect_identical(
        dim(prob3),
        c(length(wov$frequency), 1L)
    )
    
    prob4 <- probability(wov, c("us", "people"), mode = "character")
    expect_true(is.matrix(prob4))
    expect_identical(
        prob4[1,],
        c("us" = "let", "people" = "american")
    )
    expect_identical(
        dim(prob4),
        c(length(wov$frequency), 2L)
    )
    expect_warning(
        probability(wov, c("xx"), mode = "numeric"),
        '"xx" is not found'
    )
    expect_warning(
        probability(wov, c("xx", "yyy", "us"), mode = "numeric"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
            is.matrix(probability(wov, c("xx", "yyy"), mode = "numeric"))
        )
    )
    expect_warning(
        probability(wov, c("xx", "yyy", "us"), mode = "character"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
            is.matrix(probability(wov, c("xx", "yyy"), mode = "character"))
        )
    )
    
    prob5 <- probability(wov, c("us" = 1, "people" = -1), mode = "numeric")
    expect_equal(ncol(prob5), 1)
    expect_true(is.matrix(prob5))
    expect_identical(
        dimnames(prob5),
        list(names(wov$frequency), NULL)
    )
    
    prob6 <- probability(wov, c("us" = 1, "people" = -1), mode = "character")
    expect_equal(ncol(prob6), 1)
    expect_true(is.matrix(prob6))
    expect_identical(
        dimnames(prob6),
        NULL
    )
    expect_error(
        probability(wov, c(TRUE, FALSE), mode = "character"),
        "targets must be a character vector or a named numeric vector"
    )
    expect_error(
        probability(wov, c(1, -1), mode = "character"),
        "targets must be named"
    )
    wov$normalize <- TRUE
    expect_error(
        probability(wov, c(1, -1), mode = "character"),
        "x must be trained with normalize = FALSE"
    )
    
    # doc2vec
    expect_error(
        probability(dov, c("us" = 1, "people" = -1), mode = "numeric"),
        "x must be a trained textmodel_wordvector object"
    )
    expect_error(
        probability(list(), c("us" = 1, "people" = -1)),
        "x must be a textmodel_wordvector object"
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

test_that("print and as.matrix works with old objects", {

    wov_nn <- readRDS("../data/word2vec_v0.5.1.RDS") 
    expect_identical(dim(as.matrix(wov_nn)), c(5360L, 10L))
    expect_error(as.matrix(wov_nn, layer = "documents"),
                 "'arg' should be \"words\"")
    expect_output(
        print(wov_nn),
        paste(
            "",
            "Call:",
            "textmodel_word2vec(x = toks, dim = 10, min_count = 2, iter = 10, ",
            "    sample = 1, normalize = FALSE)",
            "",
            "10 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    
    wov_nm <- readRDS("../data/word2vec-norm_v0.5.1.RDS")
    expect_identical(dim(as.matrix(wov_nm)), c(5360L, 10L))
    expect_error(as.matrix(wov_nm, layer = "documents"),
                 "'arg' should be \"words\"")
    expect_output(
        print(wov_nm),
        paste(
            "",
            "Call:",
            "textmodel_word2vec(x = toks, dim = 10, min_count = 2, iter = 10, ",
            "    sample = 1, normalize = TRUE)",
            "",
            "10 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    
    dov <- readRDS("../data/doc2vec_v0.5.1.RDS")
    expect_identical(dim(as.matrix(dov)), c(5234L, 10L))
    expect_error(as.matrix(dov, layer = "words"),
                 "x does not have the layer for words")
    expect_output(
        print(dov),
        paste(
            "",
            "Call:",
            "textmodel_doc2vec(x = dfmt, model = wov)",
            "",
            "10 dimensions; 5,234 documents", sep = "\n"), fixed = TRUE
    )
})

test_that("class check functions work as expected", {
    
    # word2vec
    expect_silent(
        wordvector:::check_word2vec(wov)
    )
    expect_error(
        wordvector:::check_word2vec(dov),
        "model must be a trained textmodel_word2vec"
    )
    
    # doc2vec
    expect_silent(
        wordvector:::check_doc2vec(dov)
    )
    expect_error(
        wordvector:::check_doc2vec(wov),
        "model must be a trained textmodel_doc2vec"
    )

})

test_that("old arguments still works", {
    
    expect_type(
        probability(wov, c("us", "people"), mode = "words"),
        "character"
    )
    
    expect_type(
        probability(wov, c("us", "people"), mode = "values"),
        "double"
    )
    
    expect_type(
        similarity(wov, c("us", "people"), mode = "words"),
        "character"
    )
    
    expect_type(
        similarity(wov, c("us", "people"), mode = "values"),
        "double"
    )
    
})

test_that("perplexity works", {
    
    # infrequent words
    word1 <- c("good", "nice", "excellent", "positive", "fortunate", "correct", "superior", 
              "bad", "nasty", "poor", "negative", "unfortunate",  "wrong", "inferior")
    suppressWarnings(
        ppl1 <- perplexity(wov, word1, dfmt)
    )
    expect_equal(ppl1, 3.217, tol = 0.001)
    
    # frequent words
    word2 <- c("america", "us", "people", "government", "state", "nation", "world", "peace", "public")
    suppressWarnings(
        ppl2 <- perplexity(wov, word2, dfmt)
    )
    expect_equal(ppl2, 2.719, tol = 0.001)
    
    expect_error(
        perplexity(wov, word2, list),
        "data must be a tokens or dfm"
    )
})
