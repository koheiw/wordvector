library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()

wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)
dov <- textmodel_doc2vec(toks, wov)
dov_gp <- textmodel_doc2vec(toks, wov, group_data = TRUE)

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
    
    # docvector with model
    expect_equal(
        dim(dov$values), c(5234L, 50L)
    )
    expect_equal(
        class(dov), "textmodel_docvector"
    )
    expect_output(
        print(dov),
        paste(
            "",
            "Call:",
            "textmodel_doc2vec(x = toks, model = wov)",
            "",
            "50 dimensions; 5,234 documents.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(print(dov)), "textmodel_docvector"
    )
    expect_equal(
        names(dov),
        c("values", "dim", "concatenator", "docvars", "call", "version")
    )
    
    # docvector with grouped data
    expect_identical(
        dim(dov_gp$values), c(59L, 50L)
    )
    expect_equal(
        class(dov_gp), "textmodel_docvector"
    )
    expect_equal(
        names(dov_gp),
        c("values", "dim", "concatenator", "docvars", "call", "version")
    )
    
})

test_that("textmodel_doc2vec works with different objects", {
    
    expect_equal(
        class(textmodel_doc2vec(toks, wov)),
        "textmodel_docvector"
    )
    
    expect_equal(
        class(textmodel_doc2vec(as.tokens_xptr(toks), wov)),
        "textmodel_docvector"
    )
    
    expect_error(
        textmodel_doc2vec(toks, list),
        "The object for 'model' must be a trained textmodel_wordvector"
    )
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

test_that("textmodel_word2doc returns zero for emptry documents (#17)", {
    toks <- tokens(c("Citizens of the United States", ""))
    dov <- textmodel_doc2vec(toks, wov)
    expect_true(all(dov$values[1,] != 0))
    expect_true(all(dov$values[2,] == 0))
})


