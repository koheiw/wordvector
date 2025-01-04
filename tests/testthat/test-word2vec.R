library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()
toks_grp <- tokens_group(toks)

wov <- word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)
dov <- doc2vec(toks_grp, wov)
dov_nm <- doc2vec(toks_grp, min_count = 10, sample = 1)

test_that("word2vec works", {
    
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
            "word2vec(x = toks, dim = 50, min_count = 2, iter = 10, sample = 1)",
            "",
            "50 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(print(wov)), "textmodel_wordvector"
    )
    
    # docvector with model
    expect_equal(
        dim(dov$values), c(59L, 50L)
    )
    expect_equal(
        dim(dov$weights), c(5360L, 50L)
    )
    expect_equal(
        class(dov), "textmodel_docvector"
    )
    expect_identical(
        dov$sample, 1.0
    )
    expect_identical(
        dov$min_count, 2L
    )
    expect_output(
        print(dov),
        paste(
            "",
            "Call:",
            "doc2vec(x = toks_grp, model = wov)",
            "",
            "50 dimensions; 59 documents.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(print(dov)), "textmodel_docvector"
    )
    expect_equal(
        names(dov),
        c("values", "weights", "type", "dim", "min_count", "frequency", "window", "iter", 
          "alpha", "use_ns", "ns_size", "sample", "concatenator", "call", "version")
    )
    
    # docvector without model
    expect_identical(
        dim(dov_nm$values), c(59L, 50L)
    )
    expect_identical(
        dim(dov_nm$weights), c(1405L, 50L)
    )
    expect_equal(
        class(dov_nm), "textmodel_docvector"
    )
    expect_identical(
        dov_nm$sample, 1.0
    )
    expect_identical(
        dov_nm$min_count, 10L
    )
    
    expect_equal(
        names(dov_nm),
        c("values", "weights", "type", "dim", "min_count", "frequency", "window", "iter", 
          "alpha", "use_ns", "ns_size", "sample", "concatenator", "call", "version")
    )
    
})

test_that("doc2vec works with different objects", {
    
    expect_equal(
        class(doc2vec(toks, wov)),
        "textmodel_docvector"
    )
    
    expect_equal(
        class(doc2vec(as.tokens_xptr(toks), wov)),
        "textmodel_docvector"
    )
    
    expect_error(
        doc2vec(toks, list),
        "The object for 'model' must be a trained textmodel_wordvector"
    )
})

test_that("normalize is working", {
    
    skip_on_cran()
    
    wov0 <- word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                     normalize = FALSE)
    expect_false(wov0$normalize)
    
    wov1 <- word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1,
                     normalize = TRUE)
    expect_true(wov1$normalize)
    
})

