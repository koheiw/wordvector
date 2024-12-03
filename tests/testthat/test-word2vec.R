library(quanteda)
library(wordvector)

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
    expect_equal(
        dim(wov$vectors), c(5360, 50)
    )
    expect_equal(
        wov$sample, 1
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
    
    # docvector with model
    expect_equal(
        dim(dov$vectors), c(59, 50)
    )
    expect_equal(
        class(dov), "textmodel_docvector"
    )
    expect_equal(
        dov$sample, 1
    )
    expect_equal(
        dov$min_count, 2L
    )
    expect_output(
        print(dov),
        paste(
            "",
            "Call:",
            "doc2vec(toks_grp, wov)",
            "",
            "50 dimensions; 59 documents.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        names(dov),
        c("vectors", "type", "dim", "min_count", "frequency", "window", "iter", 
          "alpha", "use_ns", "ns_size", "sample", "concatenator", "call", "version")
    )
    
    # docvector without model
    expect_equal(
        dim(dov_nm$vectors), c(59, 50)
    )
    expect_equal(
        class(dov_nm), "textmodel_docvector"
    )
    expect_equal(
        dov_nm$sample, 1
    )
    expect_equal(
        dov_nm$min_count, 10L
    )
    
    expect_equal(
        names(dov_nm),
        c("vectors", "type", "dim", "min_count", "frequency", "window", "iter", 
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

