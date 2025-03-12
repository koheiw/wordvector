library(quanteda)
library(wordvector)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()

wov <- textmodel_lsa(toks, dim = 50, min_count = 2, sample = 0)
dov <- textmodel_doc2vec(toks, wov, group_data = TRUE)

test_that("word2vec words", {
    
    # wordvector
    expect_equal(
        class(wov), "textmodel_wordvector"
    )
    expect_identical(
        dim(wov$values), c(5360L, 50L)
    )
    expect_equal(
        wov$weight, "count"
    )
    expect_identical(
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
            "textmodel_lsa(x = toks, dim = 50, min_count = 2, sample = 0)",
            "",
            "50 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(print(wov)), "textmodel_wordvector"
    )
    expect_equal(
        names(dov),
        c("values", "dim", "concatenator", "docvars", "call", "version")
    )
    
    # docvector with model
    expect_identical(
        dim(dov$values), c(59L, 50L)
    )
    expect_equal(
        class(dov), "textmodel_docvector"
    )
    expect_output(
        print(dov),
        paste(
            "",
            "Call:",
            "textmodel_doc2vec(x = toks, model = wov, group_data = TRUE)",
            "",
            "50 dimensions; 59 documents.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(print(dov)), "textmodel_docvector"
    )
    expect_equal(
        names(dov),
        c("values", "dim", "concatenator", "docvars", "call", "version")
    )
})
