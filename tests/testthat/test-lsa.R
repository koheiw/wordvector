library(quanteda)
library(wordvector)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()
toks_grp <- tokens_group(toks)

wov <- lsa(toks, dim = 50, min_count = 2, sample = 0)
dov <- doc2vec(toks_grp, wov)
#dov_nm <- doc2vec(toks_grp, min_count = 10, sample = 0)

test_that("word2vec words", {
    
    # wordvector
    expect_equal(
        class(wov), "textmodel_wordvector"
    )
    expect_equal(
        dim(wov$vectors), c(5360, 50)
    )
    expect_equal(
        wov$weight, "count"
    )
    expect_equal(
        wov$min_count, 2L
    )
    expect_equal(
        featfreq(dfm_trim(dfm(toks), 2)),
        wov$frequency
    )
    expect_output(
        print(wov),
        paste(
            "",
            "Call:",
            "lsa(x = toks, dim = 50, min_count = 2, sample = 0)",
            "",
            "50 dimensions; 5,360 words.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        names(dov),
        c("vectors", "dim", "min_count", "frequency", "engine", "weight", 
          "concatenator", "call", "version")
    )
    
    # docvector with model
    expect_equal(
        dim(dov$vectors), c(59, 50)
    )
    expect_equal(
        class(dov), "textmodel_docvector"
    )
    expect_equal(
        dov$weight, "count"
    )
    expect_equal(
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
        names(dov),
        c("vectors", "dim", "min_count", "frequency", "engine", "weight", 
          "concatenator", "call", "version")
    )
    
    # docvector without model
    # expect_equal(
    #     dim(dov_nm$vectors), c(59, 50)
    # )
    # expect_equal(
    #     class(dov_nm), "textmodel_docvector"
    # )
    # expect_equal(
    #     dov_nm$sample, 0
    # )
    # expect_equal(
    #     dov_nm$min_count, 10L
    # )
})
