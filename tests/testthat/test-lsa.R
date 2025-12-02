library(quanteda)
library(wordvector)

corp <- head(data_corpus_inaugural, 59) %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE)

dfmt <- dfm(toks, remove_padding = TRUE) 

set.seed(1234)
wov <- textmodel_lsa(toks, dim = 50, min_count = 2, sample = 0)
dov <- as.textmodel_doc2vec(dfmt, wov)
dov_gp <- as.textmodel_doc2vec(dfmt, wov, group_data = TRUE)

test_that("word2vec words", {
    
    # wordvector
    expect_equal(
        class(wov), 
        c("textmodel_lsa", "textmodel_wordvector")
    )
    expect_identical(
        dim(wov$values$word), c(5360L, 50L)
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
        class(expect_output(print(wov))), 
        class(wov)
    )
    expect_equal(
        names(dov),
        c("values", "dim", "tolower", "concatenator", "docvars", "normalize", 
          "call", "version")
    )
    
    # docvector with model
    expect_identical(
        dim(dov$values$word), c(5360L, 50L)
    )
    expect_equal(
        dim(dov$values$doc), c(5234L, 50L)
    )
    expect_equal(
        class(dov), 
        c("textmodel_doc2vec", "textmodel_wordvector")
    )
    expect_output(
        print(dov),
        paste(
            "",
            "Call:",
            "as.textmodel_doc2vec(x = dfmt, model = wov)",
            "",
            "50 dimensions; 5,234 documents.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(expect_output(print(dov))), 
        class(dov)
    )
    expect_equal(
        names(dov),
        c("values", "dim", "tolower", "concatenator", "docvars", "normalize", "call", "version")
    )
    
    # docvector with grouped data
    expect_identical(
        dim(dov_gp$values$word), c(5360L, 50L)
    )
    expect_identical(
        dim(dov_gp$values$doc), c(59L, 50L)
    )
    expect_equal(
        class(dov_gp), 
        c("textmodel_doc2vec", "textmodel_wordvector")
    )
    expect_equal(
        names(dov_gp),
        c("values", "dim", "tolower", "concatenator", "docvars", "normalize", "call", "version")
    )
})

test_that("tolower is working", {
    
    skip_on_cran()
    
    wov0 <- textmodel_lsa(toks, dim = 50, iter = 10, min_count = 2, 
                          tolower = FALSE)
    expect_equal(dim(wov0$values$word),
                 c(5556L, 50L))
    
    
    wov1 <- textmodel_lsa(toks, dim = 50, iter = 10, min_count = 2, 
                          tolower = TRUE)
    expect_equal(dim(wov1$values$word),
                 c(5360L, 50L))
    
})

