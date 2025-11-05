library(quanteda)
library(wordvector)

corp <- head(data_corpus_inaugural, 59) 

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE,
               concatenator = " ") %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_compound(data_dictionary_LSD2015, keep_unigrams = TRUE)

dfmt <- dfm(toks, remove_padding = TRUE) 

wov_nn <- readRDS("../data/word2vec_v0.5.1.RDS") 
wov_nm <- readRDS("../data/word2vec-norm_v0.5.1.RDS")

test_that("as.matrix() works with old objects", {
    
    skip_on_cran()
    
    expect_identical(dim(as.matrix(wov_nn)), c(5360L, 10L))
    expect_error(as.matrix(wov_nn, layer = "documents"),
                 "'arg' should be \"words\"")
    expect_identical(dim(as.matrix(wov_nm)), c(5360L, 10L))
    expect_error(as.matrix(wov_nm, layer = "documents"),
                 "'arg' should be \"words\"")

})

test_that("print() works", {
    
    skip_on_cran()
    
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

})

test_that("as.textmodel_doc2vec() works", {
    
    skip_on_cran()
    
    wov_nn <- readRDS("../data/word2vec_v0.5.1.RDS") 
    expect_identical(
        rownames(as.matrix(as.textmodel_doc2vec(dfmt, wov_nn))),
        docnames(dfmt)
    )
    
    wov_nm <- readRDS("../data/word2vec-norm_v0.5.1.RDS")
    expect_identical(
        rownames(as.matrix(as.textmodel_doc2vec(dfmt, wov_nm))),
        docnames(dfmt)
    )
})
