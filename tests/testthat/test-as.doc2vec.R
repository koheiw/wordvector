library(quanteda)
library(wordvector)
options(wordvector.threads = 2)

corp <- head(data_corpus_inaugural, 59)

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE,
               concatenator = " ") %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_compound(data_dictionary_LSD2015, keep_unigrams = TRUE)

dfmt <- dfm(toks, remove_padding = TRUE) 

set.seed(1234)
wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)

test_that("textmodel_doc2vec works", {
    
    # dfm
    dov1 <- as.textmodel_doc2vec(dfmt, wov)
    
    expect_equal(
        names(dov1),
        c("values", "weights", "type", "dim", "frequency", "window",  "iter", 
          "alpha", "use_ns", "ns_size", "sample", "normalize",  "min_count", 
          "tolower", "concatenator", "docvars", "ntoken",  "call", "version")
    )
    expect_equal(
        dim(dov1$values$word), c(5363L, 50L)
    )
    expect_equal(
        dim(dov1$values$doc), c(59L, 50L)
    )
    expect_null(dov1$weights)
    expect_false(dov1$normalize)
    expect_equal(
        dov1$frequency,
        featfreq(dfm_trim(dfmt, min_termfreq = 2))
    )
    expect_equal(
        class(dov1), c("textmodel_doc2vec", "textmodel_wordvector")
    )
    expect_output(
        print(dov1),
        paste(
            "",
            "Call:",
            "as.textmodel_doc2vec(x = dfmt, model = wov)",
            "",
            "50 dimensions; 59 documents.", sep = "\n"), fixed = TRUE
    )
    expect_error(
        probability(dov1),
        "x must be a trained textmodel_wordvector object"
    )
    expect_error(
        as.textmodel_doc2vec(dfmt, list()),
        "model must be a trained textmodel_word2vec, textmodel_doc2vec or textmodel_lsa"
    )
    
    # tokens
    dov2 <- as.textmodel_doc2vec(toks, wov, compound = FALSE)
    
    expect_identical(
        dov2$values$doc,
        dov1$values$doc
    )
    expect_equal(
        names(dov2),
        c("values", "weights", "type", "dim", "frequency", "window",  "iter", 
          "alpha", "use_ns", "ns_size", "sample", "normalize",  "min_count", 
          "tolower", "concatenator", "docvars", "ntoken",  "call", "version")
    )
    expect_equal(
        dim(dov2$values$word), c(5363L, 50L)
    )
    expect_equal(
        dim(dov2$values$doc), c(59L, 50L)
    )
    expect_equal(
        docnames(toks),
        rownames(dov2$values$doc)
    )
    expect_error(
        probability(dov2),
        "x must be a trained textmodel_wordvector object"
    )
    
    # matrix
    mat <- as.matrix(dov1, normalize = FALSE)
    dov3 <- as.textmodel_doc2vec(mat)
    
    expect_identical(
        dov3$values$doc,
        dov1$values$doc
    )
    expect_equal(
        names(dov3),
        c("values", "weights", "type", "dim", "frequency", "window",  "iter", 
          "alpha", "use_ns", "ns_size", "sample", "normalize",  "min_count", 
          "tolower", "concatenator", "docvars", "ntoken",  "call", "version")
    )
    expect_null(
        dov3$values$word
    )
    expect_equal(
        dim(dov3$values$doc), c(59L, 50L)
    )
    expect_equal(
        docnames(toks),
        rownames(dov3$values$doc)
    )
    expect_error(
        probability(dov3),
        "x must be a trained textmodel_wordvector object"
    )
    expect_error(
        as.textmodel_doc2vec(unname(mat)),
        "x must have rownames for documents"
    )
    mat[3,] <- NA
    expect_error(
        as.textmodel_doc2vec(mat),
        "x must be a numeric matrix without NA"
    )
    expect_error(
        as.textmodel_doc2vec(matrix(nrow = 0, ncol = 10)),
        "x is an empty matrix"
    )
})

test_that("as.textmodel_doc2vec works only with DM", {
    
    skip_on_cran()
    
    # DM
    dov1 <- textmodel_doc2vec(head(toks, 1000), dim = 10, type = "dm", use_ns = FALSE)
    
    expect_equal(
        class(as.textmodel_doc2vec(dfmt, dov1)), 
        c("textmodel_doc2vec", "textmodel_wordvector")
    )
    
    # DBOW
    dov2 <- textmodel_doc2vec(head(toks, 1000), dim = 10, type = "dbow", use_ns = FALSE)
    expect_error(
        as.textmodel_doc2vec(dfmt, dov2), 
        "x does not have the layer for words"
    )
})

test_that("textmodel_doc2vec returns zero for emptry documents (#17)", {
    toks <- tokens(c("Citizens of the United States", "")) %>% 
        tokens_tolower()
    dfmt <- dfm(toks)
    dov <- as.textmodel_doc2vec(dfmt, wov)
    expect_true(all(dov$values$doc[1,] != 0))
    expect_true(all(dov$values$doc[2,] == 0))
})

test_that("textmodel_doc2vec compounds tokens internally", {
    
    toks1 <- tokens(c("Hard working", "look forward"))
    mat1 <- matrix(rnorm(20), nrow = 2, dimnames = list(c("hard working", "look forward")))
    wov1 <- as.textmodel_word2vec(mat1, concatenator = " ", tolower = TRUE)
    dov1 <- as.textmodel_doc2vec(toks1, wov1)
    
    expect_true(
        all(dov1$values$doc != 0)
    )
    expect_true(
        wov1$tolower
    )
    expect_error(
        as.textmodel_doc2vec(toks1, wov, compound = c(TRUE, FALSE)),
        "The length of compound must be 1"
    )
    
    toks2 <- tokens(c("働きもの", "期待する")) 
    mat2 <- matrix(rnorm(20), nrow = 2, dimnames = list(c("働きもの", "期待する")))
    wov2 <- as.textmodel_word2vec(mat2, concatenator = "", tolower = FALSE)
    dov2 <- as.textmodel_doc2vec(toks2, wov2)
    expect_true(
        all(dov2$values$doc != 0)
    )
    expect_identical(
        wov2$concatenator,
        ""
    )
    expect_false(
        wov2$tolower
    )
})

