library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

corp <- head(data_corpus_inaugural, 59)

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE,
               concatenator = " ") %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_compound(data_dictionary_LSD2015, keep_unigrams = TRUE)

test_that("textmodel_doc2vec works", {
    
    # DM
    expect_output(
        dov1 <- textmodel_doc2vec(toks, dim = 50, iter = 5, min_count = 2, verbose = TRUE),
        "Training distributed memory model with 50 dimensions"
    )
    expect_equal(dov1$type, "dm")
    expect_false(dov1$normalize)
    expect_equal(
        names(dov1),
        c("values", "weights", "type", "dim", "frequency", "window",  "iter", "alpha", 
          "use_ns", "ns_size", "sample", "normalize",  "min_count", "concatenator", "docvars", "call", "version")
    )
    expect_equal(
        dim(dov1$values$word), c(5363L, 50L)
    )
    expect_equal(
        dim(dov1$values$doc), c(59L, 50L)
    )
    expect_output(
        print(dov1),
        paste(
            "",
            "Call:",
            "textmodel_doc2vec(x = toks, dim = 50, min_count = 2, iter = 5, ",
            "    verbose = TRUE)",
            "",
            "50 dimensions; 59 documents.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(expect_output(print(dov1))), 
        class(dov1)
    )
    
    expect_equal(
        rownames(probability(dov1, c("good", "bad"), layer = "words", mode = "numeric")),
        rownames(dov1$values$word)
    )
    
    expect_equal(
        rownames(probability(dov1, c("good", "bad"), layer = "documents", mode = "numeric")),
        rownames(dov1$values$doc)
    )
    
    # DBOW
    expect_output(
        dov2 <- textmodel_doc2vec(toks, dim = 50, type = "dbow", iter = 5, min_count = 2, verbose = TRUE),
        "Training distributed BOW model with 50 dimensions"
    )
    expect_equal(dov2$type, "dbow")
    expect_false(dov2$normalize)
    expect_equal(
        names(dov2),
        c("values", "weights", "type", "dim", "frequency", "window",  "iter", "alpha", 
          "use_ns", "ns_size", "sample", "normalize",  "min_count", "concatenator", "docvars", "call", "version")
    )
    expect_null(
        dov2$values$word
    )
    expect_equal(
        dim(dov2$values$doc), c(59L, 50L)
    )
    expect_output(
        print(dov2),
        paste(
            "",
            "Call:",
            "textmodel_doc2vec(x = toks, dim = 50, type = \"dbow\", min_count = 2, ", 
            "    iter = 5, verbose = TRUE)",
            "",
            "50 dimensions; 59 documents.", sep = "\n"), fixed = TRUE
    )
    expect_equal(
        class(expect_output(print(dov2))), 
        class(dov2)
    )
    
    expect_error(
        probability(dov2, c("good", "bad"), layer = "words", mode = "numeric"),
        "x does not have the layer for words"
    )
    
    expect_equal(
        rownames(probability(dov2, c("good", "bad"), layer = "documents", mode = "numeric")),
        rownames(dov2$values$doc)
    )
})

test_that("textmodel_doc2vec works hierachical softmax", {
    
    skip_on_cran()
    
    # DM
    dov1 <- textmodel_doc2vec(head(toks, 1000), dim = 10, type = "dm", use_ns = FALSE)
    expect_equal(
        class(dov1), 
        c("textmodel_doc2vec", "textmodel_wordvector")
    )
    expect_false(
        dov1$use_ns
    )
    expect_equal(
        dov1$type, 
        "dm"
    )
    
    # DBOW
    dov2 <- textmodel_doc2vec(head(toks, 1000), dim = 10, type = "dbow", use_ns = FALSE)
    expect_equal(
        class(dov2), 
        c("textmodel_doc2vec", "textmodel_wordvector")
    )
    expect_false(
        dov2$use_ns
    )
    expect_equal(
        dov2$type, 
        "dbow"
    )
})

test_that("textmodel_doc2vec works with pre-trained models", {
    
    skip_on_cran()
    
    # Zero learning
    dov0_pre <- textmodel_doc2vec(toks, type = "dm")
    dov0 <- textmodel_doc2vec(toks, type = "dm", iter = 1, model = dov0_pre,
                              alpha = 0)
    
    expect_identical(as.matrix(dov0, layer = "words"), 
                     as.matrix(dov0_pre, layer = "words"))
    
    # DM
    dov1_pre <- textmodel_doc2vec(toks, type = "dm")
    dov1 <- textmodel_doc2vec(toks, type = "dm", iter = 5, model = dov1_pre)
    
    r <- cor(t(as.matrix(dov1_pre, layer = "words"))[,1:1000], 
             t(as.matrix(dov1, layer = "words"))[,1:1000])
    expect_gt(median(diag(r)), 0.9)
    
    r <- cor(t(as.matrix(dov1_pre, layer = "documents")),
             t(as.matrix(dov1, layer = "documents")))
    expect_gt(median(diag(r)), 0.8)
    
    # DBOW
    dov2_pre <- textmodel_doc2vec(toks, type = "dbow")
    dov2 <- textmodel_doc2vec(toks, type = "dbow", iter = 5, model = dov2_pre)
    
    r <- cor(t(as.matrix(dov2_pre, layer = "documents")), 
             t(as.matrix(dov2, layer = "documents")))
    expect_gt(median(diag(r)), 0.8)
    
    # errors
    expect_error(
        textmodel_doc2vec(toks, type = "dbow", iter = 1, model = list()),
        "model must be a trained textmodel_word2vec or textmodel_doc2vec", fixed = TRUE
    )
    expect_warning(
        textmodel_doc2vec(toks, type = "dbow", iter = 1, model = dov1_pre),
        "dim, type and use_na are overwritten by the pre-trained model", fixed = TRUE
    )
})
