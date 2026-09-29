build_word2vec <- function(...) {
    
    args <- list(...)
    result <- list(
        values = list(),
        weights = matrix(),
        type = NULL,
        dim = 50,
        frequency = NULL,
        window = NULL,
        iter = NULL,
        alpha = NULL,
        use_ns = NULL,
        ns_size = NULL,
        sample = NULL,
        normalize = NULL,
        min_count = 5,
        tolower = TRUE,
        concatenator = "_",
        call = NULL,
        version = utils::packageVersion("wordvector")
    )
    for (m in intersect(names(result), names(args$model)))
        result[m] <- args$model[m]
    for (n in intersect(names(result), names(args)))
        result[n] <- args[n]
    class(result) <- c("textmodel_word2vec", "textmodel_wordvector")
    return(result)
}

build_doc2vec <- function(docname, ...) {
    
    args <- list(...)
    result <- list(
        values = list(),
        weights = matrix(),
        type = NULL,
        dim = 50,
        frequency = NULL,
        window = NULL,
        iter = NULL,
        alpha = NULL,
        use_ns = NULL,
        ns_size = NULL,
        sample = NULL,
        normalize = NULL,
        min_count = 5,
        tolower = TRUE,
        concatenator = "_",
        docvars = data.frame(),
        ntoken = NULL,
        call = NULL,
        version = utils::packageVersion("wordvector")
    )
    for (m in intersect(names(result), names(args$model)))
        result[m] <- args$model[m]
    for (n in intersect(names(result), names(args)))
        result[n] <- args[n]
    rownames(result$values$doc) <- docname
    rownames(result$docvars) <- docname
    class(result) <- c("textmodel_doc2vec", "textmodel_wordvector")
    return(result)
}
