#' Create a word2vec model
#' 
#' Create a word2vec model from a matrix.
#' @param x a matrix containing word vectors in rows.
#' @param ... additional arguments.
#' @returns Returns a dummy textmodel_word2vec object
#' @export
as.textmodel_word2vec <- function(x, ...) {
    UseMethod("as.textmodel_word2vec")
}


#' @export
#' @method as.textmodel_word2vec matrix
as.textmodel_word2vec.matrix <- function(x, ...) {
    
    if (is.null(rownames(x)))
        stop("x must have rownames for words")
    if (!is.numeric(x) || any(is.na(x)))
        stop("x must be a numeric matrix without NA")
    colnames(x) <- NULL
    
    result <- list(
        "values" = list("word" = x),
        "weights" = matrix(),
        "dim" = ncol(x),
        "tolower" = FALSE,
        "concatenator" = "_", 
        "docvars" = data.frame(),
        "normalize" = FALSE,
        "call" = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE), 
        "version" = utils::packageVersion("wordvector")
    )
    class(result) <- c("textmodel_word2vec", "textmodel_wordvector")
    return(result)
}

