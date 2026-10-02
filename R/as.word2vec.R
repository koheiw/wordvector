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
as.textmodel_word2vec.matrix <- function(x, tolower = FALSE, concatenator = "_", ...) {
    
    tolower <- check_logical(tolower)
    concatenator <- check_character(concatenator)
    
    if (nrow(x) == 0 || ncol(x) == 0)
        stop("x is an empty matrix")
    if (!is.numeric(x) || any(is.na(x)))
        stop("x must be a numeric matrix without NA")
    if (is.null(rownames(x)))
        stop("x must have rownames for words")
    colnames(x) <- NULL
    
    result <- build_word2vec(
        values = list("word" = x),
        weights = NULL,
        dim = ncol(x),
        tolower = tolower,
        concatenator = concatenator, 
        normalize = FALSE,
        call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE),
        ...
    )
    return(result)
}

