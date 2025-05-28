
#' @export
as.matrix.textmodel_docvector <- function(x, ...){
    return(x$values) 
}

#' Create distributed representation of documents
#' 
#' Create distributed representation of documents as weighted word vectors.
#' @param x a [quanteda::tokens] object.
#' @param model a textmodel_wordvector object.
#' @param normalize if `TRUE`, normalized word vectors before creating document vectors.
#' @param weight weight the word vectors by user-provided values; either a single value or 
#'    multiple values sorted in the same order as the word vectors.
#' @param pattern [quanteda::pattern] to select words to apply `weight`. 
#' @param group_data if `TRUE`, apply `dfm_group(x)` before creating document vectors.
#' @param ... additional arguments passed to [quanteda::object2id].
#' @returns Returns a textmodel_docvector object with the following elements:
#'   \item{values}{a matrix for document vectors.}
#'   \item{dim}{the size of the document vectors.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{docvars}{document variables copied from `x`.}
#'   \item{normalize}{if the document vectors are normalized.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordvector package.}
#' @export
textmodel_doc2vec <- function(x, model, normalize = FALSE, 
                              weight = 1.0, pattern = NULL, 
                              group_data = FALSE, ...) {
    UseMethod("textmodel_doc2vec")
}

#' @export
#' @method textmodel_doc2vec tokens
textmodel_doc2vec.tokens <- function(x, model, normalize = FALSE, 
                                     weight = 1.0, pattern = NULL, 
                                     group_data = FALSE, ...) {
    
    if (!identical(class(model), "textmodel_wordvector"))
        stop("The object for 'model' must be a trained textmodel_wordvector")

    x <- dfm(x, remove_padding = TRUE, tolower = FALSE)
    result <- textmodel_doc2vec(x, model = model, normalize = normalize,
                                weight = weight, pattern = pattern, 
                                group_data = group_data, ...)
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}

#' @export
#' @method textmodel_doc2vec dfm
textmodel_doc2vec.dfm <- function(x, model = NULL, normalize = FALSE, 
                                  weight = 1.0, pattern = NULL,
                                  group_data = FALSE, ...) {
    
    conc <- meta(x, field = "concatenator", type = "object")
    wov <- as.matrix(model, normalize)
    
    if (is.null(pattern)) {
        n <- nrow(wov)
        if (length(weight) == 1)
            weight <- rep(weight, n)
        weight <- check_double(weight, min = 0, min_len = n, max_len = n)
        wov <- wov * weight 
    } else {
        weight <- check_double(weight, min = 0)
        ids <- object2id(pattern, types = rownames(wov), match_pattern = "single", 
                         keep_nomatch = FALSE, concatenator = conc,
                         ...)
        if (length(ids) > 0) {
            i <- unique(unlist(ids, use.names = FALSE))
            wov[i,] <- wov[i,,drop = FALSE] * rep(weight, length(i))
        }
    }
    
    if (group_data)
        x <- dfm_group(x)
    x <- dfm_match(x, rownames(wov))
    
    l <- rowSums(x) == 0
    dov <- Matrix::tcrossprod(x, t(wov)) # NOTE: consider using proxyC::prod
    dov <- dov / sqrt(Matrix::rowSums(dov ^ 2) / ncol(dov))
    dov[l,] <- 0
    
    result <- list(
        "values" = dov,
        "dim" = model$dim,
        "concatenator" = conc, 
        "docvars" = x@docvars,
        "normalize" = normalize,
        "call" = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE), 
        "version" = utils::packageVersion("wordvector")
    )
    class(result) <- "textmodel_docvector"
    return(result)
}

doc2vec <- function(...) {
    .Deprecated("textmodel_doc2vec")
    textmodel_doc2vec(...)
}

