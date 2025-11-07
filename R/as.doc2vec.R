#' Create distributed representation of documents
#' 
#' Create distributed representation of documents as weighted word vectors.
#' @param x a [quanteda::tokens] or [quanteda::dfm] object.
#' @param model a textmodel_wordvector object.
#' @param normalize if `TRUE`, normalized word vectors before creating document vectors.
#' @param weights weight the word vectors by user-provided values; either a single value or 
#'    multiple values sorted in the same order as the word vectors.
#' @param pattern [quanteda::pattern] to select words to apply `weights`. 
#' @param group_data if `TRUE`, apply `dfm_group(x)` before creating document vectors.
#' @param ... additional arguments passed to [quanteda::object2id].
#' @returns Returns a textmodel_docvector object with the following elements:
#'   \item{values}{a list of matrices for word and document vectors.}
#'   \item{dim}{the size of the document vectors.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{docvars}{document variables copied from `x`.}
#'   \item{normalize}{if the document vectors are normalized.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordvector package.}
#' @export
as.textmodel_doc2vec <- function(x, model, normalize = FALSE, 
                              weights = 1.0, pattern = NULL, 
                              group_data = FALSE, ...) {
    UseMethod("as.textmodel_doc2vec")
}


#' @export
#' @method as.textmodel_doc2vec dfm
as.textmodel_doc2vec.dfm <- function(x, model = NULL, normalize = FALSE, 
                                  weights = 1.0, pattern = NULL,
                                  group_data = FALSE, ...) {
    
    model <- upgrade_pre06(model)
    model <- check_model(model, c("word2vec", "lsa"))
    conc <- meta(x, field = "concatenator", type = "object")

    wov <- as.matrix(model, normalize)
    if (is.null(pattern)) {
        n <- nrow(wov)
        if (length(weights) == 1)
            weights <- rep(weights, n)
        weights <- check_double(weights, min = 0, min_len = n, max_len = n)
        wov <- wov * weights 
    } else {
        weights <- check_double(weights, min = 0)
        ids <- object2id(pattern, types = rownames(wov), match_pattern = "single", 
                         keep_nomatch = FALSE, concatenator = conc,
                         ...)
        if (length(ids) > 0) {
            i <- unique(unlist(ids, use.names = FALSE))
            wov[i,] <- wov[i,,drop = FALSE] * rep(weights, length(i))
        }
    }
    
    if (group_data)
        x <- dfm_group(x)
    x <- dfm_match(x, rownames(wov))
    
    l <- rowSums(x) == 0
    dov <- as.matrix(Matrix::tcrossprod(x, t(wov))) # NOTE: consider using proxyC::prod
    dov <- dov / sqrt(rowSums(dov ^ 2) / ncol(dov))
    dov[l,] <- 0
    
    result <- list(
        "values" = list("word" = wov, "doc" = dov),
        "dim" = model$dim,
        "concatenator" = conc, 
        "docvars" = x@docvars,
        "normalize" = normalize,
        "call" = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE), 
        "version" = utils::packageVersion("wordvector")
    )
    class(result) <- c("textmodel_doc2vec", "textmodel_wordvector")
    return(result)
}

#' @export
as.matrix.textmodel_doc2vec <- function(x, normalize = TRUE, 
                                        layer = c("documents", "words"), ...) {
        
    x <- upgrade_pre06(x)
    normalize <- check_logical(normalize)
    layer <- match.arg(layer)
    
    if (layer == "words") {
        result <- x$values$word
    } else {
        result <- x$values$doc
        if (is.null(result))
            stop("only doc2vec models have the document layer")
    }
    if (normalize) {
        v <- sqrt(rowSums(result ^ 2) / ncol(result))
        result <- result / v
    }
    return(result) 
}


