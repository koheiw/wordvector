#' Word2vec model
#' 
#' Train a word2vec model (Mikolov et al., 2013) using a [quanteda::tokens] object.
#' @param x a [quanteda::tokens] or [quanteda::tokens_xptr] object.
#' @param dim the size of the word vectors.
#' @param type the architecture of the model; either "cbow" (continuous back-of-words), 
#'   "sg" (skip-gram), or "dm" (distributed memory).
#' @param min_count the minimum frequency of the words. Words less frequent than 
#'   this in `x` are removed before training.
#' @param window the size of the word window. Words within this window are considered 
#'   to be the context of a target word.
#' @param iter the number of iterations in model training.
#' @param alpha the initial learning rate.
#' @param use_ns if `TRUE`, negative sampling is used. Otherwise, hierarchical softmax 
#'   is used.
#' @param ns_size the size of negative samples. Only used when `use_ns = TRUE`.
#' @param sample the rate of sampling of words based on their frequency. Sampling is 
#'   disabled when `sample = 1.0`
#' @param tolower lower-case all the tokens before fitting the model.
#' @param model a trained Word2vec model; if provided, its word vectors are updated for `x`.
#' @param include_data if `TRUE`, the resulting object includes the data supplied as `x`.
#' @param verbose if `TRUE`, print the progress of training.
#' @param ... additional arguments.
#' @returns Returns a textmodel_word2vec object with the following elements:
#'   \item{values}{a list of a matrix for word vector values.}
#'   \item{weights}{a matrix for word vector weights.}
#'   \item{dim}{the size of the word vectors.}
#'   \item{type}{the architecture of the model.}
#'   \item{frequency}{the frequency of words in `x`.}
#'   \item{window}{the size of the word window.}
#'   \item{iter}{the number of iterations in model training.}
#'   \item{alpha}{the initial learning rate.}
#'   \item{use_ns}{the use of negative sampling.}
#'   \item{ns_size}{the size of negative samples.}
#'   \item{min_count}{the value of min_count.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{data}{the original data supplied as `x` if `include_data = TRUE`.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordvector package.}
#' @details
#'  If `type = "dm"`, it trains a doc2vec model but saves only 
#'  word vectors to save storage space. [wordvector::textmodel_doc2vec] should be 
#'  used to access document vectors. 
#'     
#'  Users can changed the number of processors used for the parallel computing via
#'  `options(wordvector_threads)`.
#' 
#' @references 
#'   Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). 
#'   Distributed Representations of Words and Phrases and their Compositionality. 
#'   https://arxiv.org/abs/1310.4546.
#' @export
#' @examples
#' \donttest{
#' library(quanteda)
#' library(wordvector)
#' 
#' # pre-processing
#' corp <- data_corpus_news2014 
#' toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
#'    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
#'    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
#'                  padding = TRUE) %>% 
#'    tokens_tolower()
#'
#' # train word2vec
#' wov <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, sample = 0.001)
#'
#' # find similar words
#' head(similarity(wov, c("berlin", "germany", "france"), mode = "words"))
#' head(similarity(wov, c("berlin" = 1, "germany" = -1, "france" = 1), mode = "values"))
#' head(similarity(wov, analogy(~ berlin - germany + france), mode = "words"))
#' }
textmodel_word2vec <- function(x, dim = 50, type = c("cbow", "sg", "dm"), 
                               min_count = 5, window = ifelse(type == "sg", 10, 5), 
                               iter = 10, alpha = 0.05, model = NULL, 
                               use_ns = TRUE, ns_size = 5, sample = 0.001, tolower = TRUE,
                               include_data = FALSE, verbose = FALSE, ...) {
    UseMethod("textmodel_word2vec")
}

#' @import quanteda
#' @useDynLib wordvector
#' @export
#' @method textmodel_word2vec tokens
#' 
textmodel_word2vec.tokens <- function(x, dim = 50, type = c("cbow", "sg", "dm"), 
                               min_count = 5, window = ifelse(type == "sg", 10, 5), 
                               iter = 10, alpha = 0.05, model = NULL, 
                               use_ns = TRUE, ns_size = 5, sample = 0.001, tolower = TRUE,
                               include_data = FALSE, verbose = FALSE, ...) {
    
    type <- ifelse(type == "skip-gram", "sg", type) # for backward compatibility
    type <- match.arg(type)
    wordvector(x, dim, type, FALSE, min_count, window, iter, alpha, model, 
               use_ns, ns_size, sample, tolower, include_data, verbose, ...)
    
}

wordvector <- function(x, dim = 50, type = c("cbow", "sg", "dm", "dbow"), 
                       doc2vec = FALSE, 
                       min_count = 5, window = ifelse(type == "sg", 10, 5), 
                       iter = 10, alpha = 0.05, model = NULL, 
                       use_ns = TRUE, ns_size = 5, sample = 0.001, tolower = TRUE,
                       include_data = FALSE, verbose = FALSE, ..., 
                       normalize = FALSE) {

    type <- match.arg(type)
    dim <- check_integer(dim, min = 2)
    min_count <- check_integer(min_count, min = 0)
    window <- check_integer(window, min = 1)
    iter <- check_integer(iter, min = 1)
    use_ns <- check_logical(use_ns)
    ns_size <- check_integer(ns_size, min_len = 1)
    alpha <- check_double(alpha, min = 0)
    sample <- check_double(sample, min = 0)
    normalize <- check_logical(normalize)
    tolower <- check_logical(tolower)
    include_data <- check_logical(include_data)
    verbose <- check_logical(verbose)
    
    if (normalize)
        .Defunct(msg = "'normalize' is defunct. Use 'as.matrix(x, normalize = TRUE)' instead.")
    
    if (!is.null(model)) {
        model <- upgrade_pre06(model)
        if (doc2vec) {
            model <- check_model(model, c("word2vec", "doc2vec"))
        } else {
            model <- check_model(model, c("word2vec"))
        }
        if (model$dim != dim || model$type != type || model$use_ns != use_ns) {
            dim <- model$dim
            type <- model$type
            use_ns <- model$use_ns
            warning("dim, type and use_na are overwritten by the pre-trained model", 
                    call. = FALSE)
        }
    }
    
    if (include_data)
        y <- as.tokens(x)
    
    x <- as.tokens_xptr(x)
    if (tolower)
        x <- tokens_tolower(x)
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    
    result <- cpp_word2vec(x, model, size = dim, window = window,
                           sample = sample, withHS = !use_ns, negative = ns_size, 
                           threads = get_threads(), iterations = iter,
                           alpha = alpha, 
                           type = match(type, c("cbow", "sg", "dm", "dbow", "dbow2")), 
                           normalize = FALSE, 
                           doc2vec = doc2vec,
                           verbose = verbose)
    
    if (!is.null(result$message))
        stop("Failed to train word2vec (", result$message, ")")
    
    result$type <- type
    result$min_count <- min_count
    result$tolower <- tolower
    result$concatenator <- meta(x, field = "concatenator", type = "object")
    if (include_data) # NOTE: consider removing
        result$data <- y
    if (doc2vec) {
        result$docvars <- attr(x, "docvars")
        rownames(result$docvars) <- docnames(x)
        rownames(result$values$doc) <- docnames(x)
    }
    result$call <- try(match.call(sys.function(-2), call = sys.call(-2)), silent = TRUE)
    result$version <- utils::packageVersion("wordvector")
    if (doc2vec) {
        class(result) <- c("textmodel_doc2vec", "textmodel_wordvector")
    } else {
        class(result) <- c("textmodel_word2vec", "textmodel_wordvector")
    }
    return(result)
}

word2vec <- function(...) {
    .Deprecated("textmodel_word2vec")
    textmodel_word2vec(...)
}

#' Print method for trained word vectors
#' @param x for print method, the object to be printed
#' @param ... not used.
#' @method print textmodel_word2vec
#' @keywords internal
#' @return an invisible copy of `x`. 
#' @export
print.textmodel_word2vec <- function(x, ...) {
    x <- upgrade_pre06(x)
    cat("\nCall:\n")
    print(x$call)
    cat("\n", prettyNum(x$dim, big.mark = ","), " dimensions; ",
        prettyNum(nrow(x$values$word), big.mark = ","), " words.",
        "\n", sep = "")
    invisible(x)
}

#' Print method for trained document vectors
#' @param x for print method, the object to be printed
#' @param ... unused
#' @method print textmodel_doc2vec
#' @keywords internal
#' @return an invisible copy of `x`. 
#' @export
print.textmodel_doc2vec <- function(x, ...) {
    x <- upgrade_pre06(x)
    cat("\nCall:\n")
    print(x$call)
    cat("\n", prettyNum(x$dim, big.mark = ","), " dimensions; ",
        prettyNum(nrow(x$values$doc), big.mark = ","), " documents.",
        "\n", sep = "")
    invisible(x)
}

#' Extract word or document vectors
#'
#' Extract word or document vectors from a `textmodel_word2vec` or `textmodel_doc2vec` object.
#' @rdname as.matrix
#' @param x a `textmodel_word2vec` or `textmodel_doc2vec` object.
#' @param normalize if `TRUE`, returns normalized vectors.
#' @param layer the layer from which the vectors are extracted.
#' @param group \[experimental\] average sentence or paragraph vectors from the same document. 
#'   Silently ignored when `layer = "word"`. 
#' @param ... not used.
#' @return a matrix that contain the word or document vectors in rows.
#' @export
as.matrix.textmodel_word2vec <- function(x, normalize = TRUE, 
                                         layer = "words", ...){
    
    x <- upgrade_pre06(x)
    layer <- match.arg(layer)
    normalize <- check_logical(normalize)
    
    result <- x$values$word
    if (normalize) {
        v <- sqrt(rowSums(result ^ 2) / ncol(result))
        result <- result / v
    }
    return(result) 
}

# for old objects before v0.6.0

#' @noRd
#' @method print textmodel_docvector
#' @export
print.textmodel_docvector <- print.textmodel_doc2vec

#' @noRd
#' @method print textmodel_wordvector
#' @export
print.textmodel_wordvector <- print.textmodel_word2vec

#' @noRd
#' @export
as.matrix.textmodel_docvector <- as.matrix.textmodel_doc2vec

#' @noRd
#' @export
as.matrix.textmodel_wordvector <- as.matrix.textmodel_word2vec

