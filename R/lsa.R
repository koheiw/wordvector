#' Latent Semantic Analysis model
#' 
#' Train a Latent Semantic Analysis model (Deerwester et al., 1990) on a [quanteda::tokens] object.
#' @param x a [quanteda::tokens] object.
#' @param dim the size of the word vectors.
#' @param min_count the minimum frequency of the words. Words less frequent than 
#'    this in `x` are removed before training.
#' @param engine select the engine perform SVD to generate word vectors.
#' @param weight weighting scheme passed to [quanteda::dfm_weight()]. 
#' @param verbose if `TRUE`, print the progress of training.
#' @param ... additional arguments.
#' @returns Returns a fitted textmodel_wordvector with the following elements:
#'   \item{vectors}{a matrix for word vectors.}
#'   \item{frequency}{the frequency of words in `x`.}
#'   \item{engine}{the SVD engine used.}
#'   \item{weight}{weighting scheme.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordvector package.}
#' @references 
#'   Deerwester, S. C., Dumais, S. T., Landauer, T. K., Furnas, G. W., & Harshman, R. A. (1990). 
#'   Indexing by latent semantic analysis. JASIS, 41(6), 391–407.
#' @export
#' @examples
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
#' # train LSA
#' lsa <- lsa(toks, dim = 10, min_count = 20, verbose = TRUE)
#' analogy(lsa, ~ people)
#' analogy(lsa, ~ people + killed)
#' head(similarity(lsa, c("people", "killed"), mode = "word"))
lsa <- function(x, dim = 50, min_count = 5L, engine = c("RSpectra", "irlba", "rsvd"), 
                weight = "count", verbose = FALSE, ...) {
    UseMethod("lsa")   
}

#' @import quanteda
#' @export
lsa.tokens <- function(x, dim = 50L, min_count = 5L, engine = c("RSpectra", "irlba", "rsvd"), 
                       weight = "count", verbose = FALSE, ...) {
    
    engine <- match.arg(engine)
    dim <- check_integer(dim, min = 2)
    min_count <- check_integer(min_count, min = 0)
    verbose <- check_logical(verbose)
    
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    x <- dfm(x, remove_padding = TRUE)
    if (engine %in% c("RSpectra", "irlba", "rsvd")) {
        if (verbose) {
            cat(sprintf("Performing SVD into %d dimensions\n", dim))
            cat(sprintf("...using %s\n", engine))
        }
        svd <- get_svd(x, dim, engine, weight, ...)
        if (verbose)
            cat("...complete\n")
        wov <- svd$v
        rownames(wov) <- featnames(x)
    }
    result <- list(
        vectors = wov,
        dim = dim,
        min_count = min_count,
        frequency = featfreq(x),
        engine = engine,
        weight = weight,
        concatenator = meta(x, field = "concatenator", type = "object"),
        call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE),
        version = utils::packageVersion("wordvector")
    )
    class(result) <- "textmodel_wordvector"
    return(result)
}

#' @importFrom quanteda dfm_weight
#' @importFrom methods as
get_svd <- function(x, k, engine, weight = "count", reduce = FALSE, ...) {
    if (reduce) {
        # NOTE: generalize? featfreq(x) ^ (1 / reduce)
        x <- dfm_weight(x, weights = 1 / sqrt(featfreq(x)))
    } else {
        if (weight == "sqrt") {
            x@x <- sqrt(x@x)
        } else {
            x <- dfm_weight(x, scheme = weight)
        }
    }
    if (engine == "RSpectra") {
        result <- RSpectra::svds(as(x, "dgCMatrix"), k = k, nu = 0, nv = k, ...)
    } else if (engine == "rsvd") {
        result <- rsvd::rsvd(as(x, "dgCMatrix"), k = k, nu = 0, nv = k, ...)
    } else {
        result <- irlba::irlba(as(x, "dgCMatrix"), nv = k, right_only = TRUE, ...)
    }
    return(result)
}
