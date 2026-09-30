wordvector <- function(x, dim = 50, type = c("cbow", "sg", "dm", "dbow"), 
                       doc2vec = FALSE, 
                       min_count = 5, window = ifelse(type == "sg", 10, 5), 
                       iter = 10, alpha = 0.05, model = NULL, 
                       use_ns = TRUE, ns_size = 5, sample = 0.001, tolower = TRUE,
                       include_data = FALSE, verbose = FALSE, ..., 
                       normalize = FALSE) {
    
    opt <- quanteda_options("verbose")
    quanteda_options(verbose = FALSE)
    
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
    
    temp <- cpp_word2vec(x, model, size = dim, window = window,
                         sample = sample, withHS = !use_ns, negative = ns_size, 
                         threads = get_threads(), iterations = iter,
                         alpha = alpha, 
                         type = match(type, c("cbow", "sg", "dm", "dbow", "dbow2")), 
                         normalize = FALSE, 
                         doc2vec = doc2vec,
                         verbose = verbose)
    
    if (!is.null(temp$message))
        stop("Failed to train word2vec (", temp$message, ")")
    
    if (doc2vec) {
        result <- build_doc2vec(
            model = temp,
            docname = docnames(x),
            type = type,
            min_count = min_count,
            tolower = tolower,
            concatenator = meta(x, field = "concatenator", type = "object"),
            docvars = attr(x, "docvars"),
            ntoken = ntoken(x, remove_padding = TRUE),
            call = try(match.call(sys.function(-2), call = sys.call(-2)), silent = TRUE)
        )
    } else {
        result <- build_word2vec(
            model = temp,
            type = type,
            min_count = min_count,
            tolower = tolower,
            concatenator = meta(x, field = "concatenator", type = "object"),
            call = try(match.call(sys.function(-2), call = sys.call(-2)), silent = TRUE)
        )
    }
    if (include_data) # NOTE: consider removing
        result$data <- y
    quanteda_options(verbose = opt) # restore
    return(result)
}