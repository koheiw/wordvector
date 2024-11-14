get_threads <- function() {
    
    # respect other settings
    default <- c("tbb" = as.integer(Sys.getenv("RCPP_PARALLEL_NUM_THREADS")),
                 "omp" = as.integer(Sys.getenv("OMP_THREAD_LIMIT")),
                 "max" = cpp_get_max_thread())
    default <- unname(min(default, na.rm = TRUE))
    suppressWarnings({
        value <- as.integer(getOption("wordvector_threads", default))
    })
    if (length(value) != 1 || is.na(value)) {
        stop("wordvector_threads must be an integer")
    }
    return(value)
}