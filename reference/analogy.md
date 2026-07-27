# Convert formula to named character vector

Convert a formula to a named character vector in analogy tasks.

## Usage

``` r
analogy(formula)
```

## Arguments

- formula:

  a [formula](https://rdrr.io/r/stats/formula.html) object that defines
  the relationship between words using `+` or `-` operators.

## Value

a named character vector to be passed to
[`similarity()`](https://koheiw.github.io/wordvector/reference/similarity.md).

## See also

[`similarity()`](https://koheiw.github.io/wordvector/reference/similarity.md)

## Examples

``` r
analogy(~ berlin - germany + france)
#>  berlin germany  france 
#>       1      -1       1 
analogy(~ quick - quickly + slowly)
#>   quick quickly  slowly 
#>       1      -1       1 
```
