library(tidyverse)
library(ggplot2)
library(rstan)
library(Rcpp)

source("hessian_helper.R")

N = 20

x = seq(0.0, 1.0, length = N)
(y = sapply(x, function(xv) rnorm(1, a * xv + b, sigma)))

list(x = x, y = y) %>% as.tibble %>%
  ggplot(aes(x, y)) +
  geom_point()

load_model("models/linear_regression.stan", list(N = N, x = x, y = y))

out = list()
for(i in 1:2) {
  a = rnorm(1)
  b = rnorm(1)
  sigma = exp(rnorm(1))
  out[[i]] = check_constants_are_constant(c(a, b, sigma))
}
