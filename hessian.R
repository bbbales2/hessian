library(tidyverse)
library(ggplot2)
library(rstan)
library(Rcpp)

source("hessian_helper.R")

a = 1.5
b = 0.5
sigma = 0.25
N = 20

x = seq(0.0, 1.0, length = N)
(y = sapply(x, function(xv) rnorm(1, a * xv + b, sigma)))

list(x = x, y = y) %>% as.tibble %>%
  ggplot(aes(x, y)) +
  geom_point()

load_model("models/linear_regression.stan", list(N = N, x = x, y = y))

jacobian(c(a, b, sigma))
h = hessian(c(a, b, sigma))
vec = rnorm(3)
hv = hessian_vector(c(a, b, sigma), vec)
solve(h$hess, vec)
hessian_solve(c(a, b, sigma), vec, rep(1, 3), 1e-1)

model = stan_model("models/linear_regression.stan")
fit = optimizing(model, data = list(N = N, x = x, y = y), hessian = TRUE)

(jac = jacobian(fit$par[1:3]))
(hess = hessian(fit$par[1:3]))

samples = sampling(model, data = list(N = N, x = x, y = y), chains = 1)

pairs(samples, pars = c("a", "b", "sigma_log"))

eigen(hess)
