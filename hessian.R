library(tidyverse)
library(ggplot2)
library(rstan)
library(Rcpp)

a = 1.5
b = 0.5
sigma = 0.25
N = 20

x = seq(0.0, 1.0, length = N)
(y = sapply(x, function(xv) rnorm(1, a * xv + b, sigma)))

list(x = x, y = y) %>% as.tibble %>%
  ggplot(aes(x, y)) +
  geom_point()

stan_rdump(list("N", "x", "y"), file = "data.dump")

system('/home/bbales2/cmdstan/bin/stanc --allow_undefined models/linear_regression.stan')
system('mv linear_regression_model.cpp linear_regression_model.hpp')

sourceCpp("helper.cpp")
jacobian("data.dump", c(a, b, sigma))
hessian("data.dump", c(a, b, sigma))

model = stan_model("models/linear_regression.stan")
fit = optimizing(model, data = list(N = N, x = x, y = y), hessian = TRUE)

(jac = jacobian("data.dump", fit$par[1:3]))
(hess = hessian("data.dump", fit$par[1:3]))

samples = sampling(model, data = list(N = N, x = x, y = y), chains = 1)

pairs(samples, pars = c("a", "b", "sigma_log"))

eigen(hess)
