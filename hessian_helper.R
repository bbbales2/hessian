library(rstan)
library(Rcpp)

load_model = function(model_file, params_list) {
  envir = list2env(params_list)
  stan_rdump(names(params_list), file = "data.dump", envir = envir)
  
  if(system(paste0('/home/bbales2/cmdstan/bin/stanc --allow_undefined --name=linear_regression_model ', model_file)) != 0) {
    stop("model '", model_file,"' does not exist")
  }
  system('mv linear_regression_model.cpp linear_regression_model.hpp')
  
  sourceCpp("helper.cpp")
  set_data("data.dump")
}