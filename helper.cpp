#include <Rcpp.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(rstan)]]

#include "stan/math.hpp"
#include "stan/math/fwd/mat.hpp"
#include "linear_regression_model.hpp"
#include <stan/io/dump.hpp>
#include <iostream>
#include <fstream>

stan::io::dump readData(std::string filename) {
  std::ifstream dumpFile(filename.c_str());
  
  if(!dumpFile.good()) {
    dumpFile.close();
    throw std::domain_error("Error opening dump file");
  }
  
  stan::io::dump dFile(dumpFile);

  dumpFile.close();
  return dFile;
}

static linear_regression_model_namespace::linear_regression_model *model = NULL;

// [[Rcpp::export]]
void set_data(std::string filename) {
  stan::io::dump dfile = readData(filename);
  if(model != NULL) {
    delete model;
  }
  model = new linear_regression_model_namespace::linear_regression_model(dfile, &Rcpp::Rcout);
}

// [[Rcpp::export]]
Rcpp::List jacobian(std::vector<double> params) {
  using namespace Rcpp;
  using stan::math::var;
  using stan::math::fvar;
  
  if(model == NULL) {
    throw std::invalid_argument("Must call set_data before jacobian");
  }
  
  NumericVector jac(params.size());
  
  std::vector<var> params_r;
  std::vector<int> params_i({});
  
  params_r.insert(params_r.begin(), params.begin(), params.end());
    
  var lp = model->log_prob<true, true, var>(params_r, params_i, &Rcpp::Rcout);
  
  lp.grad();
  
  for(size_t i = 0; i < params_r.size(); i++) {
    jac(i) = params_r[i].adj();
  }
  
  List out;
  
  out["u"] = lp.val();
  out["jac"] = jac;

  stan::math::recover_memory();
  
  return out;
}

// [[Rcpp::export]]
Rcpp::List hessian(std::vector<double> params) {
  using namespace Rcpp;
  using stan::math::var;
  using stan::math::fvar;
  
  if(model == NULL) {
    throw std::invalid_argument("Must call set_data before jacobian");
  }
  
  NumericVector jac(params.size());
  NumericMatrix hess(params.size(), params.size());
  
  std::vector<int> params_i({});
  
  double lp_ = 0.0;
  
  for(size_t i = 0; i < params.size(); i++) {
    std::vector<fvar<var> > params_r;
    for(auto v : params)
      params_r.push_back(v);
    
    params_r[i].d_ = 1.0;
    fvar<var> lp = model->log_prob<true, true, fvar<var> >(params_r, params_i, &Rcpp::Rcout);

    jac(i) = lp.tangent().val();
    
    lp.d_.grad();
    for(size_t j = 0; j < params_r.size(); j++) {
      hess(i, j) = params_r[j].val().adj();
    }
    
    lp_ = lp.val().val(); // Same every time
  }
  
  List out;
  
  out["u"] = lp_;
  out["jac"] = jac;
  out["hess"] = hess;
  
  stan::math::recover_memory();
  
  return out;
}
