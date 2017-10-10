data {
  int<lower = 1> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real a;
  real b;
  real sigma_log;
}

transformed parameters {
  real sigma = exp(sigma_log);
}

model {
  y ~ normal(a * x + b, sigma);
  
  target += sigma_log;
}