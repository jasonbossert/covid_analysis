data {
  int<lower=1> T;                           // Number of timesteps
  int<lower=0> y[T];                     // Timeseries
  
  vector<lower=0>[1 + 2] betas_center;      // centers for beta priors
  vector<lower=0>[1 + 2] betas_scale;       // scales for beta priors
}

transformed data {
  int P = 2;
  int lags = 3;
  int<lower=1> Ps[lags] = {1, 7, 8};

  // Definition of sigma for normally distributed errors
  real sigma_scale = sum(sqrt(square(to_vector(y[2:T]) - to_vector(y[1:(T-1)])))) / dims(y)[1];

  // Define the design matrix
  matrix<lower=0>[T, 1 + lags] L = rep_matrix(0, T, 1 + lags);
  L[:, 1] = rep_vector(1, T);

  for (p in 1:lags) {
    int lag = Ps[p];
    L[lag + 1:, 1 + p] = to_vector(y[:T-lag]);
  }
}

parameters {
  real<lower=0> sigma;
  vector[1 + P] betas;    // +1 to account for intercept
}

transformed parameters {
  vector[1 + lags] betas_ext = [betas[1], betas[2], betas[3], -1 * betas[2] * betas[3]]';
  vector[T] nu;

  nu = L * betas_ext;
}

model {
  // Priors
  sigma ~ normal(0, sigma_scale);
  betas ~ normal(betas_center, betas_scale);

  // Liklihood
  to_vector(y) ~ normal(nu, sigma);
}

generated quantities {
  real y_pred[T];
  real log_lik[T];

  y_pred = normal_rng(nu, sigma);
  for (t in 1:T) {
    log_lik[t] = normal_lpdf(y[t] | nu[t], sigma);
  }
}
