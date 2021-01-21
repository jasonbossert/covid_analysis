data {
  int<lower=1> T;                           // Number of timesteps
  int<lower=0> y[T];                        // Timeseries
  
  vector[1 + 2] betas_center;      // centers for beta priors
  vector<lower=0>[1 + 2] betas_scale;       // scales for beta priors
}

transformed data {
  int P = 2;
  int lags = 3;
  int<lower=1> Ps[lags] = {1, 7, 8};

  // Define the design matrix
  matrix<lower=0>[T, 1 + lags] L = rep_matrix(0, T, 1 + lags);
  L[:, 1] = rep_vector(1, T);

  for (p in 1:lags) {
    int lag = Ps[p];
    L[lag + 1:, 1 + p] = log(to_vector(y[:T-lag]) + 1);
  }
}

parameters {
  vector[1 + P] betas;        // +1 to account for intercept
}

transformed parameters {
  vector[1 + lags] betas_ext = [betas[1], betas[2], betas[3], -1 * betas[2] * betas[3]]';
  vector[T] nu;

  nu = L * betas_ext;
}

model {
  // Priors
  betas ~ normal(betas_center, betas_scale);

  // Liklihood
  y ~ poisson_log(nu);
}

generated quantities {
  real y_pred[T];
  vector[T] log_lik;

  for (t in 1:T) {
    log_lik[t] = poisson_log_lpmf(y[t] | nu[t]);
  }


  for (t in 1:T) {
    if (nu[t] < 20 && nu[t] > -20) 
      y_pred[t] = poisson_log_rng(nu[t]);
    else if (nu[t] > 20)
      y_pred[t] = poisson_log_rng(20);
    else
      y_pred[t] = poisson_log_rng(-20);
  }
}
