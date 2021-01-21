data {
  int<lower=1> T;                           // Number of timesteps
  int<lower=0> y[T];                        // Timeseries
  
  vector[1 + 2] betas_center;      // centers for beta priors
  vector<lower=0>[1 + 2] betas_scale;       // scales for beta priors
  real inv_phi_alpha;
  real inv_phi_beta;
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
  vector[1 + P] betas;
  real<lower=0> inv_phi;
}

transformed parameters {
  vector[1 + lags] betas_ext = [betas[1], betas[2], betas[3], -1 * betas[2] * betas[3]]';
  vector[T] nu;
  real phi = 1.0 / inv_phi;

  nu = L * betas_ext;

}

model {
  betas ~ normal(betas_center, betas_scale);
  inv_phi ~ gamma(inv_phi_alpha, inv_phi_beta);

  y ~ neg_binomial_2_log(nu, phi);
}

generated quantities {
  real y_pred[T];
  vector[T] log_lik;

  for (t in 1:T) {
    log_lik[t] = neg_binomial_2_log_lpmf(y[t] | nu[t], phi);
  }


  for (t in 1:T) {
    if (nu[t] < 15 && nu[t] > -15) 
      y_pred[t] = neg_binomial_2_log_rng(nu[t], phi);
    else if (nu[t] > 15)
      y_pred[t] = neg_binomial_2_log_rng(15,  phi);
    else
      y_pred[t] = neg_binomial_2_log_rng(-15, phi);
  }
}
