functions {
  /**
  *
  *
  *
  */
  int num_nonzero_ts(int[ , ] ts) {
    int N = dims(ts)[1];
    int N_nonzero = 0;

    for (n in 1:N) {
      if (sum(ts[n, :]) > 0) {
        N_nonzero = N_nonzero + 1;
      }
    }
    return N_nonzero;
  }

  /**
  *
  *
  *
  */
  int[] idxs_nonzero_ts(int[ , ] ts) {
    int N = dims(ts)[1];
    int N_nonzero = num_nonzero_ts(ts);

    int idxs[N_nonzero] = rep_array(0, N_nonzero);
    int idx = 1;
    for (n in 1:N) {
      if (sum(ts[n, :]) > 0) {
        idxs[idx] = n;
        idx = idx + 1;
      }
    }
    return idxs;
  }
}

data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=0> K;
  int<lower=0> y[N, T];

  matrix<lower=0>[T, K] X[N];

  real<lower=0> mu_scale;
  real<lower=0> phi_scale;
  real<lower=0> betas_scale;
  real<lower=0> disp_scale;
}

transformed data {
  int Nz = num_nonzero_ts(y);
  int nz_idxs[Nz] = idxs_nonzero_ts(y);

  int P = 4;
  int<lower=0> lags = 7;
  int<lower=1> Ps[lags] = {1, 7, 8, 14, 15, 21, 22};

  matrix<lower=0>[T, lags] Y_p[N] = rep_array(rep_matrix(0, T, lags), N);
  
  for (n in 1:N) {
    for (p in 1:lags) {
      int lag = Ps[p];
      Y_p[n, lag+1:, p] = to_vector(y[n, :T-lag]);
    }
  }
}

parameters {
  vector[N] mu;
  real<lower=0> inv_disp[N];

  vector[P] phi_raw[Nz];
  vector[K] betas_raw[N];

  real mu_phi[P];
  real<lower=0> sigma_phi[P];
  real mu_betas[K];
  real<lower=0> sigma_betas[K];
}

transformed parameters {
  vector[lags] lag_coeffs[Nz];
  vector[T] nu[N];
  real disp[N];

  vector[P] phi[Nz];
  vector[K] betas[N];

  for (p in 1:P) {
    phi[:, p] = to_array_1d(mu_phi[p] + sigma_phi[p] * to_vector(phi_raw[:, p]));
  }
  for (k in 1:K) {
    betas[:, k] = to_array_1d(mu_betas[k] + sigma_betas[k] * to_vector(betas_raw[:, k]));
  }

  for (n in 1:N) {
    disp[n] = 1.0 / inv_disp[n];
  } 

  for (n in 1:N) {
    nu[n] = mu[n] + log(X[n] + 1.001) * betas[n];
  }

  for (n in 1:Nz) {
    int idx = nz_idxs[n];
    lag_coeffs[n] = [phi[n, 1], phi[n, 2], -1*phi[n, 1]*phi[n, 2], phi[n, 3], -1*phi[n, 1]*phi[n, 3], phi[n, 4], -1*phi[n, 1]*phi[n, 4]]';
    nu[idx] = nu[idx] + log(Y_p[idx] + 1.001) * lag_coeffs[n];
  }

}

model {
  // Heirarchical Priors
  mu_phi ~ normal(0, phi_scale);
  sigma_phi ~ normal(0, phi_scale);
  mu_betas ~ normal(0, betas_scale);
  sigma_betas ~ normal(0, betas_scale);

  // Regular Prior
  mu ~ normal(0, mu_scale);
  inv_disp ~ normal(0, disp_scale);

  // Center Parameterization Priors
  for (n in 1:Nz) {
    phi_raw[n] ~ normal(0, 1);
  }
  for (n in 1:N) {
    betas_raw[n] ~ normal(0, 1);
  }

  // Liklihood
  for (n in 1:N) {
    y[n] ~ neg_binomial_2_log(nu[n], disp[n]);
  }
}

generated quantities {
  real y_pred[N, T];
  real log_lik[N, T];

  for (n in 1:N) { 
    for (t in 1:T) {
      log_lik[n, t] = neg_binomial_2_log_lpmf(y[n, t] | nu[n, t], disp[n]);
    }

    for (t in 1:T) {
      if (nu[n, t] < 15 && nu[n, t] > -15) 
        y_pred[n, t] = neg_binomial_2_log_rng(nu[n, t], disp[n]);
      else if (nu[n, t] > 15)
        y_pred[n, t] = neg_binomial_2_log_rng(15,  disp[n]);
      else
        y_pred[n, t] = neg_binomial_2_log_rng(-15, disp[n]);
    }
  }
}
