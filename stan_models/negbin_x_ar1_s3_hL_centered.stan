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

  /**
  *
  *
  *
  */
  int[] idxs_zero_ts(int[ , ] ts) {
    int N = dims(ts)[1];
    int N_zero = N - num_nonzero_ts(ts);

    int idxs[N_zero] = rep_array(0, N_zero);
    int idx = 1;
    for (n in 1:N) {
      if (sum(ts[n, :]) == 0) {
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
  int z_idxs[N-Nz] = idxs_zero_ts(y);

  int P = 4;
  int<lower=0> lags = 7;
  int<lower=1> Ps[lags] = {1, 7, 8, 14, 15, 21, 22};

  matrix<lower=0>[T, 1 + lags + K] L[N] = rep_array(rep_matrix(0, T, 1 + lags + K), N);

  L[:, :, 1] = rep_array(rep_vector(1, T), N);
  for (n in 1:N) {
    L[n, :, (lags+2):] = log(X[n] + 1.001);
  }
  
  for (n in 1:N) {
    for (p in 1:lags) {
      int lag = Ps[p];
      L[n, lag+1:, p + 1] = log(to_vector(y[n, :T-lag]) + 1.001);
    }
  }
}

parameters {
  real<lower=0> inv_disp[N];

  vector[N] mu;
  vector[P] phi[Nz];
  vector[K] betas[N];

  real mu_mu;
  real<lower=0> sigma_mu;
  real mu_phi[P];
  real<lower=0> sigma_phi[P];
  real mu_betas[K];
  real<lower=0> sigma_betas[K];
}

transformed parameters {
  vector[lags] lag_coeffs[Nz];
  vector[T] nu[N];
  real disp[N];

  vector[1 + P + K] pars[N];
  vector[1 + lags + K] pars_expanded[N];

  // Dispersion
  for (n in 1:N) {
    disp[n] = 1.0 / inv_disp[n];
  } 

  // Lag Coeffs
  for (n in 1:Nz) {
    lag_coeffs[n] = [phi[n, 1], phi[n, 2], -1*phi[n, 1]*phi[n, 2], phi[n, 3], -1*phi[n, 1]*phi[n, 3], phi[n, 4], -1*phi[n, 1]*phi[n, 4]]';
  }

  // Combine parameters
  pars[:, 1] = to_array_1d(mu);
  pars[nz_idxs, 2:(P+1)] = phi;
  pars[z_idxs, 2:(P+1)] = rep_array(rep_vector(0, P), N-Nz);
  pars[:, (P+2):] = betas;

  pars_expanded[:, 1] = to_array_1d(mu);
  pars_expanded[nz_idxs, 2:(lags+1)] = lag_coeffs;
  pars_expanded[z_idxs, 2:(lags+1)] = rep_array(rep_vector(0, lags), N-Nz);
  pars_expanded[:, (lags+2):] = betas;

  // Calculate nu
  for (n in 1:N) {
    nu[n] = L[n] * pars_expanded[n];
  }
}

model {
  // Heirarchical Priors
  mu_mu ~ normal(0, mu_scale);
  sigma_mu ~ normal(0, mu_scale);
  mu_phi ~ normal(0, phi_scale);
  sigma_phi ~ normal(0, phi_scale);
  mu_betas ~ normal(0, betas_scale);
  sigma_betas ~ normal(0, betas_scale);

  // Regular Priors
  inv_disp ~ normal(0, disp_scale);
  mu ~ normal(mu_mu, sigma_mu);
  for (n in 1:Nz) {
    phi[n] ~ normal(mu_phi, sigma_phi);
  }
  for (n in 1:N) {
    betas[n] ~ normal(mu_betas, sigma_betas);
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
