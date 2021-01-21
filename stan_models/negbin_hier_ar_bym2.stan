functions {
  /**
  *
  *
  *
  */
  int num_zero_ts(int[ , ] ts) {
    int N = dims(ts)[1];
    int Nz = 0;

    for (n in 1:N) {
      if (sum(ts[n, :]) == 0) {
        Nz = Nz + 1;
      }
    }
    return Nz;
  }

  /**
  *
  *
  *
  */
  int[] idxs_zero_ts(int[ , ] ts) {
    int N = dims(ts)[1];
    int Nz = num_zero_ts(ts);

    int idxs[Nz] = rep_array(0, Nz);
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
  // General Parameters
  int<lower=1> N;   // Number of counties
  int<lower=1> T;   // Number of time points

  // Response
  int<lower=0> y[N, T];   // Number of deaths per county per time point

  // Dispersion Scales
  real inv_phi_alpha;   // Alpha parameter for phi's gamma prior
  real inv_phi_beta;   // Beta (rate) parameter for phi's gamma prior

  // Timeseries Parameters
  int<lower=0> K_time;   // Numer of timeseries covariates
  matrix<lower=0>[T, K_time] X_time[N];   // Design matrix of timeseries covariates for each county

  vector[4 + K_time] mu_betas_center;   // Center for center for hierarchical prior
  vector<lower=0>[4 + K_time] mu_betas_scale;   // Scale for center for hierarchical prior

  vector<lower=0>[4 + K_time] tau_betas_center;   // Center for scale for hierarchical prior
  vector<lower=0>[4 + K_time] tau_betas_scale;   // Scale for scale for hierarchical prior

  // Spatial Parameters
  int<lower=0> N_edges;   //   Total number of edges in adjacencey graph
  int<lower=1, upper=N> node1[N_edges];   // Node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];   // And node1[i] < node2[i]

  int<lower=1> K_spatial;   // Number of spatial covariates
  matrix[N, K_spatial] X_spatial;   // Spatial design matrix

  real<lower=0> spatial_scaling_factor;   // Scale for the variance of the spatial effects (INLA)
  real intercept_center;
  real<lower=0> intercept_scale;
}

transformed data {
  int Nz = num_zero_ts(y);
  int z_idxs[Nz] = idxs_zero_ts(y);

  int P = 4;
  int<lower=0> lags = 7;
  int<lower=1> Ps[lags] = {1, 7, 8, 14, 15, 21, 22};

  matrix<lower=0>[T, lags + K_time] L[N] = rep_array(rep_matrix(0, T, lags + K_time), N);

  for (n in 1:N) {
    L[n, :, (lags+1):] = log(X_time[n] + 1.001);
  }
  
  for (n in 1:N) {
    for (p in 1:lags) {
      int lag = Ps[p];
      L[n, lag+1:, p] = log(to_vector(y[n, :T-lag]) + 1.001);
    }
  }
}

parameters {
  // Dispersion Parameters
  real<lower=0> inv_phi[N];

  // Timeseries Parameters
  real<lower=0> inv_eta;

  matrix[N, P + K_time] raw_betas;

  vector[P + K_time] mu_betas;
  vector<lower=0>[P + K_time] tau_betas;
  corr_matrix[P + K_time] Omega_betas;

  // Spatial Parameters
  vector[K_spatial] alphas;       // covariates

  real<lower=0> sigma_spatial;   // overall standard deviation
  real<lower=0, upper=1> rho;   // Proportion unstructured vs. spatially structured variance

  vector[N] theta_spatial;   // Heterogeneous/unstructured/random effects
  vector[N] phi_spatial;   // Spatially structured effects
}

transformed parameters {
  // Timeseries Parameters
  real eta = 1.0 / inv_eta;
  matrix[N, P + K_time] betas;
  matrix[N, lags + K_time] betas_expanded;

  // Spatial Parameters
  vector[N] convolved_re;

  // Regression Parameters
  real phi[N];
  matrix[N, T] nu;

  // Noncentered Parameterization
  for (n in 1:N) {
    betas[n, :] = mu_betas' + raw_betas[n, :] * quad_form_diag(Omega_betas, tau_betas);
  }

  // Dispersion
  for (n in 1:N) {
    phi[n] = 1.0 / inv_phi[n];
  } 

  // Set degenerate parameters to zero
  betas[z_idxs, 1:P] = rep_matrix(0, Nz, P);

  // Expand Parameters to allow for AR 
  betas_expanded[:, (lags+1):] = betas[:, (P+1):];  // Covariate Regression

  betas_expanded[:, 1:lags] = [betas[:, 1]', 
                                   betas[:, 2]', 
                                   (-1 * betas[:, 1] .* betas[:, 2])', 
                                   betas[:, 3]', 
                                   (-1 * betas[:, 1] .* betas[:, 3])', 
                                   betas[:, 4]', 
                                   (-1 * betas[:, 1] .* betas[:, 4])']';
  
  // Calculate spatial component
  convolved_re =  sqrt(1 - rho) * theta_spatial + sqrt(rho / spatial_scaling_factor) * phi_spatial;

  // Calculate nu
  for (n in 1:N) {
    nu[n] = (L[n] * betas_expanded[n]')';
  }

  nu = nu + rep_matrix(X_spatial * alphas + convolved_re * sigma_spatial, T);
}

model {
  // Dispersion Prior
  inv_phi ~ gamma(inv_phi_alpha, inv_phi_beta);

  // Timeseries: Non-centered Parameterization Priors
  to_vector(raw_betas) ~ normal(0, 1);

  // Timeseries: Heirarchical Priors
  mu_betas ~ normal(mu_betas_center, mu_betas_scale);
  tau_betas ~ normal(tau_betas_center, tau_betas_scale);

  inv_eta ~ normal(0, .3);
  Omega_betas ~ lkj_corr(eta);

  // Spatial: BYM Priors
  target += -0.5 * dot_self(phi_spatial[node1] - phi_spatial[node2]);
  sum(phi_spatial) ~ normal(0, 0.001 * N);  // equivalent to mean(phi) ~ normal(0,0.001)

  theta_spatial ~ normal(0.0, 1.0);
  sigma_spatial ~ normal(0, 1.0);
  rho ~ beta(0.5, 0.5);

  // Spatial: Regression Priors
  alphas[1] ~ normal(intercept_center, intercept_scale);
  alphas[2:] ~ normal(0, 1);

  // Liklihood
  for (n in 1:N) {
    y[n] ~ neg_binomial_2_log(nu[n], phi[n]);
  }
}

generated quantities {
  real y_pred[N, T];
  real log_lik[N, T];

  for (n in 1:N) { 
    for (t in 1:T) {
      log_lik[n, t] = neg_binomial_2_log_lpmf(y[n, t] | nu[n, t], phi[n]);
    }

    for (t in 1:T) {
      if (nu[n, t] < 15 && nu[n, t] > -15) 
        y_pred[n, t] = neg_binomial_2_log_rng(nu[n, t], phi[n]);
      else if (nu[n, t] > 15)
        y_pred[n, t] = neg_binomial_2_log_rng(15,  phi[n]);
      else
        y_pred[n, t] = neg_binomial_2_log_rng(-15, phi[n]);
    }
  }
}
