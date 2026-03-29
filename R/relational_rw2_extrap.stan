// relational_rw2_mortality_extrap.stan

data {
  int<lower=1> N_train;                // Number of age groups for training (e.g., ages 0-80)
  int<lower=1> N_total;                // Total number of age groups (e.g., ages 0-110)
  array[N_train] int<lower=0> D_train; // Observed deaths (ONLY for training ages!)
  vector[N_total] log_E;               // Log exposures (FULL age range)
  vector[N_total] log_std;             // Log standard schedule (FULL age range)
}

parameters {
  real alpha;                
  real<lower=0> beta;        
  vector[N_total] gamma;           // The RW2 vector spans the FULL age range
  real<lower=0> sigma_gamma; 
  real<lower=0> theta;       
}

transformed parameters {
  vector[N_total] log_mu;
  
  // The linear predictor is calculated for ALL ages
  log_mu = alpha + beta * log_std + gamma + log_E; 
}

model {
  // 1. Global Priors
  alpha ~ normal(0, 5);             
  beta ~ normal(1, 1);              
  theta ~ exponential(0.1);         
  sigma_gamma ~ normal(0, 0.5);     
  
  // 2. The RW2 Prior (Runs over the FULL age range)
  gamma[1] ~ normal(0, 10);
  gamma[2] ~ normal(0, 10);
  
  for (x in 3:N_total) {
    gamma[x] ~ normal(2 * gamma[x-1] - gamma[x-2], sigma_gamma);
  }
  
  // 3. Identifiability Constraint
  // We strictly apply this ONLY to the training ages so we don't accidentally
  // pull the extrapolated forecast back toward zero artificially.
  sum(gamma[1:N_train]) ~ normal(0, 0.001 * N_train);
  
  // 4. The Likelihood
  // The model only "sees" the death counts up to N_train!
  D_train ~ neg_binomial_2_log(log_mu[1:N_train], theta);
}