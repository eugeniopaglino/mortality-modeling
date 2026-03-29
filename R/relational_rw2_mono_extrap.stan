// relational_rw2_mono_extrap.stan

data {
  int<lower=1> N_train;                
  int<lower=1> N_total;                
  array[N_train] int<lower=0> D_train; 
  vector[N_total] log_E;               
  vector[N_total] log_std;             
  int<lower=1> mono_start_idx;         
}

parameters {
  real alpha;                
  real<lower=0> beta;        
  real<lower=0> sigma_gamma; 
  real<lower=0> theta;       
  
  // Model gamma directly across all ages
  vector[N_total] gamma; 
}

transformed parameters {
  vector[N_total] log_mu;
  
  // The linear predictor for all ages
  log_mu = alpha + beta * log_std + gamma + log_E; 
}

model {
  // 1. Global Priors
  alpha ~ normal(0, 5);             
  beta ~ normal(1, 1);              
  theta ~ exponential(0.1);         
  sigma_gamma ~ normal(0, 0.5);     
  
  // 2. The RW2 Prior (Applied smoothly to ALL ages!)
  gamma[1] ~ normal(0, 10);
  gamma[2] ~ normal(0, 10);
  
  for (x in 3:N_total) {
    gamma[x] ~ normal(2 * gamma[x-1] - gamma[x-2], sigma_gamma);
  }
  
  // 3. The Soft Monotonicity Penalty
  // Only applied to ages after mono_start_idx (e.g., age 40+)
  for (x in mono_start_idx:N_total) {
    real current_rate = alpha + beta * log_std[x] + gamma[x];
    real prev_rate    = alpha + beta * log_std[x-1] + gamma[x-1];
    
    // If the sampler tries to dip the mortality curve, slap it with a massive penalty
    if (current_rate < prev_rate) {
      target += -100 * (prev_rate - current_rate); 
    }
  }
  
  // 4. Identifiability
  // Constrain the sum of the training deviations to be zero
  sum(gamma[1:N_train]) ~ normal(0, 0.001 * N_train);
  
  // 5. Likelihood
  // Only evaluates the data we allowed it to "see"
  D_train ~ neg_binomial_2_log(log_mu[1:N_train], theta);
}