// relational_rw2_mortality.stan

data {
  int<lower=1> N;            // Number of age groups
  array[N] int<lower=0> D;   // Observed death counts
  vector[N] log_E;           // Log exposures (the offset)
  vector[N] log_std;         // Log of the standard mortality schedule
}

parameters {
  real alpha;                // Intercept (level shift relative to standard)
  real<lower=0> beta;        // Coefficient on standard schedule (strictly positive!)
  vector[N] gamma;           // Age-specific smooth deviations
  real<lower=0> sigma_gamma; // Standard deviation of the RW2 process 
  real<lower=0> theta;       // Overdispersion parameter
}

transformed parameters {
  vector[N] log_mu;
  
  // The linear predictor: Intercept + (Slope * Standard) + Age Effect + Offset
  log_mu = alpha + beta * log_std + gamma + log_E; 
}

model {
  // 1. Priors for global parameters
  // alpha is 0 if the overall level perfectly matches the standard
  alpha ~ normal(0, 5);             
  
  // beta is constrained to be > 0 in the parameters block.
  // A normal(1, 1) prior acts as a truncated normal, pulling the slope toward 1
  // (meaning the shape perfectly matches the standard) but allowing it to vary.
  beta ~ normal(1, 1);              
  
  theta ~ exponential(0.1);         
  sigma_gamma ~ normal(0, 0.5);     
  
  // 2. The RW2 Prior for the smooth age deviations
  gamma[1] ~ normal(0, 10);
  gamma[2] ~ normal(0, 10);
  
  for (x in 3:N) {
    gamma[x] ~ normal(2 * gamma[x-1] - gamma[x-2], sigma_gamma);
  }
  
  // 3. Identifiability Constraint
  // Forces the RW2 deviations to sum to zero, ensuring alpha and beta 
  // remain interpretable as the global baseline adjustments.
  sum(gamma) ~ normal(0, 0.001 * N);
  
  // 4. The Likelihood
  D ~ neg_binomial_2_log(log_mu, theta);
}