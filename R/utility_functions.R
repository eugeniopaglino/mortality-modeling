library(MASS)
library(matrixStats)
library(data.table)

plot_posterior_predictive <- function(model, data, n_sims = 1000, plot_title = "Posterior Predictive Check") {
  
  # 1. Make a local copy of the data so we don't accidentally modify the global dataset
  dt <- copy(data)
  
  # 2. Extract model parameters
  betas <- coef(model)
  V <- vcov(model)
  theta <- model$theta
  
  # 3. Simulate parameter uncertainty
  sim_betas <- mvrnorm(n_sims, mu = betas, Sigma = V)
  
  # 4. Dynamically build the model matrix from whatever formula the model used!
  # delete.response() removes 'Dx' from the formula so we just get the right-hand side predictors
  form <- delete.response(terms(model))
  X_mat <- model.matrix(form, data = dt)
  offset_val <- log(dt$Ex)
  
  # 5. Vectorized calculations for speed
  log_mu_mat <- X_mat %*% t(sim_betas) + offset_val
  mu_mat <- exp(log_mu_mat)
  
  # Simulate counts
  sim_y_vec <- rnbinom(length(mu_mat), size = theta, mu = mu_mat)
  sim_y <- matrix(sim_y_vec, nrow = nrow(dt), ncol = n_sims)
  
  # Convert directly to log rates (using pmax to prevent log(0) errors!)
  sim_log_rates <- log(pmax(sim_y, 0.5) / dt$Ex)
  
  # 6. Extract Quantiles
  dt[, c("pred_log_mx_lower", "pred_log_mx_median", "pred_log_mx_upper") := 
       as.data.table(rowQuantiles(sim_log_rates, probs = c(0.025, 0.5, 0.975)))]
  
  # 7. Dynamically calculate y-axis limits so no data gets cut off
  y_lims <- range(c(log(dt$Mx), dt$pred_log_mx_lower, dt$pred_log_mx_upper), na.rm = TRUE)
  
  # 8. Plot
  plot(dt$x, log(dt$Mx), 
       pch = 16, col = rgb(0.2, 0.2, 0.2, alpha = 0.5),
       ylim = y_lims,
       xlab = "Age (x)", ylab = expression(log(m(x))),
       main = plot_title,
       bty = "l")
  
  lines(dt$x, dt$pred_log_mx_lower, col = "red", lwd = 1.5, lty = 2)
  lines(dt$x, dt$pred_log_mx_upper, col = "red", lwd = 1.5, lty = 2)
  lines(dt$x, dt$pred_log_mx_median, col = "red", lwd = 2)
  
  legend("topleft", 
         legend = c("Observed Log Rates", "Median Prediction", "95% Prediction Interval"), 
         pch = c(16, NA, NA), 
         lty = c(NA, 1, 2), 
         col = c(rgb(0.2, 0.2, 0.2, alpha = 0.5), "red", "red"), 
         lwd = c(NA, 2, 1.5),
         bty = "n")
}

plot_posterior_extrapolation <- function(model, full_data, fit_max_age, n_sims = 1000, plot_title = "Posterior Extrapolation") {
  
  # 1. Make a local copy of the FULL data
  dt <- copy(full_data)
  
  # 2. Extract model parameters (fitted on the restricted data)
  betas <- coef(model)
  V <- vcov(model)
  theta <- model$theta
  
  # 3. Simulate parameter uncertainty
  sim_betas <- mvrnorm(n_sims, mu = betas, Sigma = V)
  
  # 4. Dynamically build the model matrix for the FULL age range
  form <- delete.response(terms(model))
  X_mat <- model.matrix(form, data = dt)
  offset_val <- log(dt$Ex)
  
  # 5. Vectorized calculations for speed
  log_mu_mat <- X_mat %*% t(sim_betas) + offset_val
  mu_mat <- exp(log_mu_mat)
  
  # Simulate counts
  sim_y_vec <- rnbinom(length(mu_mat), size = theta, mu = mu_mat)
  sim_y <- matrix(sim_y_vec, nrow = nrow(dt), ncol = n_sims)
  
  # Convert directly to log rates
  sim_log_rates <- log(pmax(sim_y, 0.5) / dt$Ex)
  
  # 6. Extract Quantiles
  dt[, c("pred_log_mx_lower", "pred_log_mx_median", "pred_log_mx_upper") := 
       as.data.table(rowQuantiles(sim_log_rates, probs = c(0.025, 0.5, 0.975)))]
  
  # 7. Dynamically calculate y-axis limits so extreme extrapolations fit on screen
  y_lims <- range(c(log(dt$Mx), dt$pred_log_mx_lower, dt$pred_log_mx_upper), na.rm = TRUE)
  
  # 8. Plot
  plot(dt$x, log(dt$Mx), 
       pch = 16, col = rgb(0.2, 0.2, 0.2, alpha = 0.5),
       ylim = y_lims,
       xlab = "Age (x)", ylab = expression(log(m(x))),
       main = plot_title,
       bty = "l")
  
  # Add the vertical line to separate interpolation from extrapolation
  abline(v = fit_max_age, col = "blue", lty = 3, lwd = 2)
  
  lines(dt$x, dt$pred_log_mx_lower, col = "red", lwd = 1.5, lty = 2)
  lines(dt$x, dt$pred_log_mx_upper, col = "red", lwd = 1.5, lty = 2)
  lines(dt$x, dt$pred_log_mx_median, col = "red", lwd = 2)
  
  legend("topleft", 
         legend = c("Observed Log Rates", "Median Prediction", "95% Interval", "End of Training Data"), 
         pch = c(16, NA, NA, NA), 
         lty = c(NA, 1, 2, 3), 
         col = c(rgb(0.2, 0.2, 0.2, alpha = 0.5), "red", "red", "blue"), 
         lwd = c(NA, 2, 1.5, 2),
         bty = "n")
}

library(mgcv)

plot_gam_extrapolation <- function(model, full_data, fit_max_age, n_sims = 1000, plot_title = "Extrapolation: GAM (Penalized Spline)") {
  
  # 1. Make a local copy of the FULL data
  dt <- copy(full_data)
  
  # 2. Extract model parameters
  betas <- coef(model)
  V <- vcov(model) # This automatically extracts the Bayesian posterior covariance matrix for the smooths!
  
  # 3. Extract theta from the mgcv Negative Binomial family
  # Passing TRUE returns the parameter on the standard (untransformed) scale
  theta <- model$family$getTheta(TRUE) 
  
  # 4. Simulate parameter uncertainty
  sim_betas <- mvrnorm(n_sims, mu = betas, Sigma = V)
  
  # 5. Build the matrix using the "lpmatrix" method (Crucial for GAMs!)
  # This maps the new age values precisely to the spline basis functions built during training
  X_mat <- predict(model, newdata = dt, type = "lpmatrix")
  offset_val <- log(dt$Ex)
  
  # 6. Vectorized calculations
  log_mu_mat <- X_mat %*% t(sim_betas) + offset_val
  mu_mat <- exp(log_mu_mat)
  
  # 7. Simulate counts and convert to log rates
  sim_y_vec <- rnbinom(length(mu_mat), size = theta, mu = mu_mat)
  sim_y <- matrix(sim_y_vec, nrow = nrow(dt), ncol = n_sims)
  sim_log_rates <- log(pmax(sim_y, 0.5) / dt$Ex)
  
  # 8. Extract Quantiles
  dt[, c("pred_log_mx_lower", "pred_log_mx_median", "pred_log_mx_upper") := 
       as.data.table(rowQuantiles(sim_log_rates, probs = c(0.025, 0.5, 0.975)))]
  
  # 9. Dynamically calculate y-axis limits
  y_lims <- range(c(log(dt$Mx), dt$pred_log_mx_lower, dt$pred_log_mx_upper), na.rm = TRUE)
  
  # 10. Plot
  plot(dt$x, log(dt$Mx), 
       pch = 16, col = rgb(0.2, 0.2, 0.2, alpha = 0.5),
       ylim = y_lims,
       xlab = "Age (x)", ylab = expression(log(m(x))),
       main = plot_title,
       bty = "l")
  
  # Vertical line separating training from extrapolation
  abline(v = fit_max_age, col = "blue", lty = 3, lwd = 2)
  
  lines(dt$x, dt$pred_log_mx_lower, col = "red", lwd = 1.5, lty = 2)
  lines(dt$x, dt$pred_log_mx_upper, col = "red", lwd = 1.5, lty = 2)
  lines(dt$x, dt$pred_log_mx_median, col = "red", lwd = 2)
  
  legend("topleft", 
         legend = c("Observed Log Rates", "Median Prediction", "95% Interval", "End of Training Data"), 
         pch = c(16, NA, NA, NA), 
         lty = c(NA, 1, 2, 3), 
         col = c(rgb(0.2, 0.2, 0.2, alpha = 0.5), "red", "red", "blue"), 
         lwd = c(NA, 2, 1.5, 2),
         bty = "n")
}