# This example assumes time-invariant measurement errors across time points.
# You can revise the code if you assume time-varying measurement errors across time points.

rm(list=ls())
library(MASS)
library(truncnorm)
library(rstan)
library(bayesplot)
library(lavaan)
library(splines)
library(ggplot2)
library(loo)
library(rstan)
library(ggplot2)
library(dplyr)
library(splines)
library(plotrix)
library(coda)
library(rjags)

#setwd(your_file_path)
#data <- read.csv("your_csv_file.csv")

# Assuming the data include 4 repeated measures
head(data)

# Data frame should be looked like this:
# y1 y2 y3 y4
# 1 29 35 44 46
# 2  3 22 26 28
# 3 29 29 32 37
# 4 42 39 44 46
# 5 24 22 28 30
# 6 14 23 30 34


n1 <- length(data$y1)


# Spaghetti plot
sample_data <- data[sample(1:(n1), 20), ]
sample_data$id <- 1:nrow(sample_data)


ndat=reshape(sample_data, idvar="case", timevar="time", v.names="values", 
             varying=list(c("y1","y2","y3","y4")), times=c(1, 2, 3, 4),
             direction="long")

ndat = ndat[order(ndat$case), ]
row.names(ndat)<-NULL

plot(ndat$time,ndat$values,type="n",xlab="Trial",
     ylab="Planes landed",
     bty="l",cex.lab=1,pch=21, xaxt="n", yaxt="n",
     xlim=c(1, 4),ylim=c(0, 55),
     main = bquote("Spaghetti Plot for Sample Data (" *italic("n") * "= 20)"), cex.main=1.25)

points(ndat$time,ndat$values,pch=16,col="black",cex=1)
for(i in unique(ndat$case)){
  lines(ndat$time[ndat$case==i],ndat$values[ndat$case==i],type="l",lty=1,lwd=1.5)
}

library(plotrix)
axis(2, at=seq(0, 55, 5), labels=seq(0, 55, 5), tick=T)
axis(1, at=seq(1, 4, 1), labels=seq(2, 5, 1), tick=T)

dev.off()

# MCMC sampling
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

N <- n1 # number of individuals
T <- 4  # number of time points
Y <- as.data.frame(data)

# Define the latent growth model in lavaan 
lgm_model <- "
    i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
    s =~ 1*y2 + 2*y3 + 3*y4
    i ~ 1
    s ~ 1
    y1 ~ 0*1
    y2 ~ 0*1
    y3 ~ 0*1
    y4 ~ 0*1
    i ~~ s
    "

# Fit the latent growth model using lavaan
fit_lgm <- growth(lgm_model, data = Y)
summary(fit_lgm)

# Extract estimates from lavaan to use as priors
lgm_estimates <- parameterEstimates(fit_lgm)
mu_intercept <- lgm_estimates$est[lgm_estimates$lhs == "i" & lgm_estimates$op == "~1"]
mu_slope <- lgm_estimates$est[lgm_estimates$lhs == "s" & lgm_estimates$op == "~1"]
sigma_intercept <- sqrt(lgm_estimates$est[lgm_estimates$lhs == "i" & lgm_estimates$op == "~~" & lgm_estimates$rhs == "i"])
sigma_slope <- sqrt(lgm_estimates$est[lgm_estimates$lhs == "s" & lgm_estimates$op == "~~" & lgm_estimates$rhs == "s"])
sigma_residual <- sqrt(lgm_estimates$est[lgm_estimates$lhs == "y1" & lgm_estimates$op == "~~" & lgm_estimates$rhs == "y1"])

# Prepare the data for Stan
time <- 1:T
stan_data <- list(
  N = nrow(data),
  T = ncol(data),
  y = as.matrix(data),
  time = time,
  mu_intercept_prior = mu_intercept,
  mu_slope_prior = mu_slope,
  sigma_intercept_prior = sigma_intercept,
  sigma_slope_prior = sigma_slope,
  sigma_residual_prior = sigma_residual
)


lgm_stan_model <- "
data {
  int<lower=0> N;
  int<lower=0> T;
  array[N, T] real y;
  array[T] real time;
  real mu_intercept_prior;
  real mu_slope_prior;
  real<lower=0> sigma_intercept_prior;
  real<lower=0> sigma_slope_prior;
  real<lower=0> sigma_residual_prior;
}

parameters {
  array[N] vector[2] alpha;
  vector<lower=0>[2] tau;
  corr_matrix[2] Omega;

  real<lower=0> sigma_measurement;

  real mu_intercept;
  real mu_slope;
  matrix[N, T-1] delta;
}

transformed parameters {
  cov_matrix[2] Sigma = quad_form_diag(Omega, tau);
}

model {
  vector[2] mu;

  mu_intercept ~ normal(mu_intercept_prior, 5);
  mu_slope ~ normal(mu_slope_prior, 1);
  sigma_measurement ~ student_t(3, 0, sigma_residual_prior);

  tau[1] ~ student_t(3, 0, sigma_intercept_prior);
  tau[2] ~ student_t(3, 0, sigma_slope_prior);
  Omega ~ lkj_corr(2);

  mu[1] = mu_intercept;
  mu[2] = mu_slope;

  for (n in 1:N) {
    alpha[n] ~ multi_normal(mu, Sigma);
  }

  for (t in 1:(T-1)) {
    delta[,t] ~ normal(0, tau[2] / sqrt(T-1));
  }

  // Likelihood
  for (i in 1:N) {
    real latent_y;
    latent_y = alpha[i, 1];

    y[i,1] ~ normal(latent_y, sigma_measurement);
    for (t in 2:T) {
      latent_y = latent_y + alpha[i, 2] * (time[t] - time[t-1]) + delta[i, t-1];
      y[i,t] ~ normal(latent_y, sigma_measurement);
    }
  }
}

generated quantities {
  array[N, T] real y_pred;
  real sigma_intercept = tau[1];
  real sigma_slope = tau[2];
  real rho_intercept_slope = Omega[1,2];
  real cov_intercept_slope = tau[1] * tau[2] * Omega[1,2];

  for (i in 1:N) {
    real latent_y;
    latent_y = alpha[i, 1];

    y_pred[i,1] = normal_rng(latent_y, sigma_measurement);
    for (t in 2:T) {
      latent_y = latent_y + alpha[i, 2] * (time[t] - time[t-1]) + delta[i, t-1];
      y_pred[i,t] = normal_rng(latent_y, sigma_measurement);
    }
  }
}

"
writeLines(lgm_stan_model, "blgm.stan")

# Fit the LGM model using Stan
fit <- stan(file = "blgm.stan", data = stan_data,
            iter = 4000, warmup = 2000, chains = 3, cores = 4)

# Extract full samples from the LGM fit
samples <- rstan::extract(fit)

intercepts_matrix <- samples$alpha[, , 1]
slopes_matrix <- samples$alpha[, , 2]

# To use the full MCMC samples, we'll reshape the data
n_iter <- dim(samples$alpha)[1]
N <- dim(samples$alpha)[2]


# Optionally, select a subset of iterations to reduce computational load
# For example, select 400 iterations evenly spaced
selected_iterations <- seq(1, n_iter, length.out = 400)
intercept_samples <- samples$alpha[selected_iterations, , 1]
slope_samples <- samples$alpha[selected_iterations, , 2]

# Flatten the samples into vectors
intercept_vector <- as.vector(t(intercept_samples))
slope_vector <- as.vector(t(slope_samples))

# Create the spline basis using the intercept samples
n_knots <- 5
knots <- quantile(intercept_vector, probs = seq(0, 1, length.out = n_knots + 2)[-c(1, n_knots + 2)])
boundary_knots <- range(intercept_vector)
X <- ns(intercept_vector, knots = knots, Boundary.knots = boundary_knots, intercept = TRUE)


# Prepare the data for the P-spline model in Stan
stan_data_pspline <- list(
  N = length(intercept_vector),
  K = ncol(X),
  X = X,
  y = slope_vector,
  x = intercept_vector,
  mu_intercept_prior = mean(samples$mu_intercept[selected_iterations]),
  mu_slope_prior = mean(samples$mu_slope[selected_iterations]),
  
  sigma_intercept_prior = mean(samples$tau[selected_iterations, 1]),
  sigma_slope_prior = mean(samples$tau[selected_iterations, 2]),
  
  sigma_residual_prior = mean(samples$sigma_measurement[selected_iterations]),
  smoothness_alpha = 1,
  smoothness_beta = 0.1
)

# Write the Stan model for the natural cubic spline LCSM
writeLines(
  '
data {
  int<lower=1> N;  // number of observations
  int<lower=1> K;  // number of basis functions
  matrix[N, K] X;  // natural cubic spline basis matrix
  vector[N] y;     // response variable (slopes)
  vector[N] x;     // predictor variable (intercepts)
  
  // Prior parameters from LCSM
  real mu_intercept_prior;
  real mu_slope_prior;
  real<lower=0> sigma_intercept_prior;
  real<lower=0> sigma_slope_prior;
  real<lower=0> sigma_residual_prior;
  
  // Smoothness parameters
  real<lower=0> smoothness_alpha;
  real<lower=0> smoothness_beta;
}

parameters {
  vector[K] beta;  // spline coefficients
  real<lower=0> sigma;  // residual standard deviation
  real<lower=0> tau;    // precision parameter for smoothness penalty
}

model {
  // Likelihood
  y ~ normal(X * beta, sigma);
  
  // Smoothness penalty (second-order differences)
  for (k in 3:K) {
    target += -0.5 * tau * pow(beta[k] - 2 * beta[k-1] + beta[k-2], 2);
  }
  
  // Priors
  beta ~ normal(0, 10);  // weakly informative prior for spline coefficients
  sigma ~ normal(sigma_residual_prior, sigma_residual_prior / 10);  // informative prior based on LCSM
  tau ~ gamma(smoothness_alpha, smoothness_beta);  // prior for smoothness parameter
}
', "natural_cubic_spline_lgm.stan")

# Fit the P-spline model using Stan
fit_pspline <- stan(file = "natural_cubic_spline_lgm.stan", data = stan_data_pspline,
                    iter = 4000, warmup = 2000, chains = 3, cores = 4)


# Extract posterior samples from the P-spline model
pspline_samples <- rstan::extract(fit_pspline)

# Plotting the results
plot_bayesian_psplines <- function(intercept_vector, slope_vector, pspline_samples, knots, boundary_knots) {
  # Create a data frame for the scatter plot
  df_scatter <- data.frame(intercept = intercept_vector, slope = slope_vector)
  
  # Calculate the posterior mean and credible intervals
  n_posterior <- dim(pspline_samples$beta)[1]
  n_points <- 1000
  x_range <- seq(min(intercept_vector), max(intercept_vector), length.out = n_points)
  X_pred <- ns(x_range, knots = knots, Boundary.knots = boundary_knots, intercept = TRUE)
  
  y_pred <- matrix(nrow = n_posterior, ncol = n_points)
  for (i in 1:n_posterior) {
    y_pred[i, ] <- X_pred %*% pspline_samples$beta[i, ]
  }
  
  mean_pred <- colMeans(y_pred)
  ci_lower <- apply(y_pred, 2, quantile, probs = 0.025)
  ci_upper <- apply(y_pred, 2, quantile, probs = 0.975)
  
  # Create data frames for line and ribbon
  df_line <- data.frame(x = x_range, y = mean_pred)
  df_ribbon <- data.frame(x = x_range, lower = ci_lower, upper = ci_upper)
  
  # Create the plot
  p <- ggplot() +
    geom_point(data = df_scatter, aes(x = intercept, y = slope), alpha = 0.3, color = "gray50") +
    geom_line(data = df_line, aes(x = x, y = y), color = "red", size = 1) +
    geom_ribbon(data = df_ribbon, aes(x = x, ymin = lower, ymax = upper), alpha = 0.2) +
    theme_minimal() +
    labs(x = "Intercept", y = "Slope", 
         title = "Bayesian P-spline Regression",
         subtitle = "Red line: Posterior mean, Shaded area: 95% credible interval")
  
  return(p)
}

p <- plot_bayesian_psplines(
  intercept_vector = intercept_vector,
  slope_vector = slope_vector,
  pspline_samples = pspline_samples,
  knots = knots,
  boundary_knots = boundary_knots
)

print(p)


# ------ Segmented Regression with Estimated Transition Point using JAGS ------

# Prepare data for JAGS
jags_data <- list(
  N = length(intercept_vector),
  x = intercept_vector,
  y = slope_vector,
  min_x = min(intercept_vector),
  max_x = max(intercept_vector)
)

# Define the JAGS model
model_string <- "
model {
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- alpha[segment[i]] + beta[segment[i]] * x[i]
  }

  # Segment assignment based on change point
  for (i in 1:N) {
    segment[i] <- 1 + step(x[i] - cp)
  }

  # Prior for change point
  cp ~ dunif(min_x, max_x)

  # Priors for segment parameters
  for (j in 1:2) {
    alpha[j] ~ dnorm(0, 0.001)
    beta[j] ~ dnorm(0, 0.001)
  }

  # Prior for precision
  tau ~ dgamma(0.001, 0.001)
}
"

initial_values <- list(
  list(cp = mean(intercept_vector), alpha = c(0, 0), beta = c(0, 0), tau = 1),
  list(cp = median(intercept_vector), alpha = c(0, 0), beta = c(0, 0), tau = 1),
  list(cp = quantile(intercept_vector, 0.75), alpha = c(0, 0), beta = c(0, 0), tau = 1)
)

jags_model <- jags.model(textConnection(model_string), data = jags_data, 
                         inits = initial_values, n.chains = 3)

update(jags_model, 1000)
variable_names <- c("cp", "alpha", "beta", "tau")
jags_samples <- coda.samples(jags_model, variable.names = variable_names, n.iter = 5000)

summary(jags_samples)

cp_samples <- as.matrix(jags_samples)[, "cp"]
cp_mean <- mean(cp_samples)
cp_ci <- quantile(cp_samples, c(0.025, 0.975))

alpha_samples <- as.matrix(jags_samples)[, grep("alpha", colnames(as.matrix(jags_samples)))]
beta_samples <- as.matrix(jags_samples)[, grep("beta", colnames(as.matrix(jags_samples)))]

alpha_mean <- apply(alpha_samples, 2, mean)
beta_mean <- apply(beta_samples, 2, mean)


# Plotting the segmented regression
plot_segmented_regression <- function(intercept_vector, slope_vector, alpha_mean, beta_mean, cp_mean, cp_ci) {
  df <- data.frame(intercept = intercept_vector, slope = slope_vector)
  
  # Segment assignment
  df$segment <- ifelse(df$intercept < cp_mean, 1, 2)
  
  # Create regression lines for each segment
  x_range1 <- seq(min(df$intercept[df$segment == 1]), max(df$intercept[df$segment == 1]), length.out = 50)
  x_range2 <- seq(min(df$intercept[df$segment == 2]), max(df$intercept[df$segment == 2]), length.out = 50)
  
  y_pred1 <- alpha_mean[1] + beta_mean[1] * x_range1
  y_pred2 <- alpha_mean[2] + beta_mean[2] * x_range2
  
  # Plot
  p <- ggplot(df, aes(x = intercept, y = slope)) +
    geom_point(alpha = 0.3, color = "gray50") +
    geom_vline(xintercept = cp_mean, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = cp_ci, color = "green", linetype = "dotted", size = 0.5) +
    geom_line(data = data.frame(x = x_range1, y = y_pred1), aes(x = x, y = y), color = "blue", size = 1) +
    geom_line(data = data.frame(x = x_range2, y = y_pred2), aes(x = x, y = y), color = "blue", size = 1) +
    annotate("text", x = cp_mean, y = max(df$slope), label = "Estimated Transition Point", color = "red", vjust = -1) +
    theme_minimal() +
    labs(x = "Intercept", y = "Slope",
         title = "Segmented Regression with Estimated Transition Point",
         subtitle = "Blue lines: Segment-wise regression lines\nRed dashed: Transition point mean, Green dotted: 95% CI")
  
  return(p)
}

# Create and display the segmented regression plot
p_segmented <- plot_segmented_regression(
  intercept_vector = intercept_vector,
  slope_vector = slope_vector,
  alpha_mean = alpha_mean,
  beta_mean = beta_mean,
  cp_mean = cp_mean,
  cp_ci = cp_ci
)

print(p_segmented)



# Plotting the results with combined P-splines and transition points
plot_bayesian_psplines_with_transition <- function(intercept_vector, slope_vector, pspline_samples, 
                                                   knots, boundary_knots, cp_mean, cp_ci) {
  df_scatter <- data.frame(intercept = intercept_vector, slope = slope_vector)
  
  # Calculate the posterior mean and credible intervals for P-splines
  n_posterior <- dim(pspline_samples$beta)[1]
  n_points <- 1000
  x_range <- seq(min(intercept_vector), max(intercept_vector), length.out = n_points)
  X_pred <- ns(x_range, knots = knots, Boundary.knots = boundary_knots, intercept = TRUE)
  
  y_pred <- matrix(nrow = n_posterior, ncol = n_points)
  for (i in 1:n_posterior) {
    y_pred[i, ] <- X_pred %*% pspline_samples$beta[i, ]
  }
  
  mean_pred <- colMeans(y_pred)
  ci_lower <- apply(y_pred, 2, quantile, probs = 0.025)
  ci_upper <- apply(y_pred, 2, quantile, probs = 0.975)
  
  # Create data frames for line and ribbon
  df_line <- data.frame(x = x_range, y = mean_pred)
  df_ribbon <- data.frame(x = x_range, lower = ci_lower, upper = ci_upper)
  
  # Create the combined plot
  p <- ggplot() +
    geom_point(data = df_scatter, aes(x = intercept, y = slope), alpha = 0.3, color = "gray50") +
    geom_ribbon(data = df_ribbon, aes(x = x, ymin = lower, ymax = upper), alpha = 0.2) +
    geom_line(data = df_line, aes(x = x, y = y), color = "blue", size = 1) +
    
    geom_vline(xintercept = cp_mean, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = cp_ci, color = "green", linetype = "dotted", size = 0.5) +
    
    annotate("text", x = cp_mean, y = max(slope_vector), 
             label = "Estimated Transition Point", color = "red", vjust = -1) +
    theme_minimal() +
    labs(x = "Intercept", y = "Slope", 
         title = "Bayesian P-spline Regression with Transition Point",
         subtitle = "Red line: P-spline mean fit\nGreen dashed: Transition point\nGreen dotted: 95% CI\nShaded area: 95% credible interval")
  
  return(p)
}

# Create and display the combined plot
p_combined <- plot_bayesian_psplines_with_transition(
  intercept_vector = intercept_vector,
  slope_vector = slope_vector,
  pspline_samples = pspline_samples,
  knots = knots,
  boundary_knots = boundary_knots,
  cp_mean = cp_mean,
  cp_ci = cp_ci
)

print(p_combined)



# Superimposing everything
# Combined the fitted P-splines, transition point estimation results, with segmented regression results
plot_combined_regressions <- function(intercept_vector, slope_vector, 
                                      pspline_samples, knots, boundary_knots,
                                      alpha_mean, beta_mean, cp_mean, cp_ci) {

    df_scatter <- data.frame(intercept = intercept_vector, slope = slope_vector)
  
  # Calculate P-spline predictions
  n_posterior <- dim(pspline_samples$beta)[1]
  n_points <- 1000
  x_range <- seq(min(intercept_vector), max(intercept_vector), length.out = n_points)
  X_pred <- ns(x_range, knots = knots, Boundary.knots = boundary_knots, intercept = TRUE)
  
  y_pred <- matrix(nrow = n_posterior, ncol = n_points)
  for (i in 1:n_posterior) {
    y_pred[i, ] <- X_pred %*% pspline_samples$beta[i, ]
  }
  
  mean_pred <- colMeans(y_pred)
  ci_lower <- apply(y_pred, 2, quantile, probs = 0.025)
  ci_upper <- apply(y_pred, 2, quantile, probs = 0.975)
  
  # Create data frames for P-spline results
  df_pspline <- data.frame(x = x_range, y = mean_pred)
  df_pspline_ci <- data.frame(x = x_range, lower = ci_lower, upper = ci_upper)
  
  # Calculate segmented regression predictions
  x_range1 <- seq(min(intercept_vector), cp_mean, length.out = 100)
  x_range2 <- seq(cp_mean, max(intercept_vector), length.out = 100)
  
  y_pred1 <- alpha_mean[1] + beta_mean[1] * x_range1
  y_pred2 <- alpha_mean[2] + beta_mean[2] * x_range2
  
  df_segment1 <- data.frame(x = x_range1, y = y_pred1)
  df_segment2 <- data.frame(x = x_range2, y = y_pred2)
  
  p <- ggplot() +
    
    geom_point(data = df_scatter, aes(x = intercept, y = slope), 
               alpha = 0.3, color = "gray50") +
    
    geom_ribbon(data = df_pspline_ci, 
                aes(x = x, ymin = lower, ymax = upper),
                alpha = 0.2, fill = "blue") +
    geom_line(data = df_pspline, 
              aes(x = x, y = y, color = "P-spline"),
              size = 1) +
    
    geom_line(data = df_segment1,
              aes(x = x, y = y, color = "Segmented"),
              size = 1) +
    geom_line(data = df_segment2,
              aes(x = x, y = y, color = "Segmented"),
              size = 1) +
    
    geom_vline(xintercept = cp_mean, 
               color = "red", linetype = "dashed", size = 0.8) +
    geom_vline(xintercept = cp_ci[1], 
               color = "green", linetype = "dotted", size = 0.5) +
    geom_vline(xintercept = cp_ci[2], 
               color = "green", linetype = "dotted", size = 0.5) +
    
    scale_color_manual(name = "Regression Type",
                       values = c("P-spline" = "blue", "Segmented" = "orange")) +
    
    theme_minimal() +
    labs(x = "Intercept", y = "Slope",
         title = "Combined P-spline and Segmented Regression Analysis",
         subtitle = paste("Red dashed line: Estimated transition point", 
                          "\nGreen dotted lines: 95% credible interval")) +
    theme(legend.position = "top")
  
  return(p)
}

# Create and display the combined plot
p_combined2 <- plot_combined_regressions(
  intercept_vector = intercept_vector,
  slope_vector = slope_vector,
  pspline_samples = pspline_samples,
  knots = knots,
  boundary_knots = boundary_knots,
  alpha_mean = alpha_mean,
  beta_mean = beta_mean,
  cp_mean = cp_mean,
  cp_ci = cp_ci
)

print(p_combined2)

