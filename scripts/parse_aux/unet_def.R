# Written by Dánnell Quesada-Chacón

# Until now, only leaky relu (lelu) from the "exotic" activation functions implemented

ConvUnit <- function(x, filters = filters, kernel_size = 3, activation = "relu", BaNorm = TRUE, DropOut = TRUE, spatialDOrate = 0.25, alpha = 0.3) {
  x <- layer_conv_2d(x, filters = filters, kernel_size = c(kernel_size, kernel_size), padding = "same")

  if (BaNorm) {
    x <- x %>% layer_batch_normalization()
  }

  if (activation == "lelu" | activation == "leaky_relu") {
    x <- x %>% layer_activation_leaky_relu(alpha = alpha)
  } else {
    x <- x %>% layer_activation(activation = activation)
  }

  if (DropOut) {
    x <- x %>% layer_spatial_dropout_2d(rate = spatialDOrate)
  }
  return(x)
}

ConvBlock <- function(x, filters = filters, kernel_size = 3, activation = "relu", BaNorm = TRUE, DropOut = TRUE, spatialDOrate = 0.25, alpha = 0.3) {
  x <- x %>%
    ConvUnit(., filters = filters, kernel_size = kernel_size, activation = activation, BaNorm = BaNorm, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha) %>%
    ConvUnit(., filters = filters, kernel_size = kernel_size, activation = activation, BaNorm = BaNorm, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha)
  return(x)
}

# Convolutional 2D stack
conv2d_stack <- function(x, filters = c(50, 25, 1), kernel_size = 3, activation = "relu", alpha = 0.3) {
  layers_stack <- x
  for (i in 1:length(filters)) {
    layers_stack <- layers_stack %>% layer_conv_2d(filters = filters[i], kernel_size = c(kernel_size, kernel_size), padding = "same")

    if (activation == "lelu" | activation == "leaky_relu") {
      layers_stack <- layers_stack %>% layer_activation_leaky_relu(alpha = alpha)
    } else {
      layers_stack <- layers_stack %>% layer_activation(activation = activation)
    }
  }
  return(layers_stack)
}

# Dense layers stack
dense_stack <- function(x, dense_units = c(256, 128), activation = "relu", alpha = 0.3) {
  layers_stack <- x
  for (i in 1:length(dense_units)) {
    layers_stack <- layers_stack %>%
      layer_dense(dense_units[i]) %>%
      layer_batch_normalization()

    if (activation == "lelu" | activation == "leaky_relu") {
      layers_stack <- layers_stack %>% layer_activation_leaky_relu(alpha = alpha)
    } else {
      layers_stack <- layers_stack %>% layer_activation(activation = activation)
    }
  }
  return(layers_stack)
}

# To simplify its use:
param_bernoulli <- function(x, output_shape, name = "Pr") {
  parameter1 <- layer_dense(x, units = output_shape, activation = "sigmoid")
  parameter2 <- layer_dense(x, units = output_shape)
  parameter3 <- layer_dense(x, units = output_shape)
  outputs <- layer_concatenate(list(parameter1, parameter2, parameter3), name = name)
  return(outputs)
}

param_bernoulli_sp <- function(x, output_shape, name = "Pr") {
  parameter1 <- layer_dense(x, units = output_shape, use_bias = FALSE, activation = "sigmoid")
  parameter2 <- layer_dense(x, units = output_shape, use_bias = FALSE) %>% tf$keras$activations$softplus()
  parameter3 <- layer_dense(x, units = output_shape, use_bias = FALSE) %>% tf$keras$activations$softplus()
  outputs <- layer_concatenate(list(parameter1, parameter2, parameter3), name = name)
  return(outputs)
}

param_gamma <- function(x, output_shape, name = "WS") {
  parameter1 <- layer_dense(x, units = output_shape)
  parameter2 <- layer_dense(x, units = output_shape)
  outputs <- layer_concatenate(list(parameter1, parameter2), name = name)
  return(outputs)
}

param_gamma_sp <- function(x, output_shape, name = "WS") {
  parameter1 <- layer_dense(x, units = output_shape, use_bias = FALSE) %>% tf$keras$activations$softplus()
  parameter2 <- layer_dense(x, units = output_shape, use_bias = FALSE) %>% tf$keras$activations$softplus()
  outputs <- layer_concatenate(list(parameter1, parameter2), name = name)
  return(outputs)
}

param_gengamma <- function(x, output_shape, name = "WS") {
  parameter1 <- layer_dense(x, units = output_shape)
  parameter2 <- layer_dense(x, units = output_shape)
  parameter3 <- layer_dense(x, units = output_shape)
  outputs <- layer_concatenate(list(parameter1, parameter2, parameter3), name = name)
  return(outputs)
}

param_genbernoulli <- function(x, output_shape, name = "Pr") {
  parameter1 <- layer_dense(x, units = output_shape, use_bias = FALSE, activation = "sigmoid")
  parameter2 <- layer_dense(x, units = output_shape, use_bias = FALSE) %>% tf$keras$activations$softplus()
  parameter3 <- layer_dense(x, units = output_shape, use_bias = FALSE) %>% tf$keras$activations$softplus()
  parameter4 <- layer_dense(x, units = output_shape, use_bias = FALSE) %>% tf$keras$activations$softplus()
  outputs <- layer_concatenate(list(parameter1, parameter2, parameter3, parameter4), name = name)
  return(outputs)
}

param_genbernoulli <- function(x, output_shape, name = "Pr") {
  parameter1 <- layer_dense(x, units = output_shape, activation = "sigmoid")
  parameter2 <- layer_dense(x, units = output_shape)
  parameter3 <- layer_dense(x, units = output_shape)
  parameter4 <- layer_dense(x, units = output_shape)
  outputs <- layer_concatenate(list(parameter1, parameter2, parameter3, parameter4), name = name)
  return(outputs)
}

param_weibull <- function(x, output_shape, name = "WS") {
  parameter1 <- layer_dense(x, units = output_shape)
  parameter2 <- layer_dense(x, units = output_shape)
  outputs <- layer_concatenate(list(parameter1, parameter2), name = name)
  return(outputs)
}

param_gauss <- function(x, output_shape, name = "T") {
  parameter1 <- layer_dense(x, units = output_shape)
  parameter2 <- layer_dense(x, units = output_shape)
  outputs <- layer_concatenate(list(parameter1, parameter2), name = name)
  return(outputs)
}

param_assign <- function(loss_names, var, branch, output_shape){
  if(any(loss_names[[var]] == c("gaussianLoss", "lognormalLoss"))) {
    branch <- param_gauss(branch, output = output_shape, name = var)
  } else if(loss_names[[var]] == "gammaLoss"){
    branch <- param_gamma(branch, output = output_shape, name = var) 
  } else if(any(loss_names[[var]] == c("gammaLoss_sp", "gammaLoss_sp_log"))){
    branch <- param_gamma_sp(branch, output = output_shape, name = var) 
  } else if(loss_names[[var]] == "gengammaLoss"){
    branch <- param_gengamma(branch, output = output_shape, name = var) 
  } else if(any(loss_names[[var]] == c("bernouilliGammaLoss", "bernouilliWeibullLoss", "bernouilliLognormalLoss"))) {
    branch <- param_bernoulli(branch, output_shape = output_shape, name = var) 
  } else if (loss_names[[var]] == "bernouilliGenGammaLoss"){
    branch <- param_genbernoulli(branch, output_shape = output_shape, name = var)           
  } else if (any(loss_names[[var]] == c("bernouilliGammaLoss_sp", "bernouilliGammaLoss_sp_log"))){
    branch <- param_bernoulli_sp(branch, output_shape = output_shape, name = var)       
  } else if(loss_names[[var]] == "weibullLoss") {
    branch <- param_weibull(branch, output = output_shape, name = var) 
  } else {
    branch <- layer_dense(branch, units = output_shape, name = var)
  }
  return(branch)
}

# For U-Net
step_down <- function(x, kernel_size = 3, activation = "relu", out_curr_channel_dim, return_downsampled = TRUE, BaNorm = TRUE, DropOut = TRUE, spatialDOrate = 0.25, alpha = 0.3) {
  x_conv <- ConvBlock(x, filters = out_curr_channel_dim, kernel_size = kernel_size, activation = activation, BaNorm = BaNorm, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha)

  if (return_downsampled) {
    x_conv_and_down <- layer_max_pooling_2d(x_conv, pool_size = c(2, 2), strides = c(2, 2), padding = "same")
    return(list(x_conv = x_conv, x_conv_and_down = x_conv_and_down))
  } else {
    return(x_conv)
  }
}

# For U-Net
step_up <- function(x, down_to_conc, kernel_size = 3, activation = "relu", current_channel_dim, BaNorm = TRUE, DropOut = TRUE, spatialDOrate = 0.25, alpha = 0.3) {
  x <- layer_conv_2d_transpose(x, filters = current_channel_dim / 2, kernel_size = c(2, 2), strides = c(2, 2))
  x <- layer_concatenate(list(down_to_conc, x), axis = -1)
  x <- ConvBlock(x, filters = current_channel_dim / 2, kernel_size = kernel_size, activation = activation, BaNorm = BaNorm, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha)
  return(x)
}

# U-Net definition
model_unet <- function(n_layers, channel_seed, kernel_size = 3, activation = "relu", input_shape, output_shape, filters_last = 1, act_last = "sigmoid", BaNorm1 = TRUE, BaNorm2 = FALSE, DropOut = TRUE, spatialDOrate = 0.25, dense = FALSE, dense_units = c(256, 128), alpha1 = 0.3, alpha2 = 0.3, loss_names = loss_names) {
    
  channels <- channel_seed * (2^((1:n_layers) - 1))
  inputs <- layer_input(shape = input_shape)
  x <- inputs

  layers_path_down <- list()

  for (i_layer in 1:(n_layers - 1)) {
    helper <- step_down(x, kernel_size = kernel_size, activation = activation, out_curr_channel_dim = channels[i_layer], return_downsampled = TRUE, BaNorm = BaNorm1, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha1)
    x <- helper[["x_conv_and_down"]]
    layers_path_down[[i_layer]] <- helper[["x_conv"]]
  }

  x <- step_down(x, kernel_size = kernel_size, activation = activation, out_curr_channel_dim = channels[length(channels)], return_downsampled = FALSE, BaNorm = BaNorm1, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha1)

  for (i_layer in (n_layers):2) {
    x <- step_up(x, down_to_conc = layers_path_down[[i_layer - 1]], kernel_size = kernel_size, activation = activation, current_channel_dim = channels[i_layer], BaNorm = BaNorm1, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha1)
  }
  
  outputs <- NULL
  
  for (i_var in names(loss_names)){
    var_branch <- ConvUnit(x, filters = filters_last, kernel_size = 1, activation = act_last,
      BaNorm = BaNorm2, DropOut = FALSE, alpha = alpha2)  %>%
      layer_flatten() 
    var_branch <- param_assign(loss_names = loss_names, var = i_var, branch = var_branch, output_shape = output_shape)
    outputs <- c(outputs, var_branch)
  }

  model <- keras_model(inputs = inputs, outputs = outputs)

  return(model)
}

# U-Net++ definition

model_unetpp <- function(n_layers, channel_seed, kernel_size = 3, activation = "relu", input_shape, output_shape, filters_last = 1, act_last = "sigmoid", BaNorm1 = TRUE, BaNorm2 = FALSE, DropOut = TRUE, spatialDOrate = 0.25, dense = FALSE, dense_units = c(256, 128), alpha1 = 0.3, alpha2 = 0.3, loss_names = loss_names) {
  
  channels <- channel_seed * (2^((1:n_layers) - 1))
  inputs <- layer_input(shape = input_shape)
  x <- inputs

  convs <- list()
  ups <- list()

  conv_pools <- function(n = n_layers) {
    convx <- list()
    pool <- list()

    for (i in seq_len(n)) {
      if (i == 1) {
        convx <- append(convx, list(ConvBlock(x, filters = channels[i], kernel_size = kernel_size, activation = activation, BaNorm = BaNorm1, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha1)))
        pool <- append(pool, list(layer_max_pooling_2d(convx[[i]], pool_size = c(2, 2), strides = c(2, 2), padding = "same")))
      } else if (i == n) {
        convx <- append(convx, list(ConvBlock(pool[[i - 1]], filters = channels[i], kernel_size = kernel_size, activation = activation, BaNorm = BaNorm1, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha1)))
      } else {
        convx <- append(convx, list(ConvBlock(pool[[i - 1]], filters = channels[i], kernel_size = kernel_size, activation = activation, BaNorm = BaNorm1, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha1)))
        pool <- append(pool, list(layer_max_pooling_2d(convx[[i]], pool_size = c(2, 2), strides = c(2, 2), padding = "same")))
      }
    }
    return(list(convx, pool))
  }

  upx_y <- function(n = n_layers, y) {
    upxy <- list()
    for (i in seq_len(n - (y - 1))) {
      upxy <- append(upxy, list(layer_conv_2d_transpose(convs[[y - 1]][[i + 1]], filters = channels[i], kernel_size = c(2, 2), strides = c(2, 2))))
    }
    return(upxy)
  }

  convx_y <- function(n = n_layers, y) {
    convxy <- list()
    for (i in seq_len(n - (y - 1))) {
      conc <- list(ups[[y - 1]][[i]])

      for (j in seq_len(y - 1)) {
        conc <- append(conc, list(convs[[j]][[i]]))
      }

      convxy <- append(convxy, list(ConvBlock(do.call(layer_concatenate, list(conc, axis = -1)), filters = channels[i], kernel_size = kernel_size, activation = activation, BaNorm = BaNorm1, DropOut = DropOut, spatialDOrate = spatialDOrate, alpha = alpha1)))
    }
    return(convxy)
  }

  for (i in seq_len(n_layers)) {
    if (i == 1) {
      helper <- conv_pools(n_layers)
      convs[[i]] <- helper[[1]]
      pools <- helper[[2]]
    } else {
      ups <- append(ups, list(upx_y(n_layers, i)))
      convs <- append(convs, list(convx_y(n_layers, i)))
    }
  }
  
  x <- convs[[n_layers]][[1]]

  outputs <- NULL

  for (i_var in names(loss_names)){
    var_branch <- ConvUnit(x, filters = filters_last, kernel_size = 1, activation = act_last,
      BaNorm = BaNorm2, DropOut = FALSE, alpha = alpha2)  %>%
      layer_flatten() 
    var_branch <- param_assign(loss_names = loss_names, var = i_var, branch = var_branch, output_shape = output_shape)
    outputs <- c(outputs, var_branch)
  }

  model <- keras_model(inputs = inputs, outputs = outputs)
  return(model)
}
