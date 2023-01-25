# Written by Dánnell Quesada-Chacón

# Function that creates the strings of all the combinations possible among the given hyperparameters
models_strings_fun <- function(models = c("", "pp"), layers = c(3, 4, 5), F_first = c(16, 32, 64, 128), F_last = c(1, 3), do_u = c(TRUE), do_rate = c(0.25), dense_after_u = c(FALSE), dense_units = list(c(256, 128), c(128)), activ_main = c("lelu"), activ_last = c("lelu"), alpha_main = c(0.3), alpha_last = c(0.3), BN_main = c(TRUE), BN_last = c(FALSE, TRUE), loss_names = loss_names) {
  unique(unlist(sapply(F_first, FUN = function(z) {
    sapply(F_last, FUN = function(y) {
      sapply(do_u, FUN = function(x) {
        sapply(do_rate, FUN = function(w) {
          sapply(dense_after_u, FUN = function(v) {
            sapply(dense_units, FUN = function(u) {
              sapply(models, FUN = function(t) {
                sapply(layers, FUN = function(o) {
                  sapply(activ_last, FUN = function(s) {
                    sapply(activ_main, FUN = function(r) {
                      sapply(BN_main, FUN = function(m) {
                        sapply(BN_last, FUN = function(n) {
                          sapply(alpha_main, FUN = function(q) {
                            sapply(alpha_last, FUN = function(p) {
                              if (r == "lelu" | r == "leaky_relu") {
                                activ_s1 <- paste0(substr(r, 1, 2), "_", q)
                                alpha_s1 <- paste0(", alpha1 = ", q)
                              } else {
                                activ_s1 <- substr(r, 1, 2)
                                alpha_s1 <- ""
                              }
                              if (s == "lelu" | s == "leaky_relu") {
                                activ_s2 <- paste0(substr(s, 1, 2), "_", p)
                                alpha_s2 <- paste0(", alpha2 = ", p)
                              } else {
                                activ_s2 <- substr(s, 1, 2)
                                alpha_s2 <- ""
                              }
                              if (x) {
                                if (v) {
                                  list(
                                    paste(paste0("U", t), o, z, y, w, activ_s1, activ_s2, paste(u, collapse = "_"), m, n, sep = "-"),
                                    paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", spatialDOrate = ", w, ", dense = ", v, ", dense_units = ", list(u), alpha_s1, alpha_s2, ", loss_names = ", list(loss_names), ")")
                                  )
                                } else {
                                  list(
                                    paste(paste0("U", t), o, z, y, w, activ_s1, activ_s2, m, n, sep = "-"),
                                    paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", spatialDOrate = ", w, ", dense = ", v, alpha_s1, alpha_s2, 
                                    ", loss_names = ", list(loss_names), ")")
                                  )
                                }
                              } else if (v) {
                                list(
                                  paste(paste0("U", t), o, z, y, activ_s1, activ_s2, paste(u, collapse = "_"), m, n, sep = "-"),
                                  paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", dense = ", v, ", dense_units = ", list(u), alpha_s1, alpha_s2,
                                  ", loss_names = ", list(loss_names), ")")
                                )
                              } else {
                                list(
                                  paste(paste0("U", t), o, z, y, activ_s1, activ_s2, m, n, sep = "-"),
                                  paste0("model <- model_unet", t, "(", o, ", ", z, ", activation = '", r, "', input_shape = input_shape, output_shape = output_shape, filters_last = ", y, ", act_last = '", s, "', BaNorm1 = ", m, ", BaNorm2 = ", n, ", DropOut = ", x, ", dense = ", v, alpha_s1, alpha_s2, ", loss_names = ", list(loss_names), ")")
                                )
                              }
                            }, simplify = T)
                          }, simplify = T)
                        }, simplify = T)
                      }, simplify = T)
                    }, simplify = T)
                  }, simplify = T)
                }, simplify = T)
              }, simplify = T)
            }, simplify = T)
          }, simplify = T)
        }, simplify = T)
      }, simplify = T)
    }, simplify = T)
  }, simplify = T), recursive = F))
}

# Function which receives the strings of the architecture and returns the models
architectures <- function(architecture, input_shape, output_shape, models_strings) {
  if (architecture == "CNN1") {
    inputs <- layer_input(shape = input_shape)

    l <- conv2d_stack(inputs, filters = c(50, 25, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture == "CNN10") {
    inputs <- layer_input(shape = input_shape)

    l <- conv2d_stack(inputs, filters = c(50, 25, 10), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture == "CNN64-1") {
    inputs <- layer_input(shape = input_shape)
    x <- inputs

    l <- conv2d_stack(inputs, filters = c(64, 32, 16, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture == "CNN32-1") {
    inputs <- layer_input(shape = input_shape)
    x <- inputs

    l <- conv2d_stack(inputs, filters = c(32, 16, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture == "CNN64_3-1") {
    inputs <- layer_input(shape = input_shape)
    x <- inputs

    l <- conv2d_stack(inputs, filters = c(64, 32, 1), activation = "relu", kernel_size = 3) %>%
      layer_flatten()

    outputs <- param_bernoulli(l, output_shape = output_shape)
    model <- keras_model(inputs = inputs, outputs = outputs)
  }

  if (architecture %in% models_strings) {
    eval(parse(text = models_strings[(match(architecture, models_strings) + 1)]))
  }
  return(model)
}

# Modification of downscaleTrain.keras to return training history and allow multiple variables per model

downscaleTrain.keras.mod4 <- function(obj, model, compile.args = list(object = model), fit.args = list(object = model), clear.session = FALSE, vars2model = names(yT)) {
  compile.args[["object"]] <- model
  if (is.null(compile.args[["optimizer"]])) {
    compile.args[["optimizer"]] <- optimizer_adam()
  }
  if (is.null(compile.args[["loss"]])) {
    compile.args[["loss"]] <- "mse"
  }
  do.call(compile, args = compile.args)
  fit.args[["object"]] <- model
  fit.args[["x"]] <- obj$x.global
  fit.args[["y"]] <- sapply(vars2model, simplify =  FALSE, FUN = function(z){
    return(obj$y[[z]]$Data)
  })
  
  history <- do.call(fit, args = fit.args)
  if (isTRUE(clear.session)) {
    k_clear_session()
  } else {
    model
  }
  return(history)
}

predict_out <- function(newdata_val, newdata_train, model, yT_bin, loss = "bernouilliGammaLoss", C4R.template) {
  out <- downscalePredict.keras.single(newdata_val, model, C4R.template = C4R.template, loss = loss)
  aux2 <- downscalePredict.keras.single(newdata_train, model, C4R.template = C4R.template, loss = loss) %>% subsetGrid(var = "p")
  aux1 <- subsetGrid(out, var = "p")
  bin <- binaryGrid(aux1, ref.obs = yT_bin, ref.pred = aux2)
  bin$Variable$varName <- "bin"
  if(loss== "bernouilliGammaLoss"){
    out <- makeMultiGrid(
      subsetGrid(out, var = "p"),
      subsetGrid(out, var = "log_alpha"),
      subsetGrid(out, var = "log_beta"),
      bin) %>% redim(member = FALSE)
  } else if(any(loss== c("bernouilliGammaLoss_sp", "bernouilliGammaLoss_sp_log"))){
    out <- makeMultiGrid(
      subsetGrid(out, var = "p"),
      subsetGrid(out, var = "alpha"),
      subsetGrid(out, var = "beta"),
      bin) %>% redim(member = FALSE)
  } else if(loss== "bernouilliLognormalLoss"){
    out <- makeMultiGrid(
      subsetGrid(out, var = "p"),
      subsetGrid(out, var = "mean"),
      subsetGrid(out, var = "log_var"),
      bin) %>% redim(member = FALSE)
  } else if(loss== "bernouilliGenGammaLoss"){
    out <- makeMultiGrid(
      subsetGrid(out, var = "p"),
      subsetGrid(out, var = "log_shape"),
      subsetGrid(out, var = "log_scale1"),
      subsetGrid(out, var = "log_scale2"),
      bin) %>% redim(member = FALSE)
  }
  return(out)
}

predict_out_all <- function(newdata_val, newdata_train, model, yT_bin, loss, C4R.template) {
  out_all <- downscalePredict.keras.new(newdata_val, model, C4R.template = C4R.template, loss = loss)
  aux2 <- downscalePredict.keras.new(newdata_train, model, C4R.template = C4R.template, loss = loss)[["Pr"]] %>% subsetGrid(var = "p")
  aux1 <- subsetGrid(out_all[["Pr"]], var = "p")
  bin <- binaryGrid(aux1, ref.obs = yT_bin, ref.pred = aux2)
  bin$Variable$varName <- "bin"
  if(loss[["Pr"]] == "bernouilliGammaLoss"){
    out <- makeMultiGrid(
      subsetGrid(out_all[["Pr"]], var = "p"),
      subsetGrid(out_all[["Pr"]], var = "log_alpha"),
      subsetGrid(out_all[["Pr"]], var = "log_beta"),
      bin) %>% redim(member = FALSE)
  } else if(any(loss[["Pr"]] == c("bernouilliGammaLoss_sp", "bernouilliGammaLoss_sp_log"))){
    out <- makeMultiGrid(
      subsetGrid(out_all[["Pr"]], var = "p"),
      subsetGrid(out_all[["Pr"]], var = "alpha"),
      subsetGrid(out_all[["Pr"]], var = "beta"),
      bin) %>% redim(member = FALSE)
  } else if(loss[["Pr"]] == "bernouilliLognormalLoss"){
    out <- makeMultiGrid(
      subsetGrid(out_all[["Pr"]], var = "p"),
      subsetGrid(out_all[["Pr"]], var = "mean"),
      subsetGrid(out_all[["Pr"]], var = "log_var"),
      bin) %>% redim(member = FALSE)
  } else if(loss[["Pr"]] == "bernouilliGenGammaLoss"){
    out <- makeMultiGrid(
      subsetGrid(out_all[["Pr"]], var = "p"),
      subsetGrid(out_all[["Pr"]], var = "log_shape"),
      subsetGrid(out_all[["Pr"]], var = "log_scale1"),
      subsetGrid(out_all[["Pr"]], var = "log_scale2"),
      bin) %>% redim(member = FALSE)
  }
  out_all[["Pr"]] <- out
  return(out_all)
}

prepareData.keras.mod2 <- function (x, y, global.vars = NULL, combined.only = TRUE, spatial.predictors = NULL, 
    local.predictors = NULL, first.connection = c("dense", "conv"), 
    last.connection = c("dense", "conv"), channels = c("first", 
        "last"), time.frames = NULL) {
    x <- x %>% redim(drop = TRUE)
    if (any(getDim(x) == "member")) 
        stop("No members allowed for training keras model")
    x <- x %>% redim(var = TRUE, member = FALSE)
    if (first.connection == "dense") {
        if (any(!is.null(global.vars) || !is.null(spatial.predictors) || 
            !is.null(local.predictors))) {
            x <- do.call("prepareData", args = list(x = x, y = y, 
                global.vars = global.vars, spatial.predictors = spatial.predictors, 
                local.predictors = local.predictors))
            x$y <- NULL
            if (attr(x, "nature") == "mix") {
                x.global <- cbind(x$x.global, x$x.local[[1]]$member_1)
            }
            else if (attr(x, "nature") == "global") {
                x.global <- x$x.global
            }
            else if (attr(x, "nature") == "local") {
                x.global <- x$x.local[[1]]$member_1
            }
            attr(x.global, "data.structure") <- x
        }
        else {
            if (isRegular(x)) {
                x.global <- lapply(getVarNames(x), FUN = function(z) {
                  array3Dto2Dmat(subsetGrid(x, var = z)$Data)
                }) %>% abind::abind(along = 0)
            }
            else {
                x.global <- x$Data
            }
            x.global <- x.global %>% aperm(c(2, 3, 1))
            dim(x.global) <- c(dim(x.global)[1], prod(dim(x.global)[2:3]))
        }
        if (anyNA(x.global)) 
            stop("There are NaNs in object: x, please consider using function filterNA prior to prepareData.keras")
    }
    else if (first.connection == "conv") {
        if (!isRegular(x)) 
            stop("Object 'x' must be a regular grid")
        if (anyNA(x$Data)) 
            stop("NaNs were found in object: x, please consider using function filterNA prior to prepareData.keras")
        if (channels == "last") 
            x.global <- x$Data %>% aperm(c(2, 3, 4, 1))
        if (channels == "first") 
            x.global <- x$Data %>% aperm(c(2, 1, 3, 4))
    }
    if (!is.null(time.frames)) {
        xx.global <- array(dim = c(dim(x.global)[1] - time.frames + 
            1, time.frames, dim(x.global)[-1]))
        for (t in 1:dim(xx.global)[1]) {
            if (first.connection == "dense") 
                xx.global[t, , ] <- x.global[t:(t + time.frames - 
                  1), ]
            if (first.connection == "conv") 
                xx.global[t, , , , ] <- x.global[t:(t + time.frames - 
                  1), , , ]
        }
        x.global <- xx.global
        rm(xx.global)
    }
    
    # Add logic for several variables in a list of "y" instead of a multiGrid
    if("xyCoords" %in% names(y)){
      if (last.connection == "dense") {
          if (isRegular(y)) {
              y$Data <- array3Dto2Dmat(y$Data)
          }
          if (anyNA(y$Data)) 
              warning("removing gridpoints containing NaNs of object: y")
          ind.y <- (!apply(y$Data, MARGIN = 2, anyNA)) %>% which()
          y$Data <- y$Data[, ind.y, drop = FALSE]
      }
      else if (last.connection == "conv") {
          if (!isRegular(y)) 
              stop("Object 'y' must be a regular grid")
          if (anyNA(y$Data)) 
              stop("NaNs were found in object: y")
      }
      if (!is.null(time.frames)) {
          if (last.connection == "dense") 
              y$Data <- y$Data[time.frames:dim(y$Data)[1], , drop = FALSE]
          if (last.connection == "conv") 
              y$Data <- y$Data[time.frames:dim(y$Data)[1], , , 
                  drop = FALSE]
          y$Dates$start <- y$Dates$start[time.frames:dim(y$Data)[1]]
          y$Dates$end <- y$Dates$end[time.frames:dim(y$Data)[1]]
      }
      predictor.list <- list(y = y, x.global = x.global)
    } else { 
      # Here...
      y_full <- sapply(names(y), FUN = function(z){
        y_sub <- y[[z]]
        if (last.connection == "dense") {
            if (isRegular(y_sub)) {
                y_sub$Data <- array3Dto2Dmat(y_sub$Data)
            }
            if (anyNA(y_sub$Data)) 
                warning("removing gridpoints containing NaNs of object: y")
            ind.y <- (!apply(y_sub$Data, MARGIN = 2, anyNA)) %>% which()
            y_sub$Data <- y_sub$Data[, ind.y, drop = FALSE]
            assign("ind.y", ind.y, envir = parent.env(environment()))
        }
        else if (last.connection == "conv") {
            if (!isRegular(y_sub)) 
                stop("Object 'y' must be a regular grid")
            if (anyNA(y_sub$Data)) 
                stop("NaNs were found in object: y")
        }
        if (!is.null(time.frames)) {
            if (last.connection == "dense") 
                y_sub$Data <- y_sub$Data[time.frames:dim(y_sub$Data)[1], , drop = FALSE]
            if (last.connection == "conv") 
                y_sub$Data <- y_sub$Data[time.frames:dim(y_sub$Data)[1], , , 
                    drop = FALSE]
            y_sub$Dates$start <- y_sub$Dates$start[time.frames:dim(y_sub$Data)[1]]
            y_sub$Dates$end <- y_sub$Dates$end[time.frames:dim(y_sub$Data)[1]]
        }
        return(list(y_sub))
      }, simplify = TRUE)
      predictor.list <- list(y = y_full, x.global = x.global)        
    }

    if (last.connection == "dense") 
        attr(predictor.list, "indices_noNA_y") <- ind.y
    attr(predictor.list, "first.connection") <- first.connection
    attr(predictor.list, "last.connection") <- last.connection
    attr(predictor.list, "channels") <- channels
    attr(predictor.list, "time.frames") <- time.frames
    return(predictor.list)
}

downscalePredict.keras.single <- function (newdata, model, C4R.template, clear.session = FALSE, 
    loss = NULL) 
{
    if (is.list(model)) 
        model <- do.call("load_model_hdf5", model)
    x.global <- newdata$x.global
    n.mem <- length(x.global)
    pred <- lapply(1:n.mem, FUN = function(z) {
        x.global[[z]] %>% model$predict()
    })
    names(pred) <- paste("member", 1:n.mem, sep = "_")
    if (isTRUE(clear.session)) 
        k_clear_session()
    template <- C4R.template
    if (attr(newdata, "last.connection") == "dense") {
        ind <- attr(newdata, "indices_noNA_y")
        n.vars <- ncol(pred[[1]])/length(ind)
        if (isRegular(template)) {
            ncol.aux <- array3Dto2Dmat(template$Data) %>% ncol()
        }
        else {
            ncol.aux <- getShape(template, dimension = "loc")
        }
        pred <- lapply(1:n.mem, FUN = function(z) {
            aux <- matrix(nrow = nrow(pred[[z]]), ncol = ncol.aux)
            lapply(1:n.vars, FUN = function(zz) {
                aux[, ind] <- pred[[z]][, ((ncol(pred[[1]])/n.vars) * 
                  (zz - 1) + 1):(ncol(pred[[1]])/n.vars * zz)]
                if (isRegular(template)) 
                  aux <- mat2Dto3Darray(aux, x = template$xyCoords$x, 
                    y = template$xyCoords$y)
                aux
            })
        })
    }
    dimNames <- attr(template$Data, "dimensions")
    pred <- lapply(1:n.mem, FUN = function(z) {
        if (attr(newdata, "last.connection") == "dense") {
            lapply(1:n.vars, FUN = function(zz) {
                template$Data <- pred[[z]][[zz]]
                attr(template$Data, "dimensions") <- dimNames
                if (isRegular(template)) 
                  template <- redim(template, var = FALSE)
                if (!isRegular(template)) 
                  template <- redim(template, var = FALSE, loc = TRUE)
                return(template)
            }) %>% makeMultiGrid()
        }
        else {
            if (attr(newdata, "channels") == "first") 
                n.vars <- dim(pred$member_1)[2]
            if (attr(newdata, "channels") == "last") 
                n.vars <- dim(pred$member_1)[4]
            lapply(1:n.vars, FUN = function(zz) {
                if (attr(newdata, "channels") == "first") 
                  template$Data <- pred[[z]] %>% aperm(c(2, 1, 
                    3, 4))
                if (attr(newdata, "channels") == "last") 
                  template$Data <- pred[[z]] %>% aperm(c(4, 1, 
                    2, 3))
                template$Data <- template$Data[zz, , , , drop = FALSE]
                attr(template$Data, "dimensions") <- c("var", 
                  "time", "lat", "lon")
                return(template)
            }) %>% makeMultiGrid()
        }
    })
    pred <- do.call("bindGrid", pred) %>% redim(drop = TRUE)
    pred$Dates <- attr(newdata, "dates")
    n.vars <- getShape(redim(pred, var = TRUE), "var")
    if (n.vars > 1) {
      if (loss == "gaussianLoss") {
          pred$Variable$varName <- c("mean", "log_var")
      } else if (loss == "bernouilliGammaLoss") {
          pred$Variable$varName <- c("p", "log_alpha", "log_beta")
      } else if (loss == "bernouilliLognormalLoss") {
          pred$Variable$varName <- c("p", "mean", "log_var")
      } else if (loss == "lognormalLoss") {
          pred$Variable$varName <- c("mean", "log_var")
      } else if (any(loss== c("bernouilliGammaLoss_sp", "bernouilliGammaLoss_sp_log"))) {
          pred$Variable$varName <- c("p", "alpha", "beta")
      } else if (loss == "bernouilliGenGammaLoss") {
          pred$Variable$varName <- c("p", "log_shape", "log_scale1", "log_scale2")
      } else if (loss == "gammaLoss") {
          pred$Variable$varName <- c("log_alpha", "log_beta")
      } else if (any(loss == c("gammaLoss_sp", "gammaLoss_sp_log"))){
          pred$Variable$varName <- c("alpha", "beta")
      } else if (loss == "gengammaLoss") {
          pred$Variable$varName <- c("log_shape", "log_scale1", "log_scale2")
      } else {
          pred$Variable$varName <- paste0(pred$Variable$varName, 1:n.vars)
      }
    pred$Dates <- rep(list(pred$Dates), n.vars)
    }
    return(pred)
}

downscalePredict.keras.new <- function (newdata, model, C4R.template, clear.session = FALSE, 
    loss = NULL) 
{
    if (is.list(model)) 
        model <- do.call("load_model_hdf5", model)
    x.global <- newdata$x.global
    n.mem <- length(x.global)
    
    # Attempt to generalize to more variables without destroying too much the original code
    pred_all <- lapply(1:n.mem, FUN = function(z) {
        x.global[[z]] %>% model$predict()
    })
    names(pred_all) <- paste("member", 1:n.mem, sep = "_") 
    
    if (isTRUE(clear.session)) 
        k_clear_session()
    template <- C4R.template

    pred_flip <- lapply(1:length(pred_all[[1]]), FUN = function(j) {
      sapply(names(pred_all), FUN = function(k){
        pred_all[[k]][[j]]
      }, simplify = FALSE)
    })
    names(pred_flip) <- names(C4R.template)
      
    pred_all <- sapply(names(pred_flip), FUN = function(i){
      pred <- pred_flip[[i]]
      if (attr(newdata, "last.connection") == "dense") {
          ind <- attr(newdata, "indices_noNA_y")
          n.vars <- ncol(pred[[1]])/length(ind)
          if (isRegular(template[[i]])) {
              ncol.aux <- array3Dto2Dmat(template[[i]]$Data) %>% ncol()
                #length(template[[i]]$xyCoords$x) * length(template[[i]]$xyCoords$y) # modified from array3Dto2Dmat
          }
          else {
              ncol.aux <- getShape(template[[i]], dimension = "loc")
          }
          pred <- lapply(1:n.mem, FUN = function(z) {
              aux <- matrix(nrow = nrow(pred[[z]]), ncol = ncol.aux)
              lapply(1:n.vars, FUN = function(zz) {
                  aux[, ind] <- pred[[z]][, ((ncol(pred[[1]])/n.vars) * 
                    (zz - 1) + 1):(ncol(pred[[1]])/n.vars * zz)]
                  if (isRegular(template[[i]])) 
                    aux <- mat2Dto3Darray(aux, x = template[[i]]$xyCoords$x, 
                      y = template[[i]]$xyCoords$y)
                  aux
              })
          })
      }
      dimNames <- attr(template[[i]]$Data, "dimensions")
      pred <- lapply(1:n.mem, FUN = function(z) {
          if (attr(newdata, "last.connection") == "dense") {
              lapply(1:n.vars, FUN = function(zz) {
                  template[[i]]$Data <- pred[[z]][[zz]]
                  attr(template[[i]]$Data, "dimensions") <- dimNames
                  if (isRegular(template[[i]])) 
                    template[[i]] <- redim(template[[i]], var = FALSE)
                  if (!isRegular(template[[i]])) 
                    template[[i]] <- redim(template[[i]], var = FALSE, loc = TRUE)
                  return(template[[i]])
              }) %>% makeMultiGrid()
          }
          else {
              if (attr(newdata, "channels") == "first") 
                  n.vars <- dim(pred$member_1)[2]
              if (attr(newdata, "channels") == "last") 
                  n.vars <- dim(pred$member_1)[4]
              lapply(1:n.vars, FUN = function(zz) {
                  if (attr(newdata, "channels") == "first") 
                    template[[i]]$Data <- pred[[z]] %>% aperm(c(2, 1, 
                      3, 4))
                  if (attr(newdata, "channels") == "last") 
                    template[[i]]$Data <- pred[[z]] %>% aperm(c(4, 1, 
                      2, 3))
                  template[[i]]$Data <- template[[i]]$Data[zz, , , , drop = FALSE]
                  attr(template[[i]]$Data, "dimensions") <- c("var", 
                    "time", "lat", "lon")
                  return(template[[i]])
              }) %>% makeMultiGrid()
          }
      })
      pred <- do.call("bindGrid", pred) %>% redim(drop = TRUE)
      pred$Dates <- attr(newdata, "dates")
      n.vars <- getShape(redim(pred, var = TRUE), "var")
      if (n.vars > 1) {
        if (loss[[i]] == "gaussianLoss") {
            pred$Variable$varName <- c("mean", "log_var")
        } else if (loss[[i]] == "bernouilliGammaLoss") {
            pred$Variable$varName <- c("p", "log_alpha", "log_beta")
        } else if (loss[[i]] == "bernouilliLognormalLoss") {
          pred$Variable$varName <- c("p", "mean", "log_var")
        } else if (loss[[i]] == "lognormalLoss") {
            pred$Variable$varName <- c("mean", "log_var")
        } else if (loss[[i]] == "bernouilliGammaLoss_sp") {
            pred$Variable$varName <- c("p", "alpha", "beta")
        } else if (loss[[i]] == "bernouilliGenGammaLoss") {
            pred$Variable$varName <- c("p", "log_shape", "log_scale1", "log_scale2")
        } else if (loss[[i]] == "gammaLoss") {
            pred$Variable$varName <- c("log_alpha", "log_beta")
        } else if (loss[[i]] == "gammaLoss_sp") {
            pred$Variable$varName <- c("alpha", "beta")
        } else if (loss[[i]] == "gengammaLoss") {
            pred$Variable$varName <- c("log_shape", "log_scale1", "log_scale2")
        } else {
            pred$Variable$varName <- paste0(pred$Variable$varName, 1:n.vars)
        }
        pred$Dates <- rep(list(pred$Dates), n.vars)
      }
      return(pred)
  }, simplify = FALSE)
  return(pred_all)
}

bernouilliWeibullLoss <- function(last.connection = NULL) 
{
    if (last.connection == "dense") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("bernouilliweibull_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/3
            ocurrence = pred[, 1:D]
            shape_parameter = tf$exp(pred[, (D + 1):(D * 2)])
            scale_parameter = tf$exp(pred[, (D * 2 + 1):(D * 
                3)])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32) 
            tf$print(shape_parameter)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(shape_parameter + epsilon) + 
                (shape_parameter - 1) * tf$math$log(true + epsilon) - 
                shape_parameter * tf$math$log(scale_parameter + epsilon) - 
                tf$pow(((true)/(scale_parameter + epsilon)), shape_parameter))))
        })
    }
    else if (last.connection == "conv") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("bernouilliweibull_loss", function(true, pred) {
            K = backend()
            ocurrence = pred[, , , 1, drop = TRUE]
            shape_parameter = tf$exp(pred[, , , 2, drop = TRUE])
            scale_parameter = tf$exp(pred[, , , 3, drop = TRUE])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(shape_parameter + epsilon) + 
                (shape_parameter - 1) * tf$math$log(true + epsilon) - 
                shape_parameter * tf$math$log(scale_parameter + epsilon) - 
                tf$pow(((true)/(scale_parameter + epsilon)), shape_parameter))))
        })
    }
}


bernouilliWeibullLoss <- function(last.connection = NULL) 
{
    if (last.connection == "dense") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("bernouilliweibull_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/3
            ocurrence = pred[, 1:D]
            shape_parameter = tf$exp(pred[, (D + 1):(D * 2)])
            scale_parameter = tf$exp(pred[, (D * 2 + 1):(D * 
                3)])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32) 
            tf$print(shape_parameter)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(shape_parameter + epsilon) + 
                (shape_parameter - 1) * tf$math$log(true + epsilon) - 
                shape_parameter * tf$math$log(scale_parameter + epsilon) - 
                ((true)/(scale_parameter + epsilon))^shape_parameter)))
        })
    }
    else if (last.connection == "conv") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("bernouilliweibull_loss", function(true, pred) {
            K = backend()
            ocurrence = pred[, , , 1, drop = TRUE]
            shape_parameter = tf$exp(pred[, , , 2, drop = TRUE])
            scale_parameter = tf$exp(pred[, , , 3, drop = TRUE])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(shape_parameter + epsilon) + 
                (shape_parameter - 1) * tf$math$log(true + epsilon) - 
                shape_parameter * tf$math$log(scale_parameter + epsilon) - 
                tf$pow(((true)/(scale_parameter + epsilon)), shape_parameter))))
        })
    }
}


bernouilliLognormalLoss <- function(last.connection = NULL) 
{
    if (last.connection == "dense") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("bernouilliLogNormal_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/3
            ocurrence <- pred[, 1:D]
            mean <- pred[, (D + 1):(D * 2)]
            log_var <- pred[, (D * 2 + 1):(D * 
                3)]
            precision <- tf$exp(-log_var)
            bool_rain <- tf$cast(tf$greater(true, 0), tf$float32)
#             tf$print(mean)
            epsilon <- 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (-tf$math$log(true + epsilon) - 0.5 * log_var - log(sqrt(2*pi)) - 0.5 * precision * (tf$math$log(true + epsilon) - mean)^2)))
        })
    }
    else if (last.connection == "conv") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("bernouilliLogNormal_loss", function(true, pred) {
            K <- backend()
            ocurrence <- pred[, , , 1, drop = TRUE]
            mean <- pred[, , , 2, drop = TRUE]
            log_var <- pred[, , , 3, drop = TRUE]
            bool_rain <- tf$cast(tf$greater(true, 0), tf$float32)
            epsilon <- 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (-tf$math$log(true + epsilon) - 0.5 * log_var - log(sqrt(2*pi)) - 0.5 * precision * (tf$math$log(true + epsilon) - mean)^2)))
        })
    }
}


lognormalLoss <- function(last.connection = NULL) 
{
    if (last.connection == "dense") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("lognormal_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/2
            mean <- pred[, 1:D]
            log_var <- pred[, (D + 1):(D * 2)]
            precision <- tf$exp(-log_var)
            epsilon <- 1e-06
            return(-tf$reduce_mean(-tf$math$log(true + epsilon) - 0.5 * log_var - log(sqrt(2*pi)) - 0.5 * precision * (tf$math$log(true + epsilon) - mean)^2))
        })
    }
    else if (last.connection == "conv") {
#         custom_metric("custom_loss", function(true, pred) {
        custom_metric("lognormal_loss", function(true, pred) {
            K <- backend()
            mean <- pred[, , , 1, drop = TRUE]
            log_var <- pred[, , , 2, drop = TRUE]
            epsilon <- 1e-06
            return(-tf$reduce_mean(-tf$math$log(true + epsilon) - 0.5 * log_var - log(sqrt(2*pi)) - 0.5 * precision * (tf$math$log(true + epsilon) - mean)^2))
        })
    }
}


weibullLoss <- function(last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("weibull_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/2
            shape_parameter = tf$exp(pred[, 1:D])
            scale_parameter = tf$exp(pred[, (D + 1):(D * 2)])
            epsilon = 1e-06
            return(-tf$reduce_mean(tf$math$log(shape_parameter + epsilon) + 
                (shape_parameter - 1) * tf$math$log(true + epsilon) - 
                shape_parameter * tf$math$log(scale_parameter + epsilon) - 
                tf$pow(((true + epsilon)/(scale_parameter + epsilon)), shape_parameter)))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("weibull_loss", function(true, pred) {
            K = backend()
            shape_parameter = tf$exp(pred[, , , 1, drop = TRUE])
            scale_parameter = tf$exp(pred[, , , 2, drop = TRUE])
            epsilon = 1e-06
            return(-tf$reduce_mean(tf$math$log(shape_parameter + epsilon) + 
                (shape_parameter - 1) * tf$math$log(true + epsilon) - 
                shape_parameter * tf$math$log(scale_parameter + epsilon) - 
                tf$pow(((true + epsilon)/(scale_parameter + epsilon)), shape_parameter)))
        })
    }
}


gammaLoss <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("gamma_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/2
            shape_parameter = tf$exp(pred[, 1:D])
            scale_parameter = tf$exp(pred[, (D + 1):(D * 2)])
            epsilon = 1e-06
            return(-tf$reduce_mean((shape_parameter - 1) * tf$math$log(true +
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon)))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("gamma_loss", function(true, pred) {
            K = backend()
            shape_parameter = tf$exp(pred[, , , 1, drop = TRUE])
            scale_parameter = tf$exp(pred[, , , 2, drop = TRUE])
            epsilon = 1e-06
            return(-tf$reduce_mean((shape_parameter - 1) * tf$math$log(true +
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon)))
        })
    }
}

gammaLoss_sp <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("gamma_sp_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/2
            shape_parameter = pred[, 1:D]
            scale_parameter = pred[, (D + 1):(D * 2)]
            epsilon = 1e-06
            return(-tf$reduce_mean((shape_parameter - 1) * tf$math$log(true +
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon)))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("gamma_sp_loss", function(true, pred) {
            K = backend()
            shape_parameter = pred[, , , 1, drop = TRUE]
            scale_parameter = pred[, , , 2, drop = TRUE]
            epsilon = 1e-06
            return(-tf$reduce_mean((shape_parameter - 1) * tf$math$log(true +
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon)))
        })
    }
}

gammaLoss_sp_log <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("gamma_sp_log_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/2
            shape_parameter = pred[, 1:D]
            scale_parameter = pred[, (D + 1):(D * 2)]
            epsilon = 1e-06
            return(-tf$reduce_mean((shape_parameter - 1) * tf$math$log(true +
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon)))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("gamma_sp_log_loss", function(true, pred) {
            K = backend()
            shape_parameter = pred[, , , 1, drop = TRUE]
            scale_parameter = pred[, , , 2, drop = TRUE]
            epsilon = 1e-06
            return(-tf$reduce_mean((shape_parameter - 1) * tf$math$log(true +
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon)))
        })
    }
}



rmse <- custom_metric("rmse", function(true, pred) {
    K <- backend()
    return(K$sqrt(K$mean(K$square(pred-true))))
    })

gaussianLoss <- function (last.connection = NULL)
{
    if (last.connection == "dense") {
        custom_metric("gaussian_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/2
            mean <- pred[, 1:D]
            log_var <- pred[, (D + 1):(D * 2)]
            precision <- tf$exp(-log_var)
            return(tf$reduce_mean(0.5 * precision * (true - mean)^2 +
                0.5 * log_var + log(sqrt(2*pi))))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("gaussian_loss", function(true, pred) {
            K <- backend()
            mean <- pred[, , , 1, drop = TRUE]
            log_var <- pred[, , , 2, drop = TRUE]
            precision <- tf$exp(-log_var)
            return(tf$reduce_mean(0.5 * precision * (true - mean)^2 +
                0.5 * log_var + log(sqrt(2*pi))))
        })
    }
}


computeWeibull <- function (shape, scale, simulate = FALSE, bias = NULL, name = "WS")
{
    loc <- FALSE
    if (!isRegular(shape)) 
        loc <- TRUE
    shape %<>% redim(shape, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    scale %<>% redim(scale, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    n.mem <- getShape(shape, "member")
    out <- lapply(1:n.mem, FUN = function(z) {
        shape %<>% subsetGrid(members = z)
        scale %<>% subsetGrid(members = z)
        amo <- shape
        dimNames <- attr(shape$Data, "dimensions")
        if (isTRUE(simulate)) { # Fix for stochastic
            ntime <- getShape(shape, "time")
            if (isRegular(shape)) {
                alpha_mat <- array3Dto2Dmat(shape$Data)
                beta_mat <- array3Dto2Dmat(scale$Data)
            }
            else {
                alpha_mat <- shape$Data %>% as.matrix()
                beta_mat <- scale$Data %>% as.matrix()
            }
            aux <- matrix(nrow = ntime, ncol = ncol(alpha_mat))
            for (zz in 1:ncol(alpha_mat)) {
                aux[, zz] <- rgamma(n = ntime, shape = alpha_mat[, 
                  zz], scale = beta_mat[, zz])
            }
            if (isRegular(shape)) {
                amo$Data <- mat2Dto3Darray(aux, x = amo$xyCoords$x, 
                  y = amo$xyCoords$y)
            }
            else {
                amo$Data <- aux
                attr(amo$Data, "dimensions") <- c("time", "loc")
            }
        }
        else {
            amo$Data <- exp(scale$Data) * gamma(1 + 1/exp(shape$Data))
        }
        if (!is.null(bias)) 
            amo <- amo %>% gridArithmetics(bias, operator = "+")
        return(amo)
    }) %>% bindGrid(dimension = "member") %>% redim(drop = TRUE)
    out$Variable$varName <- name
    return(out)
}

computeGamma <- function (log_alpha, log_beta, simulate = FALSE, bias = NULL, name = "Pr") 
{
    loc <- FALSE
    if (!isRegular(log_alpha)) 
        loc <- TRUE
    log_alpha %<>% redim(log_alpha, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    log_beta %<>% redim(log_beta, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    n.mem <- getShape(log_alpha, "member")
    out <- lapply(1:n.mem, FUN = function(z) {
        log_alpha %<>% subsetGrid(members = z)
        log_beta %<>% subsetGrid(members = z)
        amo <- log_alpha
        dimNames <- attr(log_alpha$Data, "dimensions")
        if (isTRUE(simulate)) {
            ntime <- getShape(log_alpha, "time")
            if (isRegular(log_alpha)) {
                alpha_mat <- array3Dto2Dmat(exp(log_alpha$Data))
                beta_mat <- array3Dto2Dmat(exp(log_beta$Data))
            }
            else {
                alpha_mat <- exp(log_alpha$Data) %>% as.matrix()
                beta_mat <- exp(log_beta$Data) %>% as.matrix()
            }
            aux <- matrix(nrow = ntime, ncol = ncol(alpha_mat))
            for (zz in 1:ncol(alpha_mat)) {
                aux[, zz] <- rgamma(n = ntime, shape = alpha_mat[, 
                  zz], scale = beta_mat[, zz])
            }
            if (isRegular(log_alpha)) {
                amo$Data <- mat2Dto3Darray(aux, x = amo$xyCoords$x, 
                  y = amo$xyCoords$y)
            }
            else {
                amo$Data <- aux
                attr(amo$Data, "dimensions") <- c("time", "loc")
            }
        }
        else {
            amo$Data <- exp(log_alpha$Data) * exp(log_beta$Data)
        }
        if (!is.null(bias)) 
            amo <- amo %>% gridArithmetics(bias, operator = "+")
        return(amo)
    }) %>% bindGrid(dimension = "member") %>% redim(drop = TRUE)
    out$Variable$varName <- name
    return(out)
}

computeGamma_sp <- function (alpha, beta, simulate = FALSE, bias = NULL, name = "Pr") 
{
    loc <- FALSE
    if (!isRegular(alpha)) 
        loc <- TRUE
    alpha %<>% redim(alpha, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    beta %<>% redim(beta, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    n.mem <- getShape(alpha, "member")
    out <- lapply(1:n.mem, FUN = function(z) {
        alpha %<>% subsetGrid(members = z)
        beta %<>% subsetGrid(members = z)
        amo <- alpha
        dimNames <- attr(alpha$Data, "dimensions")
        if (isTRUE(simulate)) {
            ntime <- getShape(alpha, "time")
            if (isRegular(alpha)) {
                alpha_mat <- array3Dto2Dmat(alpha$Data)
                beta_mat <- array3Dto2Dmat(beta$Data)
            }
            else {
                alpha_mat <- alpha$Data %>% as.matrix()
                beta_mat <- beta$Data %>% as.matrix()
            }
            aux <- matrix(nrow = ntime, ncol = ncol(alpha_mat))
            for (zz in 1:ncol(alpha_mat)) {
                aux[, zz] <- rgamma(n = ntime, shape = alpha_mat[, 
                  zz], scale = beta_mat[, zz])
            }
            if (isRegular(alpha)) {
                amo$Data <- mat2Dto3Darray(aux, x = amo$xyCoords$x, 
                  y = amo$xyCoords$y)
            }
            else {
                amo$Data <- aux
                attr(amo$Data, "dimensions") <- c("time", "loc")
            }
        }
        else {
            amo$Data <- alpha$Data * beta$Data
        }
        if (!is.null(bias)) 
            amo <- amo %>% gridArithmetics(bias, operator = "+")
        return(amo)
    }) %>% bindGrid(dimension = "member") %>% redim(drop = TRUE)
    out$Variable$varName <- name
    return(out)
}

computeGenGamma <- function (log_shape, log_scale1, log_scale2, simulate = FALSE, bias = NULL, name = "Pr") 
{
    loc <- FALSE
    if (!isRegular(log_shape)) 
        loc <- TRUE
    log_shape %<>% redim(log_shape, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    log_scale1 %<>% redim(log_scale1, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    log_scale2 %<>% redim(log_scale2, drop = TRUE, loc = loc) %>% 
        redim(member = TRUE, loc = loc)
    n.mem <- getShape(log_shape, "member")
    out <- lapply(1:n.mem, FUN = function(z) {
        log_shape %<>% subsetGrid(members = z)
        log_scale1 %<>% subsetGrid(members = z)
        log_scale2 %<>% subsetGrid(members = z)
        amo <- log_shape
        dimNames <- attr(log_shape$Data, "dimensions")
        if (isTRUE(simulate)) {
            ntime <- getShape(log_shape, "time")
            if (isRegular(log_shape)) {
                shape_mat <- array3Dto2Dmat(exp(log_shape$Data))
                scale1_mat <- array3Dto2Dmat(exp(log_scale1$Data))
                scale2_mat <- array3Dto2Dmat(exp(log_scale2$Data))
            }
            else {
                shape_mat <- exp(log_shape$Data) %>% as.matrix()
                scale1_mat <- exp(log_scale1$Data) %>% as.matrix()
                scale2_mat <- exp(log_scale2$Data) %>% as.matrix()
            }
            aux <- matrix(nrow = ntime, ncol = ncol(shape_mat))
            for (zz in 1:ncol(shape_mat)) {
                aux[, zz] <- rgengamma.orig(n = ntime, shape = shape_mat[,zz], 
                                            scale = scale1_mat[, zz]/scale2_mat[, zz],
                                            k = scale2_mat) 
            }
            if (isRegular(log_shape)) {
                amo$Data <- mat2Dto3Darray(aux, x = amo$xyCoords$x, 
                  y = amo$xyCoords$y)
            }
            else {
                amo$Data <- aux
                attr(amo$Data, "dimensions") <- c("time", "loc")
            }
        }
        else {
            amo$Data <- exp(log_shape$Data)*gamma((exp(log_scale1$Data)+1)/exp(log_scale2$Data))/gamma(exp(log_scale1$Data)/exp(log_scale2$Data))
        }
        if (!is.null(bias)) 
            amo <- amo %>% gridArithmetics(bias, operator = "+")
        return(amo)
    }) %>% bindGrid(dimension = "member") %>% redim(drop = TRUE)
    out$Variable$varName <- name
    return(out)
}

computeGauss <- function (mean = NULL, log_var = NULL, name = "tas") 
{
    mean %<>% redim(mean, drop = TRUE) %>% redim(member = TRUE)
    log_var %<>% redim(log_var, drop = TRUE) %>% redim(member = TRUE)
    n.mem <- getShape(log_var, "member")
    out <- lapply(1:n.mem, FUN = function(z) {
        mean %<>% subsetGrid(members = z)
        log_var %<>% subsetGrid(members = z)
        t <- mean
        dimNames <- attr(mean$Data, "dimensions")
        ntime <- getShape(mean, "time")
        if (isRegular(mean)) {
            mean <- array3Dto2Dmat(mean$Data)
            sd <- array3Dto2Dmat(exp(log_var$Data) %>% sqrt())
        }
        else {
            mean <- mean$Data
            sd <- exp(log_var$Data) %>% sqrt()
        }
        aux <- matrix(nrow = ntime, ncol = ncol(mean))
        for (zz in 1:ncol(mean)) {
            aux[, zz] <- rnorm(n = ntime, mean = mean[, zz], 
                sd = sd[, zz])
        }
        if (isRegular(t)) {
            t$Data <- mat2Dto3Darray(aux, x = t$xyCoords$x, y = t$xyCoords$y)
        }
        else {
            t$Data <- aux
        }
        return(t)
    }) %>% bindGrid() %>% redim(drop = TRUE)
    out$Variable$varName <- name
    return(out)
}


computeLognormal <- function (mean = NULL, log_var = NULL, simulate = FALSE, bias = NULL, name = "Pr") 
{
    loc <- FALSE
    if (!isRegular(mean)) 
        loc <- TRUE
    mean %<>% redim(mean, drop = TRUE, loc = loc) %>% redim(member = TRUE, loc = loc)
    log_var %<>% redim(log_var, drop = TRUE, loc = loc) %>% redim(member = TRUE, loc = loc)
    n.mem <- getShape(log_var, "member")
    out <- lapply(1:n.mem, FUN = function(z) {
        mean %<>% subsetGrid(members = z)
        log_var %<>% subsetGrid(members = z)
        t <- mean
        dimNames <- attr(mean$Data, "dimensions")
        if (isTRUE(simulate)) {
          ntime <- getShape(mean, "time")
          if (isRegular(mean)) {
              mean <- array3Dto2Dmat(mean$Data)
              sd <- array3Dto2Dmat(exp(log_var$Data) %>% sqrt())
          } else {
              mean <- mean$Data %>% as.matrix()
              sd <- exp(log_var$Data) %>% sqrt() %>% as.matrix()
          }
          aux <- matrix(nrow = ntime, ncol = ncol(mean))
          for (zz in 1:ncol(mean)) {
              aux[, zz] <- rlnorm(n = ntime, meanlog = mean[, zz], 
                  sdlog = sd[, zz])
          }
          if (isRegular(t)) {
              t$Data <- mat2Dto3Darray(aux, x = t$xyCoords$x, y = t$xyCoords$y)
          } else {
            t$Data <- aux
            attr(t$Data, "dimensions") <- c("time", "loc")
          }
        } else {
          t$Data <- exp(mean$Data + (exp(log_var$Data))/2)
#           attr(t$Data, "dimensions") <- dimNames
        }
        if (!is.null(bias)) 
            t <- t %>% gridArithmetics(bias, operator = "+")
        return(t)
    }) %>% bindGrid(dimension = "member") %>% redim(drop = TRUE)
    out$Variable$varName <- name
    return(out)
}

smooth_obs <- function(z) {
  return(runif(1, min = (z - 0.1/2 + 1e-5), max = (z + 0.1/2 - 1e-5)))
}

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

# https://stackoverflow.com/questions/28927750/what-is-the-method-to-save-objects-in-a-compressed-form-using-multiple-threads-c

saveRDS.xz <- function(object,file,threads=parallel::detectCores()) {
  con <- pipe(paste0("xz -T",threads," > ",file),"wb")
  saveRDS(object, file = con, compress=FALSE)
  close(con)
}

readRDS.xz <- function(file,threads=parallel::detectCores()) {
  con <- pipe(paste0("xz -d -k -c -T",threads," ",file))
  object <- readRDS(file = con)
  close(con)
  return(object)
}

saveRDS.gz <- function(object,file,threads=parallel::detectCores()) {
  con <- pipe(paste0("pigz -p",threads," > ",file),"wb")
  saveRDS(object, file = con)
  close(con)
}

readRDS.gz <- function(file,threads=parallel::detectCores()) {
  con <- pipe(paste0("pigz -d -c -p",threads," ",file))
  object <- readRDS(file = con)
  close(con)
  return(object)
}

readRDS.p <- function(file,threads=parallel::detectCores()) {
  #Hypothetically we could use initial bytes to determine file format, but here we use the Linux command file because the readBin implementation was not immediately obvious
  fileDetails <- system2("file",args=file,stdout=TRUE)
  selector <- sapply(c("gz","XZ"),function (x) {grepl(x,fileDetails)})
  format <- names(selector)[selector]
  if (format == "gz") {
    object <- readRDS.gz(file, threads=threads)
  } else if (format == "XZ") {
    object <- readRDS.xz(file, threads=threads)
  } else {
    object <- readRDS(file)
  }
  return(object)
}

bernouilliGenGammaLoss <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("bernouilligengamma_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/4
            ocurrence = pred[, 1:D]
            shape = tf$exp(pred[, (D + 1):(D * 2)])
            scale1 = tf$exp(pred[, (D * 2 + 1):(D * 3)])
            scale2 = tf$exp(pred[, (D * 3 + 1):(D * 4)])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - ocurrence + epsilon) + bool_rain * 
                                   (tf$math$log(ocurrence + epsilon) + 
                                    shape * tf$math$log(scale2 + epsilon) +
                                    (shape - 1) * tf$math$log(true + epsilon) - 
                                    shape * tf$math$log(scale1 + epsilon) - 
                                    tf$math$lgamma((shape/scale2) + epsilon) - 
                                    tf$math$pow(true/(scale1 + epsilon), scale2))))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("bernouilligengamma_loss", function(true, pred) {
            K = backend()
            ocurrence = pred[, , , 1, drop = TRUE]
            shape = tf$exp(pred[, , , 2, drop = TRUE])
            scale1 = tf$exp(pred[, , , 3, drop = TRUE])
            scale2 = tf$exp(pred[, , , 4, drop = TRUE])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - ocurrence + epsilon) + bool_rain * 
                                   (tf$math$log(ocurrence + epsilon) + 
                                    shape * tf$math$log(scale2 + epsilon) +
                                    (shape - 1) * tf$math$log(true + epsilon) - 
                                    shape * tf$math$log(scale1 + epsilon) - 
                                    tf$math$lgamma((shape/scale2) + epsilon) - 
                                    tf$math$pow(true/(scale1 + epsilon), scale2))))
        })
    }
}


bernouilliGenGammaLoss_sp <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("bernouilligengamma_sp_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/4
            ocurrence = pred[, 1:D]
            shape = pred[, (D + 1):(D * 2)]
            scale1 = pred[, (D * 2 + 1):(D * 3)]
            scale2 = pred[, (D * 3 + 1):(D * 4)]
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-09
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - ocurrence + epsilon) + bool_rain * 
                                   (tf$math$log(ocurrence + epsilon) + 
                                    shape * tf$math$log(scale2 + epsilon) +
                                    (shape - 1) * tf$math$log(true + epsilon) - 
                                    shape * tf$math$log(scale1 + epsilon) - 
                                    tf$math$lgamma((shape/scale2) + epsilon) - 
                                    tf$math$pow((true/(scale1 + epsilon)), scale2))))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("bernouilligengamma_sp_loss", function(true, pred) {
            K = backend()
            ocurrence = pred[, , , 1, drop = TRUE]
            shape = (pred[, , , 2, drop = TRUE])
            scale1 = (pred[, , , 3, drop = TRUE])
            scale2 = (pred[, , , 4, drop = TRUE])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-09
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - ocurrence + epsilon) + bool_rain * 
                                   (tf$math$log(ocurrence + epsilon) + 
                                    shape * tf$math$log(scale2 + epsilon) +
                                    (shape - 1) * tf$math$log(true + epsilon) - 
                                    shape * tf$math$log(scale1 + epsilon) - 
                                    tf$math$lgamma((shape/scale2) + epsilon) - 
                                    tf$math$pow((true/(scale1 + epsilon)), scale2))))
        })
    }
}

gengammaLoss <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("gengamma_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/3
            shape = tf$exp(pred[, 1:D])
            scale1 = tf$exp(pred[, (D + 1):(D * 2)])
            scale2 = tf$exp(pred[, (D * 2 + 1):(D * 3)])
            epsilon = 1e-06
            return(-tf$reduce_mean(shape * tf$math$log(scale2 + epsilon) +
                                    (shape - 1) * tf$math$log(true + epsilon) - 
                                    shape * tf$math$log(scale1 + epsilon) - 
                                    tf$math$lgamma((shape/scale2) + epsilon) - 
                                    tf$math$pow(true/(scale1 + epsilon), scale2)))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("gengamma_loss", function(true, pred) {
            K = backend()
            shape = tf$exp(pred[, , , 1, drop = TRUE])
            scale1 = tf$exp(pred[, , , 2, drop = TRUE])
            scale2 = tf$exp(pred[, , , 3, drop = TRUE])
            epsilon = 1e-06
            return(-tf$reduce_mean(shape * tf$math$log(scale2 + epsilon) +
                                    (shape - 1) * tf$math$log(true + epsilon) - 
                                    shape * tf$math$log(scale1 + epsilon) - 
                                    tf$math$lgamma((shape/scale2) + epsilon) - 
                                    tf$math$pow(true/(scale1 + epsilon), scale2)))
        })
    }
}

rgengamma.orig <- function (n, shape, scale = 1, k) 
{
    r <- rbase("gengamma.orig", n = n, shape = shape, scale = scale, 
        k = k)
    for (i in seq_along(r)) assign(names(r)[i], r[[i]])
    w <- log(rgamma(n, shape = k))
    y <- w/shape + log(scale)
    ret[ind] <- exp(y)
    ret
}

bernouilliGammaLoss_sp <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("bernouilliGamma_sp_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/3
            ocurrence = pred[, 1:D]
            shape_parameter = (pred[, (D + 1):(D * 2)])
            scale_parameter = (pred[, (D * 2 + 1):(D * 3)])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(ocurrence + 
                epsilon) + (shape_parameter - 1) * tf$math$log(true + 
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon))))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("bernouilliGamma_sp_loss", function(true, pred) {
            K = backend()
            ocurrence = pred[, , , 1, drop = TRUE]
            shape_parameter = (pred[, , , 2, drop = TRUE])
            scale_parameter = (pred[, , , 3, drop = TRUE])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(ocurrence + 
                epsilon) + (shape_parameter - 1) * tf$math$log(true + 
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon))))
        })
    }
}

bernouilliGammaLoss_sp_log <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("bernouilliGamma_sp_log_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/3
            ocurrence = pred[, 1:D]
            shape_parameter = (pred[, (D + 1):(D * 2)])
            scale_parameter = (pred[, (D * 2 + 1):(D * 3)])
            true <- tf$math$log(true) %>% tf$nn$relu()
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(ocurrence + 
                epsilon) + (shape_parameter - 1) * tf$math$log(true + 
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon))))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("bernouilliGamma_sp_log_loss", function(true, pred) {
            K = backend()
            ocurrence = pred[, , , 1, drop = TRUE]
            shape_parameter = (pred[, , , 2, drop = TRUE])
            scale_parameter = (pred[, , , 3, drop = TRUE])
            true <- tf$math$log(true) %>% tf$nn$relu()
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(ocurrence + 
                epsilon) + (shape_parameter - 1) * tf$math$log(true + 
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon))))
        })
    }
}


# Orig
bernouilliGammaLoss <- function (last.connection = NULL) 
{
    if (last.connection == "dense") {
        custom_metric("custom_loss", function(true, pred) {
            K <- backend()
            D <- K$int_shape(pred)[[2]]/3
            ocurrence = pred[, 1:D]
            shape_parameter = tf$exp(pred[, (D + 1):(D * 2)])
            scale_parameter = tf$exp(pred[, (D * 2 + 1):(D * 
                3)])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(ocurrence + 
                epsilon) + (shape_parameter - 1) * tf$math$log(true + 
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon))))
        })
    }
    else if (last.connection == "conv") {
        custom_metric("custom_loss", function(true, pred) {
            K = backend()
            ocurrence = pred[, , , 1, drop = TRUE]
            shape_parameter = tf$exp(pred[, , , 2, drop = TRUE])
            scale_parameter = tf$exp(pred[, , , 3, drop = TRUE])
            bool_rain = tf$cast(tf$greater(true, 0), tf$float32)
            epsilon = 1e-06
            return(-tf$reduce_mean((1 - bool_rain) * tf$math$log(1 - 
                ocurrence + epsilon) + bool_rain * (tf$math$log(ocurrence + 
                epsilon) + (shape_parameter - 1) * tf$math$log(true + 
                epsilon) - shape_parameter * tf$math$log(scale_parameter + 
                epsilon) - tf$math$lgamma(shape_parameter + epsilon) - 
                true/(scale_parameter + epsilon))))
        })
    }
}

custom_losses <- list("custom_loss" = bernouilliGammaLoss(last.connection = "dense"),
                      "bernouilliGamma_sp_loss" = bernouilliGammaLoss_sp(last.connection = "dense"),
                      "bernouilliGamma_sp_log_loss" = bernouilliGammaLoss_sp_log(last.connection = "dense"),
                      "bernouilliLogNormal_loss" = bernouilliLognormalLoss(last.connection = "dense"),
                      "lognormal_loss" = lognormalLoss(last.connection = "dense"),
                      "gamma_loss" = gammaLoss(last.connection = "dense"),
                      "gamma_sp_loss" = gammaLoss_sp(last.connection = "dense"),
                      "gamma_sp_log_loss" = gammaLoss_sp_log(last.connection = "dense"),
                      "gaussian_loss" = gaussianLoss(last.connection = "dense"), 
                      "rmse" = rmse)


pred_pr <- function(loss_pr,seed, pred, pr_threshold, simulate) {
  print(paste0("Predicting Pr ", loss_pr, "..."))
  if(loss_pr == "bernouilliGammaLoss"){
    reset_seeds(as.integer(seed))
    pred_amo <- computeGamma(
      log_alpha = subsetGrid(pred[["Pr"]], var = "log_alpha"),
      log_beta = subsetGrid(pred[["Pr"]], var = "log_beta"),
      bias = pr_threshold,
      simulate = simulate, name = "Pr"
    )
  } else if(loss_pr == "bernouilliLognormalLoss"){
    reset_seeds(as.integer(seed))
    pred_amo <- computeLognormal(
      mean = subsetGrid(pred[["Pr"]], var = "mean"),
      log_var = subsetGrid(pred[["Pr"]], var = "log_var"),
      bias = pr_threshold,
      simulate = simulate, name = "Pr"
    )           
  } else if(loss_pr == "bernouilliGammaLoss_sp"){
    reset_seeds(as.integer(seed))
    pred_amo <- computeGamma_sp(
      alpha = subsetGrid(pred[["Pr"]], var = "alpha"),
      beta = subsetGrid(pred[["Pr"]], var = "beta"),
      bias = pr_threshold,
      simulate = simulate, name = "Pr"
    )           
  } else if(loss_pr == "bernouilliGammaLoss_sp_log"){
    reset_seeds(as.integer(seed))
    pred_amo <- computeGamma_sp(
      alpha = subsetGrid(pred[["Pr"]], var = "alpha"),
      beta = subsetGrid(pred[["Pr"]], var = "beta"),
      bias = 0,
      simulate = simulate, name = "Pr"
    )
    pred_amo$Data <- exp(pred_amo$Data)                
  } else if(loss_pr == "bernouilliGenGammaLoss"){
    reset_seeds(as.integer(seed))
    pred_amo <- computeGenGamma(
      log_shape = subsetGrid(pred[["Pr"]], var = "log_shape"),
      log_scale1 = subsetGrid(pred[["Pr"]], var = "log_scale1"),
      log_scale2 = subsetGrid(pred[["Pr"]], var = "log_scale2"),
      bias = pr_threshold,
      simulate = simulate, name = "Pr"
    )
  }

  pred_ocu <- subsetGrid(pred[["Pr"]], var = "p") %>% redim(drop = TRUE)
  pred_bin <- subsetGrid(pred[["Pr"]], var = "bin")
  pred_serie <- gridArithmetics(pred_amo, pred_bin)

  pr_all <- list(bin = pred_bin, ocu = pred_ocu, amo = pred_amo, serie = pred_serie) 
  return(pr_all)
}

pred_gamma <- function(var, seed, pred, simulate){
  reset_seeds(as.integer(seed))
  pred_g <- computeGamma(
    log_alpha = subsetGrid(pred[[var]], var = "log_alpha"),
    log_beta = subsetGrid(pred[[var]], var = "log_beta"),
    simulate = simulate,
    name = var)
  return(pred_g)
}

pred_gammasp <- function(var, seed, pred, simulate){
  reset_seeds(as.integer(seed))
  pred_g <- computeGamma_sp(
    alpha = subsetGrid(pred[[var]], var = "alpha"),
    beta = subsetGrid(pred[[var]], var = "beta"),
    simulate = simulate,
    name = var)
  return(pred_g)
}

pred_lognorm <- function(var, seed, pred, simulate){
  reset_seeds(as.integer(seed))
  pred_ln <- computeLognormal(
    mean = subsetGrid(pred[[var]], var = "mean"),
    log_var = subsetGrid(pred[[var]], var = "log_var"),
    simulate = simulate,
    name = var) 
  return(pred_ln)
}

predict_final <- function(loss_names, y_train, index_vars, newdata_val, newdata_train, model, simulate, pr_threshold, seed){
  if(length(index_vars)>1){
    if(any(names(y_train)[index_vars]=="Pr")) {
      if(any(loss_names$Pr == c("bernouilliGammaLoss","bernouilliGenGammaLoss","bernouilliGammaLoss_sp", "bernouilliGammaLoss_sp_log", "bernouilliLognormalLoss"))){
        print("Predicting all...")
        pred <- predict_out_all(newdata_val, newdata_train,
          model = model,
          pr_train_bin, C4R.template = y_train[index_vars],
          loss = loss_names[index_vars]
        )
        pred[["Pr"]] <- pred_pr(loss_pr = loss_names$Pr, seed = seed, pred = pred, pr_threshold = pr_threshold, simulate = simulate)
      } else {
        print("Predicting all...")
        pred <- downscalePredict.keras.new(newdata_val, model, C4R.template = y_train[index_vars], loss = loss_names[index_vars])
      }
    } else {
      print("Predicting all...")
      pred <- downscalePredict.keras.new(newdata_val, model, C4R.template = y_train[index_vars], loss = loss_names[index_vars])
    }
  } else {
    print("Predicting one at the time...")
    if(names(y_train)[index_vars]=="Pr") {
      if(any(loss_names$Pr == c("bernouilliGammaLoss","bernouilliGenGammaLoss","bernouilliGammaLoss_sp", "bernouilliGammaLoss_sp_log", "bernouilliLognormalLoss"))){
        print("Predicting Bernouilli pr...")
        pred <- predict_out(newdata_val, newdata_train,
          model = model,
          pr_train_bin, C4R.template = y_train[[index_vars]],
          loss = loss_names[index_vars]
        )
        pred <- list(pred)
        names(pred) <- names(y_train)[index_vars]
        pred[["Pr"]] <- pred_pr(loss_pr = loss_names$Pr, seed = seed, pred = pred, pr_threshold = pr_threshold, simulate = simulate)
      } else {
        pred <- downscalePredict.keras.single(newdata_val, model, C4R.template = y_train[[index_vars]], loss = loss_names[index_vars])
        pred <- list(pred)
        names(pred) <- names(y_train)[index_vars]
      }
    } else {
      pred <- downscalePredict.keras.single(newdata_val, model, C4R.template = y_train[[index_vars]], loss = loss_names[index_vars])
      pred <- list(pred)
      names(pred) <- names(y_train)[index_vars]      
    }
  }
  
  if(any("gammaLoss" == loss_names[index_vars])){
    for (i in names(which("gammaLoss" == loss_names[index_vars]))){
      print(paste0("Predicting ", i, " gammaLoss..."))
      pred[[i]] <- pred_gamma(var = i, seed = seed, pred = pred, simulate = simulate)
    }
  }
  
  if(any("lognormalLoss" == loss_names[index_vars])){
    for (i in names(which("lognormalLoss" == loss_names[index_vars]))){
      print(paste0("Predicting ", i, " lognormalLoss..."))
      pred[[i]] <- pred_lognorm(var = i, seed = seed, pred = pred, simulate = simulate)
    }
  }
  
  if(any("gammaLoss_sp" == loss_names[index_vars])){
    for (i in names(which("gammaLoss_sp" == loss_names[index_vars]))){
      print(paste0("Predicting ", i, " gammaLoss_sp..."))
      pred[[i]] <- pred_gammasp(var = i, seed = seed, pred = pred, simulate = simulate)
    }
  }
  
  if (any("gaussianLoss" == loss_names[index_vars])){
    for (i in names(which(loss_names[index_vars]=="gaussianLoss"))){
      print(paste0("Predicting ", i ," Gaussian..."))

      if(isTRUE(simulate)){
        reset_seeds(as.integer(seed))
        pred_aux <- computeTemperature(
          mean = subsetGrid(pred[[i]], var = "mean"),
          log_var = subsetGrid(pred[[i]], var = "log_var")
        )
      } else {
        pred_aux <- subsetGrid(pred[[i]], var = "mean")
      }
      pred_aux$Variable$varName <- i
      pred[[i]] <- pred_aux
      rm(pred_aux)
    }
  }
  return(pred)
}
