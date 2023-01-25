options(java.parameters = "-Xmx544000m")
# Written by Dánnell Quesada-Chacón

# PYTHONHASHSEED=7777 TF_DETERMINISTIC_OPS=1 R

# n_repeat <- 1
# scenario <- 8
# part <- 6

seed <- Sys.getenv("PYTHONHASHSEED")
n_repeat <- as.integer(commandArgs(trailingOnly = TRUE)[1])
scenario <- as.integer(commandArgs(trailingOnly = TRUE)[2])
part <- as.integer(commandArgs(trailingOnly = TRUE)[3])

pr_threshold <- 1
pr_condition <- "GE"

library(reticulate)
library(downscaleR.keras)

np <- import("numpy")
np <- import("numpy") # Sometimes needs to be done twice for it to load properly
rd <- import("random")

# Function to reset all seeds
reset_seeds <- function(seed = 42) {
  np$random$seed(seed)
  rd$seed(seed)
  tf$random$set_seed(seed)
  set.seed(seed)
}

reset_seeds(as.integer(seed))

library(loadeR)
library(transformeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(sp)
library(stringr)
library(doParallel)
# cores <- getOption("cl.cores", detectCores()) # to use all cores, memory issues may arise
cores <- 4 # Parallel computing for six cores, change accordingly

tensorflow::tf$config$experimental$list_physical_devices("GPU")
physical_devices <- tensorflow::tf$config$experimental$list_physical_devices("GPU")

# Allow memory growth for GPUs
sapply(1:length(physical_devices), function(x) {
  tf$config$experimental$set_memory_growth(physical_devices[[x]], TRUE)
  print(physical_devices[[x]])
})

try(k_constant(1), silent = TRUE) # Small test for the GPU
try(k_constant(1), silent = TRUE) # Sometimes needs to be done twice for it to load properly

path <- "./"

setwd(path)

source("./unet_def.R") # File with functions for U architectures
source("./aux_funs_train.R") # File with functions to ease traning process
dir.create(paste0("Data/all/", seed, "/", scenario), recursive = TRUE)
dir.create(paste0("models/all/", seed, "/", scenario), recursive = TRUE)

# To see full output of summary()
options("width" = 150)
options(warn = -1)

# Parameters and indices

u_model <- c("", "pp") # Type of u model, "" for U-Net, "pp" for U-Net++
u_layers <- c(3,4) # Depth of the u model
u_seeds <- c(64,128) # Number of filters of the first layer
u_Flast <- c(1) # Number of filters of the last ConvUnit
u_do <- c(TRUE) # Boolean for dropout within the u model
u_spdor <- c(0.25) # Fraction of the spatial dropout within the u model
u_dense <- c(FALSE) # Boolean for dense units after ConvUnit
u_dunits <- list(c(256, 128), c(128)) # Dense units, passed as a list, several layers supported
act_main <- c("lelu") # Activation function within u model and dense units
act_last <- c("lelu") # Activation function for the last ConvUnit
alpha1 <- c(0.3) # If leaky relu (lelu) used, which alpha. For u model and dense units
alpha2 <- c(0.3) # Same as above but for last ConvUnit
BN1 <- c(TRUE) # Batch normalization inside u model
BN2 <- c(TRUE) # Batch normalization for the last ConvUnit

loss_names_all <- list(
  Gauss = list(Rn = "gaussianLoss", Pr = "gaussianLoss", TM = "gaussianLoss", TN = "gaussianLoss", TX = "gaussianLoss", WS = "gaussianLoss", Pw = "gaussianLoss"),
  rmse = list(Rn = "rmse", Pr = "rmse", TM = "rmse", TN = "rmse", TX = "rmse", WS = "rmse", Pw = "rmse"),
  Dist = list(Rn = "gammaLoss", Pr = "bernouilliGammaLoss", TM = "gaussianLoss", TN = "gaussianLoss", TX = "gaussianLoss", WS = "gammaLoss", Pw = "gammaLoss")
)

loss_names <- loss_names_all[[scenario]]

loss_list_all <- list(
  Gauss = list(
    Rn = gaussianLoss(last.connection = "dense"),
    Pr = gaussianLoss(last.connection = "dense"),
    TM = gaussianLoss(last.connection = "dense"), 
    TN = gaussianLoss(last.connection = "dense"), 
    TX = gaussianLoss(last.connection = "dense"), 
    WS = gaussianLoss(last.connection = "dense"), 
    Pw = gaussianLoss(last.connection = "dense")),
  rmse = list(
    Rn = rmse,
    Pr = rmse,
    TM = rmse,
    TN = rmse, 
    TX = rmse, 
    WS = rmse, 
    Pw = rmse),
  Dist = list(
    Rn = gammaLoss(last.connection = "dense"),
    Pr = bernouilliGammaLoss(last.connection = "dense"),
    TM = gaussianLoss(last.connection = "dense"),
    TN = gaussianLoss(last.connection = "dense"), 
    TX = gaussianLoss(last.connection = "dense"), 
    WS = gammaLoss(last.connection = "dense"), 
    Pw = gammaLoss(last.connection = "dense"))
)

loss_list <- loss_list_all[[scenario]]

# Validation metrics
simulateName <- list(Rn = c(rep("deterministic", 8),  rep("stochastic",2)),
                      Pr = c(rep("deterministic", 11), rep("stochastic",1)),
                      TM = c(rep("deterministic", 8),  rep("stochastic",2)),
                      TN = c(rep("deterministic", 11), rep("stochastic",2)),
                      TX = c(rep("deterministic", 11), rep("stochastic",2)),
                      WS = c(rep("deterministic", 8),  rep("stochastic",1)),
                      Pw = c(rep("deterministic", 8),  rep("stochastic",2)))

measures <- list(Rn = c("ts.cmksdiff","ts.RMSE","ts.rp","ratio",rep("bias",6)),
                  Pr = c("ts.cmksdiff","ts.rocss","ts.RMSE","ts.rs","ratio",rep("bias", 7)),
                  TM = c("ts.cmksdiff","ts.RMSE","ts.rp","ratio",rep("bias",6)),
                  TN = c("ts.cmksdiff","ts.RMSE","ts.rp","ratio",rep("bias",9)),
                  TX = c("ts.cmksdiff","ts.RMSE","ts.rp","ratio",rep("bias",9)),
                  WS = c("ts.cmksdiff","ts.RMSE","ts.rs","ratio",rep("bias",5)),
                  Pw = c("ts.cmksdiff","ts.RMSE","ts.rp","ratio",rep("bias",6)))

index <- list(Rn = c(rep(NA,3),"sd","Mean","P02","P98","AC1","P02","P98"),
              Pr = c(rep(NA,4),"sd","Mean","P02","P98","SDII","WetAnnualMaxSpell","DryAnnualMaxSpell","P98"),
              TM = c(rep(NA,3),"sd","Mean","P02","P98","AC1","P02","P98"),
              TN = c(rep(NA,3),"sd","Mean","P02","P98","AC1","ColdAnnualMaxSpell","FB0","FA20","P02","P98"),
              TX = c(rep(NA,3),"sd","Mean","P02","P98","AC1","WarmAnnualMaxSpell","FB0","FA25","P02","P98"),
              WS = c(rep(NA,3),"sd","Mean","P02","P98","AC1","P98"),
              Pw = c(rep(NA,3),"sd","Mean","P02","P98","AC1","P02","P98"))

switch(part, "1" = {index_vars_all <- list(1:7)}, 
             "2" = {index_vars_all <- list(1,2)}, 
             "3" = {index_vars_all <- list(3,4)}, 
             "4" = {index_vars_all <- list(5,6)},
             "5" = {index_vars_all <- list(7)})

# The next 2 parameters should be changed according to the GPU memory
# batch_size <- 64
batch_size <- 512
lr <- 0.001/2
patience <- 125
validation_split <- 0.1

times_repeat <- seq_len(n_repeat) # To repeat n times

# Loading predS

x <- readRDS("./Data/all/x_32.rds")
x_train <- subsetGrid(x, var = x$Variable$varName[!x$Variable$varName %in% c("hcc","lcc","mcc")],  years = 1979:2005)

x_val <- subsetGrid(x, var = x$Variable$varName[!x$Variable$varName %in% c("hcc","lcc","mcc")], years = 2006:2015)

x_val <- scaleGrid(x_val, x_train, type = "standardize", spatial.frame = "gridbox") %>% redim(drop = TRUE)
x_train <- scaleGrid(x_train, type = "standardize", spatial.frame = "gridbox") %>% redim(drop = TRUE)
rm(x)
gc()

y <- readRDS("./Data/all/y_sud.rds")

pr_bin <- y$Pr %>% binaryGrid(threshold = pr_threshold, condition = pr_condition)

pr_train_bin <- subsetGrid(pr_bin, years = 1979:2005)

pr_val_bin <- subsetGrid(pr_bin, years = 2006:2015)

rm(pr_bin)
gc()

y_train <- sapply(names(y), FUN = function(var){
    subsetGrid(y[[var]], years = 1979:2005)
}, simplify = FALSE)

pr_train_bin_part <- y$Pr %>% subsetGrid(years = 1979:2005) %>%
  gridArithmetics(., pr_threshold, operator = "-") %>%
  binaryGrid(condition = pr_condition, threshold = 0, partial = TRUE)

y_train$Pr <- pr_train_bin_part

gc()

y_val <- sapply(names(y), FUN = function(var){
    subsetGrid(y[[var]], years = 2006:2015)
}, simplify = FALSE)

rm(y)
gc()

xy_train <- prepareData.keras.mod2(x_train, y_train,
  first.connection = "conv",
  last.connection = "dense",
  channels = "last"
)

newdata_train <- prepareNewData.keras(x_train, xy_train)
newdata_val <- prepareNewData.keras(x_val, xy_train)
gc()

sapply(index_vars_all, simplify=FALSE, FUN=function(index_vars){
  
  models_strings <- models_strings_fun(
    models = u_model, layers = u_layers, F_first = u_seeds,
    F_last = u_Flast, do_u = u_do, do_rate = u_spdor,
    dense_after_u = u_dense, dense_units = u_dunits,
    activ_main = act_main, activ_last = act_last,
    alpha_main = alpha1, alpha_last = alpha2,
    BN_main = BN1, BN_last = BN2, loss_names = loss_names[index_vars]
  )

  deepName <- c(models_strings[seq(1, length(models_strings), 2)])

  to_train <- (1:length(deepName))

  ifelse(length(index_vars)==length(loss_names), fileID <- "all", fileID <- paste0(names(loss_names)[index_vars], collapse="-"))
  ifelse(length(index_vars)==length(loss_names), fileLoss <- names(loss_names_all)[scenario], fileLoss <- paste0(loss_names[index_vars], collapse="-"))

  print(paste0(fileID, " ", fileLoss))
        
  sapply(times_repeat, simplify = FALSE, FUN = function(n) {
    if(file.exists(paste0("./Data/all/", seed, "/", scenario, "/hist_train_CNN_", 
                          fileID, "_", fileLoss, "_", n, ".rda"))) {
                            load(paste0("./Data/all/", seed, "/", scenario, "/hist_train_CNN_", fileID, "_", fileLoss, "_", n, ".rda"))
    } else {history_trains <- list()}
    
    simulateType <- c("deterministic","stochastic")
    simulateDeep <- c(FALSE,TRUE)

    print(paste0("Repetition #", n, " of ", n_repeat))
    
    sapply(to_train, simplify = FALSE, FUN = function(z) {
      
      if(!deepName[z] %in% names(history_trains)){
        print(paste0("Model ", z, " out of ", length(to_train), " = ", deepName[z]))
        gc()

        # For multi GPU, repeatability not fully tested
        if (length(physical_devices) > 1) {
          reset_seeds(as.integer(seed))
          strategy <- tf$distribute$MirroredStrategy()
          print(paste0("Number of GPUs on which the model will be distributed: ", strategy$num_replicas_in_sync))
          reset_seeds(as.integer(seed))
          with(strategy$scope(), {
            model <- architectures(
              architecture = deepName[z],
              input_shape = dim(xy_train$x.global)[-1],
              output_shape = dim(xy_train$y[[1]]$Data)[2],
              models_strings = models_strings
            )
          })
          summary(model)

          reset_seeds(as.integer(seed))
          history_train <- downscaleTrain.keras.mod4(
            obj = xy_train,
            model = model,
            clear.session = TRUE,
            vars2model = names(y_train)[index_vars],
            compile.args = list(
              "loss" = loss_list[index_vars],
              "optimizer" = optimizer_adam(lr = lr)
            ),
            fit.args = list(
              "batch_size" = batch_size,
              "epochs" = 5000,
              "validation_split" = validation_split,
              "verbose" = 1,
              "callbacks" = list(
                callback_early_stopping(patience = patience),
                callback_model_checkpoint(
                  filepath = paste0("./models/all/", seed, "/", scenario, "/", deepName[z], "_", fileID, "_", fileLoss, "_", n, ".h5"),
                  monitor = "val_loss", save_best_only = TRUE
                )
              )
            )
          )
        } else {
          # For one GPU
          reset_seeds(as.integer(seed))
          model <- architectures(
            architecture = deepName[z],
            input_shape = dim(xy_train$x.global)[-1],
            output_shape = dim(xy_train$y[[1]]$Data)[2],
            models_strings = models_strings
          )
          summary(model)

          reset_seeds(as.integer(seed))
          history_train <- downscaleTrain.keras.mod4(
            obj = xy_train,
            model = model,
            clear.session = TRUE,
            vars2model = names(y_train)[index_vars],
            compile.args = list(
              "loss" = loss_list[index_vars],
              "optimizer" = optimizer_adam(lr = lr)
            ),
            fit.args = list(
              "batch_size" = batch_size,
              "epochs" = 5000,
              "validation_split" = validation_split,
              "verbose" = 1,
              "callbacks" = list(
                callback_early_stopping(patience = patience),
                callback_model_checkpoint(
                  filepath = paste0("./models/all/", seed, "/", scenario, "/", deepName[z], "_", fileID, "_", fileLoss, "_", n, ".h5"),
                  monitor = "val_loss", save_best_only = TRUE
                )
              )
            )
          )
        }
        gc()

        hist_t <- list(history_train)
        names(hist_t) <- deepName[z]

        history_trains <<- append(history_trains, hist_t)

        # Save history after training each model
        save(history_trains, file = paste0("./Data/all/", seed, "/", scenario, "/hist_train_CNN_", fileID, "_", fileLoss, "_", n, ".rda"))
      } else {
        print(paste0("Model ", deepName[z], " already trained."))
        
        model <- do.call("load_model_hdf5", list(custom_objects = custom_losses,
                          filepath= paste0("./models/all/", seed, "/", scenario, "/", deepName[z], "_", fileID, "_", fileLoss, "_", n, ".h5")))
        
      }
         
      sapply(1:length(simulateDeep), simplify = FALSE, FUN = function(zz) {
        if(!file.exists(paste0("./Data/all/", seed, "/", scenario, "/predictions_", simulateType[zz], "_", deepName[z], "_", fileID, "_", fileLoss, "_", n, ".rda"))){
          print(simulateType[zz])
          gc()
          
          pred <- predict_final(loss_names = loss_names, y_train = y_train, index_vars = index_vars, newdata_val = newdata_val, newdata_train = newdata_train, model = model, simulate = simulateDeep[zz], pr_threshold = pr_threshold, seed = seed)
          
          save(pred,
            file = paste0("./Data/all/", seed, "/", scenario, "/predictions_", simulateType[zz], "_", deepName[z], "_", fileID, "_", fileLoss, "_", n, ".rda")
          )
          gc()
        } else {
          print(paste0("File: ", paste0("./Data/all/", seed, "/", scenario, "/predictions_", simulateType[zz], "_", deepName[z], "_", fileID, "_", fileLoss, "_", n, ".rda"), " exists, next..."))
        }
      })
    })
  })
   
  # Calculate metrics parallelly
  models <- c(deepName)

  cl <- makeCluster(mc <- cores, outfile = "")
  clusterExport(
    cl = cl, varlist = c("simulateName", "models", "measures", "index", "seed", "pr_val_bin", "y_val", 
      "times_repeat", "n_repeat", "cores", "fileID", "fileLoss", "index_vars", "scenario", "loss_names",
      "pr_condition", "pr_threshold"), 
    envir = environment()
  )
  cl.libs <- clusterEvalQ(cl = cl, expr = {
    library(reticulate)
    library(downscaleR.keras)
    library(loadeR)
    library(transformeR)
    library(climate4R.value)
    library(magrittr)
    library(gridExtra)
    library(stringr)
    library(sp)
    library(doParallel)
  })
  
  sapply(times_repeat, FUN = function(n) {
    if(!file.exists(paste0("./Data/all/", seed, "/", scenario, "/validation_CNN_", fileID, "_", fileLoss, "_", n, ".rda"))){
      print("Training done, now parallel calculation of validation metrics")
      print(paste0(index_vars, " ", fileID, " ", fileLoss))
      gc()

      validation.list <- sapply(names(measures)[index_vars], simplify = FALSE, FUN = function(var){
        aux <- sapply(1:length(measures[[var]]), simplify = FALSE, FUN = function(z) {
          gc()
          print(paste0(var, " - ", measures[[var]][z], " - ", index[[var]][z], " - ", simulateName[[var]][z]))
          
          parSapply(cl = cl, 1:length(models), FUN = function(zz) {
            args <- list()
            print(paste0("./Data/all/", seed, "/", scenario, "/predictions_", simulateName[[var]][z], "_", models[zz], "_", fileID, "_", fileLoss, "_", n, ".rda"))
            load(paste0("./Data/all/", seed, "/", scenario, "/predictions_", simulateName[[var]][z], "_", models[zz], "_", fileID, "_", fileLoss, "_", n, ".rda"))
            if (var == "Pr") {
              if (str_detect(loss_names[[var]], "bernouilli")) {
                if (simulateName[[var]][z] == "deterministic") {
                    if (measures[[var]][z] == "ts.rocss") {
                      args[["y"]] <- pr_val_bin
                      args[["x"]] <- pred[[var]]$ocu
                    } else if (measures[[var]][z] == "ts.RMSE") {
                      args[["y"]] <- y_val[[var]]
                      args[["x"]] <- pred[[var]]$amo
                      args[["condition"]] <- pr_condition
                      args[["threshold"]] <- pr_threshold
                      args[["which.wetdays"]] <- "Observation"
                    } else if (measures[[var]][z] == "ts.cmksdiff") {
                      args[["y"]] <- y_val[[var]]
                      args[["x"]] <- pred[[var]]$serie
                      args[["condition"]] <- pr_condition
                      args[["threshold"]] <- pr_threshold
                      args[["which.wetdays"]] <- "Observation"
                    } else {
                      args[["y"]] <- y_val[[var]]
                      args[["x"]] <- pred[[var]]$serie
                    }
                  } else {
                    args[["y"]] <- y_val[[var]]
                    args[["x"]] <- pred[[var]]$serie
                  }
              } else {
                  args[["y"]] <- y_val[[var]]
                  args[["x"]] <- pred[[var]]
                  if (measures[[var]][z] == "ts.cmksdiff"){
                    args[["condition"]] <- pr_condition
                    args[["threshold"]] <- pr_threshold
                    args[["which.wetdays"]] <- "Observation"
                  }                 }
            } else {
                args[["y"]] <- y_val[[var]]
                args[["x"]] <- pred[[var]]
            }
            args[["measure.code"]] <- measures[[var]][z]
            if (!is.na(index[[var]][z])) args[["index.code"]] <- index[[var]][z]
            do.call("valueMeasure", args)$Measure
          }, simplify = F) %>% makeMultiGrid()
        }) 
        names(aux) <- paste0(measures[[var]], index[[var]]) %>% gsub("NA", "",.)
        if(!identical(names(aux)[which(simulateName[[var]]=="stochastic")], character(0))){
          names(aux)[which(simulateName[[var]]=="stochastic")] <-  paste0(names(aux)[which(simulateName[[var]]=="stochastic")], "Sto")
        }
        return(aux)
      })

      validation.list$Models <-  list(models)
      save(validation.list, file = paste0("./Data/all/", seed, "/", scenario, "/validation_CNN_", fileID, "_", fileLoss, "_", n, ".rda"))  
    } else {
    print("Metrics already calculated.")
    }
  })
  stopCluster(cl = cl)
  unregister_dopar()
})
