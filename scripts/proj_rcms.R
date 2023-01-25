# Written by Dánnell Quesada-Chacón

iter <- as.character(commandArgs(trailingOnly = TRUE)[1])
part <- as.integer(commandArgs(trailingOnly = TRUE)[2])

library(reticulate)
library(downscaleR.keras)

np <- import("numpy")
np <- import("numpy") # Sometimes needs to be done twice for it to load properly
rd <- import("random")

# Function to reset all seeds
reset_seeds <- function(seed = 42L) {
  np$random$seed(seed)
  rd$seed(seed)
  tf$random$set_seed(seed)
  set.seed(seed)
}

reset_seeds()

library(loadeR)
library(transformeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(sp)
library(stringr)

library(loadeR.2nc) # version v0.1.1

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

options("width" = 150)

source("./aux_funs_train.R") # File with functions to ease traning process


# Load DL models to load for projections
df <- readRDS(paste0("./rds/proj_models_", iter, ".rds"))

x <- readRDS("./Data/all/x_32.rds")

xT <- subsetGrid(x, var = x$Variable$varName[!x$Variable$varName %in% c("hcc","lcc","mcc")],  years = 1979:2005)
xT <- scaleGrid(xT, type = "standardize", spatial.frame = "gridbox") %>% redim(drop = TRUE)

gc()

y <- readRDS("./Data/all/y_sud.rds")

pr_threshold <- 1
pr_condition <- "GE"

pr_bin <- y$Pr %>% binaryGrid(threshold = pr_threshold, condition = pr_condition)

prT_bin <- subsetGrid(pr_bin, years = 1979:2005)

rm(pr_bin)
gc()


prT_bin_part <- y$Pr %>% subsetGrid(years = 1979:2005) %>%
  gridArithmetics(., pr_threshold-0.001, operator = "-") %>%
  binaryGrid(condition = pr_condition, threshold = 0, partial = TRUE)

yT <- sapply(names(y), FUN = function(var){
    subsetGrid(y[[var]], years = 1979:2005)
}, simplify = FALSE)

yT$Pr <- prT_bin_part

gc()

xy.T <- prepareData.keras.mod2(xT, yT,
first.connection = "conv",
last.connection = "dense",
channels = "last"
)

xy.tT <- prepareNewData.keras(xT, xy.T)

rm(x,y,prT_bin_part,xT)
gc()
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


simulateName <- c("det","sto")
simulateName_long <- c("deterministic","stochastic")
simulateDeep <- c(FALSE,TRUE)

# dir.create(paste0("/data/rds/Proj")
dir.create(paste0("/data/rds/Proj_", iter))

custom_losses <- list("custom_loss" = bernouilliGammaLoss(last.connection = "dense"),
                      "gamma_loss" = gammaLoss(last.connection = "dense"),
                      "gaussian_loss" = gaussianLoss(last.connection = "dense"), 
                      "rmse" = rmse)
rows2proj <- part
gc()

sapply(rows2proj, FUN = function(i) {
  print(paste0("Variable to downscale: ",  df$var_model[i]))
  print(paste0("Trained model to use: ./models/all/", df$seed[i], "/", df$sce[i], "/", df$models[i], "_", df$var_model[i], "_", df$loss_model_name[i], "_", df$Run[i], ".h5"))
  model <- do.call("load_model_hdf5", list(custom_objects = custom_losses,
                            filepath= paste0("./models/all/", df$seed[i], "/", df$sce[i], "/", df$models[i], "_", df$var_model[i], "_", df$loss_model_name[i], "_", df$Run[i], ".h5")))

  if(df$var[i] == "Pr"){
    pred_ocu_train <- downscalePredict.keras(newdata = xy.tT,C4R.template = yT[[df$var[i]]], model = model, loss = df$loss_var_name[i]) %>% subsetGrid(var = "p")
  }
  
  gc()

  dirs_rcms <- system("find ./rds/HS/ -mindepth 3 -maxdepth 3 -type d", intern=TRUE)

  sapply(dirs_rcms, FUN = function(dir) {
    print(paste0("Directory where RCM is: ", dir))
    rcms <- list.files(dir, "*.rds*")
    dir.create(dir %>% str_replace("/HS/", paste0("/Proj_" , iter, "/")), recursive = TRUE)

    sapply(rcms, FUN = function(rcm) {
      print(paste0("Downscaling RCM: ", rcm))

      # Global attributes for all NetCDF files, add more details
      globalAttributeList <- list("Reference_dataset" = "ReKIS (Regionales Klimainformationssystem Sachsen, Sachsen-Anhalt, Thüringen), https://rekis.hydro.tu-dresden.de",
                                  "Sub-domain" = "Ore Mountains, Vogtland and Sächsische Schweiz, Saxony, Germany",
                                  "Predictor_dataset" = "ERA5",
                                  "Driving_RCM" = rcm %>% str_remove(".rds.xz"),
                                  "Author" = "Dánnell Quesada-Chacón")

      gc()

      # Logic to skip downscaling if file already exists
      if(df$loss_var_name[i] == "rmse" && file.exists(paste0(dir %>% str_replace("/HS/", paste0("/Proj_" , iter, "/")), "/", df$var[i], rcm %>% str_remove_all("EUR-25|_32") %>% str_replace(".rds.xz", "_rmse.nc")))) {return(print("File exists... Next!"))}

      if(df$loss_var_name[i] != "rmse" && all(file.exists(paste0(dir %>% str_replace("/HS/", paste0("/Proj_" , iter, "/")), "/", df$var[i], rcm %>% str_remove_all("EUR-25|_32")  %>% str_replace(".rds.xz", paste0("_", simulateName, ".nc")))))) {return(print("Files exist... Next!"))}

      xn <- readRDS(paste0(dir, "/", rcm)) %>% redim(drop = TRUE)
      print("Preparing data...")
      xyt <- prepareNewData.keras(xn, xy.T)

      rm(xn)
      gc()
      
      print("Predicting...")
      pred <- downscalePredict.keras(newdata = xyt, C4R.template = yT[[df$var[i]]], model = model, loss = df$loss_var_name[i])

      if(df$loss_var_name[i] == "rmse"){
        varAttributeList <- list("Downscaling_architecture" = df$models[i],
                          "Loss_function" = df$loss_var_name[i])

        print("Saving nc file...")
        grid2nc(pred, NetCDFOutFile = paste0(dir %>% str_replace("/HS/", paste0("/Proj_" , iter, "/")), "/", df$var[i], rcm %>% str_remove_all("EUR-25|_32") %>% str_replace(".rds.xz", "_rmse.nc")),
          globalAttributes = globalAttributeList, varAttributes = varAttributeList)
        rm(pred)
        gc()
      } else {
        sapply(1:length(simulateDeep), simplify = FALSE, FUN = function(zz) {
          print(simulateName[zz])

          varAttributeList <- list("Downscaling_architecture" = df$models[i],
                                    "Loss_function" = df$loss_var_name[i],
                                    "Simulation_type" = simulateName_long[zz])

          if(df$loss_var_name[i] == "bernouilliGammaLoss") {
            reset_seeds(as.integer(df$seed[i]))
            pred <- computeGamma(
              log_alpha = subsetGrid(pred, var = "log_alpha"),
              log_beta = subsetGrid(pred, var = "log_beta"),
              bias = pr_threshold,
              simulate = simulateDeep[zz], name = "Pr"
            ) %>% gridArithmetics(binaryGrid(subsetGrid(pred,var = "p"),ref.obs = prT_bin,ref.pred = pred_ocu_train))

          }

          if(df$loss_var_name[i] == "gammaLoss") {
            reset_seeds(as.integer(df$seed[i]))
            pred <- computeGamma(
              log_alpha = subsetGrid(pred, var = "log_alpha|WS1|GS1|Pw1|Rn1"),
              log_beta = subsetGrid(pred, var = "log_beta|WS2|GS2|Pw2|Rn2"),
              simulate = simulateDeep[zz],
              name = df$var[i])
          }
          
          if(df$loss_var_name[i] == "lognormalLoss"){
              reset_seeds(as.integer(df$seed[i]))
              pred <- computeLognormal(
                mean = subsetGrid(pred, var = "mean|WS1"),
                log_var = subsetGrid(pred, var = "log_var|WS2"),
                simulate = simulateDeep[zz], 
                name = df$var[i])
          }

          if(df$loss_var_name[i]=="gaussianLoss"){
            # for stochastic
            if(isTRUE(simulateDeep[zz])){
              reset_seeds(as.integer(df$seed[i]))
              pred_aux <- computeTemperature(
                mean = subsetGrid(pred, var = "mean"),
                log_var = subsetGrid(pred, var = "log_var")
              )
            } else {
              pred_aux <- subsetGrid(pred, var = "mean")
            }
            pred_aux$Variable$varName <- df$var[i]
            pred <- pred_aux
          }
          print("Saving nc file...")
          grid2nc(pred,NetCDFOutFile = paste0(dir %>% str_replace("/HS/", paste0("/Proj_" , iter, "/")), "/", df$var[i], rcm %>% str_remove_all("EUR-25|_32")  %>% str_replace(".rds.xz", paste0("_", simulateName[zz], ".nc"))), globalAttributes = globalAttributeList, varAttributes = varAttributeList)
          rm(pred, pred_aux)
          gc()
        })
      }
    })
  })
})
