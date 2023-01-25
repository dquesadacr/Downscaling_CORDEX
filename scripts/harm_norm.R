# Written by Dánnell Quesada

options(java.parameters = "-Xmx144000m")
library(loadeR)
library(transformeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(sp)
library(stringr)

# This script harmonizes (bias correction) and normalizes the predictors

scalingDeltaMapping <- function(grid, base, ref) {
  ### remove the seasonal trend
  grid_detrended <- scaleGrid(grid,
                base = grid,
                ref = base,
                type = "center",
                spatial.frame = "gridbox",
                time.frame = "monthly")
  ### bias correct the mean and variance
  grid_detrended_corrected <- scaleGrid(grid_detrended,
                     base = base,
                     ref = ref,
                     type = "standardize",
                     spatial.frame = "gridbox",
                     time.frame = "monthly")
  ### add the seasonal trend
  grid_corrected <- scaleGrid(grid_detrended_corrected,
                base = base,
                ref = grid,
                type = "center",
                spatial.frame = "gridbox",
                time.frame = "monthly")
  ### return
  return(grid_corrected)
}

standardize <- function(grid, base){
  scaleGrid(grid,
    base = base,
    ref = NULL,
    type = "standardize"
  )
}

saveRDS.xz <- function(object,file,threads=parallel::detectCores()) {
  con <- pipe(paste0("xz -T",threads," > ",file),"wb")
  saveRDS(object, file = con, compress=FALSE)
  close(con)
}

x <- readRDS("/data/x_32.rds")

xT <- subsetGrid(x, var = x$Variable$varName[!x$Variable$varName %in% c("hcc","lcc","mcc")],  years = 1979:2005)
rm(x)
gc()

path <- getwd()

folders <- system("find ./ -mindepth 3 -maxdepth 3 -type d", intern=TRUE)

dir.create("HS")

sapply(folders, simplify = FALSE, FUN = function(i){
    setwd(path)
    setwd(i)

    files <- list.files(pattern = "*.rds*")
    dir.create(str_replace(i, "./", "/data/rds/HS/"), recursive = TRUE, showWarnings = FALSE)

    for (j in files) {
      print(paste0(i, " -- ", j))

      output.file <- paste0(str_replace(i, "./", "/data/rds/HS/"), "/", j %>% str_remove(".nc"))
      ifelse(str_detect(j, "evaluation"), output.file %<>% str_replace("\\.rds", c(".rds", "_norm.rds")), output.file)

      if (!all(file.exists(output.file))){
        x <- readRDS(j) %>% subsetGrid(var = .$Variable$varName[!.$Variable$varName %in% c("clh","cll","clm")])
        print(paste0("Amount of NAs: ", is.na(x$Data) %>% sum, " of ", length(x$Data)))

          if (str_detect(j, "historical|evaluation")) {
            harmonize.base <- subsetGrid(x,  years = 1979:2005)
          }
          x.harm <- scalingDeltaMapping(x, base = harmonize.base, ref = xT)
          xn <- standardize(x.harm, base = xT)  %>% redim(drop = TRUE)

          print("Saving harmonized-standardized dataset")
          saveRDS.xz(xn, file = output.file[1], threads=4)
          rm(xn)

          # To test non-harmonized evaluation data
          if (str_detect(j, "evaluation")){
            xn <- standardize(x, base = xT)
            print("Saving standardized only evaluation dataset")
            saveRDS.xz(xn, file = output.file[2], threads=4)
            rm(xn)
          }
          gc()
      }
    }
})
