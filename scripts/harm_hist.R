# Written by Dánnell Quesada

options(java.parameters = "-Xmx144000m")
library(loadeR)
library(transformeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(sp)
library(stringr)

# This script harmonizes (bias correction) the historical predictors only 
# to produce the portrait plots

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

folders <- system("find ./ -mindepth 3 -maxdepth 3 -type d", intern=TRUE) %>% str_subset("/Proj_|/HS/|ERAINT", negate=TRUE)

dir.create("H_hist")

sapply(folders, simplify = FALSE, FUN = function(i){
    setwd(path)
    setwd(i)

    files <- list.files(pattern = "*.rds*") %>% str_subset("historical")
    dir.create(str_replace(i, "./", "/data/rds/H_hist/"), recursive = TRUE, showWarnings = FALSE)

    for (j in files) {
      print(paste0(i, " -- ", j))

      output.file <- paste0(str_replace(i, "./", "/data/rds/H_hist/"), "/", j %>% str_remove(".nc"))

      if (!all(file.exists(output.file))){
        x <- readRDS(j) %>% subsetGrid(var = .$Variable$varName[!.$Variable$varName %in% c("clh","cll","clm")])
        print(paste0("Amount of NAs: ", is.na(x$Data) %>% sum, " of ", length(x$Data)))

          if (str_detect(j, "historical|evaluation")) {
            harmonize.base <- subsetGrid(x,  years = 1979:2005)
          }
          x.harm <- scalingDeltaMapping(x, base = harmonize.base, ref = xT)

          print("Saving harmonized dataset")
          saveRDS.xz(x.harm, file = output.file[1], threads=4)
          rm(xn)

          gc()
      }
    }
})
