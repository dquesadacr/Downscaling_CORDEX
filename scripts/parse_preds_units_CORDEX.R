# Written by Dánnell Quesada

options(java.parameters = "-Xmx144000m")
library(climate4R.value)
library(transformeR)
library(loadeR)
library(sp)
library(tidyverse)
library(magrittr)
library(raster)
library(data.table)

# This script reads the netcdf files from CORDEX, transforms the units,
# rearrange the order of the variables to fit ERA5 and saves as .rds

nc.files <- list.files(pattern = ".nc$")

options("scipen"=10)
dir.create("rds", showWarnings=FALSE)

saveRDS.xz <- function(object,file,threads=parallel::detectCores()) {
  con <- pipe(paste0("xz -T",threads," > ",file),"wb")
  saveRDS(object, file = con, compress=FALSE)
  close(con)
}

sapply(nc.files, FUN= function(z) {
    print(paste0("Parsing: ", z, "..."))
    di <- dataInventory(z)
    nc.var <- names(di)

    sep.vars <- sapply(names(di), FUN = function(x) {
        if(is.null(di[[x]]$Dimensions$level$Values)) {
            x
        } else {
            apply(expand.grid(x, di[[x]]$Dimensions$level$Values), 1, paste, collapse= "@") %>% str_remove(., " ")
        }
    }, simplify=TRUE) %>% unlist(., recursive=TRUE, use.names=FALSE)

    # Fix order of predictors so that they fit ERA5's and remove the ones not used
    sep.vars <- sep.vars[c(c(5:9),c(15:19),c(20:24),c(10:14),c(25:29),4)]
    
    # Fix units
    x <- lapply(sep.vars, FUN = function(y) {
        print(y)
        gc()
        if(str_detect(y, "zg")){
          grid <- loadGridData(z, y) %>% gridArithmetics(., 9.80665)
          attr(grid$Variable, "units") <- "m**2 s**-2"
          attr(grid$Variable, "description") <- "Geopotential"
        } else if(str_detect(y, "cl")){
          grid <- loadGridData(z, y) %>% gridArithmetics(., 100, operator = "/")
          attr(grid$Variable, "units") <- "(0 - 1)" 
        } else {
          grid <- loadGridData(z, y)
        }
        return(grid)
    }) %>% makeMultiGrid()

    gc()
    cat(paste0("Multigrid for ", z, " done. Now saving...\n"))

    saveRDS.xz(x, file= paste0("./rds/", z, ".rds.xz"), threads = 6)

    gc()
})


