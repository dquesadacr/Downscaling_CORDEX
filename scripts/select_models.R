# Written by Dánnell Quesada-Chacón

# Load Packages (some might not be necessary anymore)
library(raster)
library(transformeR)
library(visualizeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(RColorBrewer)
library(sp)
library(sf)
library(tidyverse)
library(ggsci)
library(cowplot)
library(matrixStats)
library(rowr)
library(ggpubr)
library(grid)
library(scales)
library(viridis)
library(ggnewscale)
library(colorspace)
library(ggh4x)
library(data.table)

# Remove warnings
options(warn = -1)

source("./parse_aux/aux_results.R") # Functions to parse and plot
# source("/home/dqc/Documents/PhD/scripts/R/Proj/git/test/aux_results.R")

# Set path where the individual folders with the validation results are, change accordingly
# path <- "/home/dqc/Documents/PhD/TAURUS/Proj/Data/all/"
path <- paste0(getwd(), "/val_hist")
setwd(path)

# Create folders for figures
dir.create("./boxplots")

# Load the metrics for all seeds with 1 run first, the name should match the folder created with the validation results (without the "V-")
iteration_name <- "fin"
iteration_name <- "36"
models_sce <- raw_read_sce(iteration_name)

desc_rank <- colnames(models_sce) %>% str_subset("median") %>% str_subset("rocss|ts.rs|ts.rp")
sd_rank <- colnames(models_sce) %>% str_subset("ratiosd")
asc_rank <- colnames(models_sce) %>% str_subset("median") %>% setdiff(c(desc_rank,sd_rank)) 
# Remove ROCSS from the ranking beacuse of NaNs by rmse
desc_rank %<>% str_subset("rocss", negate = TRUE)

ranks_all <- models_sce %>%
  group_by(var) %>%
  nest() %>%
  summarise(ranks = map(data, ~ filter(., Run == 1) %>%
                          mutate(across(all_of(desc_rank), .fns = list(rank= ~percent_rank(desc(round(.x,2)))))) %>%
                          mutate(across(all_of(sd_rank), .fns = list(rank= ~percent_rank(abs(round(.x-1,2)))))) %>%
                          mutate(across(all_of(asc_rank), .fns = list(rank= ~percent_rank(abs(round(.x,2)))))) %>%
                          mutate(Sums = rowSums(across(contains("rank")), na.rm = TRUE)) %>%
                          arrange(., Sums) %>% 
                          mutate(Rank = 1:nrow(.)) %>%
                          as.tibble() %>% 
                          relocate(.,loss_var_name,var_model,loss_model_name,sce,Sums,Rank, .after = iter))) %>%
  unnest(cols = c(ranks))

# Best model per predictand, individual and jointly
ranks1pE <- ranks_all %>%
  group_by(var_model, models, iter, sce) %>%
  arrange(., var_model, models, iter) %>%
  mutate(Ssums = ifelse(n()>1, sum(Sums), Sums)) %>%
  arrange(Ssums, val_loss) %>%
  relocate(.,loss_var_name,var_model,loss_model_name,sce,Sums,Rank,Ssums, .after = iter)  %>% 
  arrange(., var, Ssums) %>%
  ungroup()

ranks1pE <- bind_rows(ranks1pE %>% filter(var_model!="all") %>% group_by(loss_model_name, var_model, var) %>% slice_head(n=1),
                      ranks1pE %>% filter(var_model=="all") %>% group_by(var) %>% slice_head(n=1)) %>% arrange(.,var, Rank)

proj_models <- ranks1pE %>% group_by(var) %>% filter(!var_model == "all") %>% arrange(Rank) %>% slice_head(n=1) %>% ungroup() %>% arrange(var, Rank)

plot_name <- "fin.pdf"

df_filter_plot_sce_all(ranks1pE, plot_name, 100)

# This is passed to the proj_rcms.R routine
saveRDS(proj_models, file = paste0("proj_models_", z, ".rds"))
