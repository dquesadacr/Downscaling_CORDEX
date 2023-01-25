# Written by Dánnell Quesada-Chacón

# Collection of functions to parse and plot the validation metrics

fix_index <- function(x, models) {
  index <- (x %>% redim(drop = TRUE))$Data
  if (length(dim(index)) == 2) {
    dim(index) <- c(1, prod(dim(index)[1:2]))
  } else {
    dim(index) <- c(nrow(index), prod(dim(index)[2:3]))
  }
  
  na_models <- (rowSums(is.na(index)) == ncol(index)) %>% which()
  if (length(na_models) > 0) {
    index <- matrix(index[-na_models, ], nrow = (nrow(index)-length(na_models)))
    
  }
  indLand <- (!apply(index, MARGIN = 2, anyNA)) %>% which()
  
  if(nrow(index)-length(na_models)>1){
    index <- index[, indLand] %>% t() %>% as.data.frame
  } else {
    index <- index[, indLand] %>% as.data.frame
  }
  
  if (length(na_models) > 0) {
    colnames(index) <- models[-na_models]
  } else {
    colnames(index) <- models
  }
  return(index)
}

calc_stat <- function(x) {
  stats <- quantile(x, probs = c(0.1, 0.25, 0.5, 0.75, 0.9))
  names(stats) <- c("ymin", "lower", "middle", "upper", "ymax")
  return(stats)
}

loss_names_all <- list(
  Gauss = list(Rn = "gaussianLoss", Pr = "gaussianLoss", TM = "gaussianLoss", TN = "gaussianLoss", TX = "gaussianLoss", WS = "gaussianLoss", Pw = "gaussianLoss"),
  rmse = list(Rn = "rmse", Pr = "rmse", TM = "rmse", TN = "rmse", TX = "rmse", WS = "rmse", Pw = "rmse"),
  Dist = list(Rn = "gammaLoss", Pr = "bernouilliGammaLoss", TM = "gaussianLoss", TN = "gaussianLoss", TX = "gaussianLoss", WS = "gammaLoss", Pw = "gammaLoss")
)

raw_read_sce <- function(folders, scenarios = NULL) {
  sapply(folders, simplify = F, function(u) {
    setwd(paste0(path, "/V-", u))
    seeds <- dir(path = "./", pattern = "validation", recursive = T) %>%
      str_split("/", simplify = T) %>%
      .[, 1] %>%
      unique()
    
    df <- sapply(seeds, simplify = F, function(x) {
      if(is.null(scenarios)){
        scenarios <- dir(path = x)}
      df_sce <- sapply(scenarios, simplify = F, function(sce){

        runs_details <- dir(path = paste0(x, "/", sce), pattern = "hist_train_CNN") %>%
          str_remove_all(., "hist_train_CNN_|.rda") %>%
          str_replace_all(., "rmse_dist", "rmse-dist") %>%
          str_replace_all(., "Loss_sp_log", "Loss-sp-log") %>%
          str_replace_all(., "Loss_log", "Loss-log") %>%
          str_replace_all(., "Loss_sp", "Loss-sp") %>%
          str_split(., "_", simplify = T) %>%
          as.data.frame(stringsAsFactors = FALSE) %>%
          mutate(across(everything(),str_replace_all, "-", "_"))
        
        vl_tib <- sapply(1:nrow(runs_details), simplify = F, function(y) {
          print(paste0(u, "--", x, "--", sce, "--", paste0(runs_details[y,], collapse = "--")))
          load(paste0("./", x, "/", sce, "/hist_train_CNN_", 
                      paste0(runs_details[y,], collapse = "_"), ".rda"), .GlobalEnv)
          
          load(paste0("./", x, "/", sce, "/validation_CNN_", 
                      paste0(runs_details[y,], collapse = "_"), ".rda"))
          
          validation.list <- validation.list[-length(validation.list)]
          
          models_all_vars <- NULL
          
          for (var in names(validation.list)) {
            bern_loss <- rownames_to_column(as.data.frame(t(sapply(names(history_trains), FUN = function(z) {
              m_index <- which.min(history_trains[[z]]$metrics$val_loss)
              
              if (length(m_index) == 1) {
                return(c(
                  loss = history_trains[[z]]$metrics$loss[m_index],
                  val_loss = history_trains[[z]]$metrics$val_loss[m_index],
                  var_loss = ifelse(any(str_detect(names(history_trains[[z]]$metrics), var) == TRUE),
                                    history_trains[[z]]$metrics[[paste0(var, "_loss")]][m_index],
                                    history_trains[[z]]$metrics$loss[m_index]),
                  var_val_loss = ifelse(any(str_detect(names(history_trains[[z]]$metrics), var) == TRUE),
                                        history_trains[[z]]$metrics[[paste0("val_", var, "_loss")]][m_index],
                                        history_trains[[z]]$metrics$val_loss[m_index]),
                  epochs = length(history_trains[[z]]$metrics$val_loss)
                ))
              } else {
                return(c(
                  loss = NA,
                  val_loss = NA,
                  var_loss = NA,
                  var_val_loss = NA,
                  epochs = length(history_trains[[z]]$metrics$val_loss)
                ))
              }
            }))), var = "models")
            
            if (length(names(history_trains)) == 1) {
              models_summ <- NULL
              for (v in names(validation.list[[var]])) {
                models_summ <- bind_cols(
                  models_summ,
                  validation.list[[var]][[v]]$Data %>%
                    as.vector() %>%
                    t() %>%
                    as_tibble() %>%
                    summarise("median.{v}" := rowMedians(as.matrix(.), na.rm = T))
                )
              }
            } else {
              models_summ <- NULL
              for (v in names(validation.list[[var]])) {
                models_summ <-
                  bind_cols(
                    models_summ,
                    subsetDimension(validation.list[[var]][[v]],
                                    dimension = "var",
                                    indices = 1:length(names(history_trains))
                    ) %>%
                      redim(drop = TRUE) %>%
                      .$Data %>%
                      apply(., 1L, c) %>%
                      t() %>%
                      as_tibble() %>%
                      summarise("median.{v}" := rowMedians(as.matrix(.), na.rm = T))
                  )
              }
            }
            
            models_all <- bind_cols(bern_loss, models_summ)
            models_all$var <- var
            models_all$loss_var_name <- ifelse(!is.null(loss_names_all[[runs_details[1,2]]][[var]]), loss_names_all[[runs_details[1,2]]][[var]], runs_details[which(runs_details[,1]==var),2])
            models_all$var_model <- runs_details[y,1]
            models_all$loss_model_name <- runs_details[y,2]
            models_all$Run <- runs_details[y,3]
            models_all$sce <- sce
            models_all_vars <- bind_rows(models_all_vars, models_all)
          }
          return(as.data.frame(models_all_vars))
        }) %>% rbindlist(., fill=TRUE)
        
        vl_tib$seed <- as.character(x)
        return(vl_tib)
      }) %>% rbindlist(., fill=TRUE)
      
      return(df_sce)
    }) %>% rbindlist(., fill=TRUE)
    
    df$iter <- as.character(u)
    return(df)
  }) %>% rbindlist(., fill=TRUE) %>%
    relocate(Run, seed, iter, .after = epochs)
}

df_filter_plot_sce_all <- function(df, plot_name, plot_height=125) {
  setwd(path)
  df_filter <- sapply(1:nrow(df), function(x) {
    print(x)
    load(paste0("./V-", df$iter[x], "/", df$seed[x], "/", df$sce[x], "/validation_CNN_", 
                df$var_model[x], "_", df$loss_model_name[x], "_", df$Run[x],".rda"))
    ylabs <- str_remove(names(validation.list[[df$var[x]]]), "ts.") %>% 
      str_replace(., "biasRel", "RB") %>% 
      str_replace(., "bias", "B") %>%
      str_replace(., "BP", "Bp") %>%
      str_replace(., "rocss", "ROCSS") %>%
      str_replace(., "rs", "Spearman") %>%
      str_replace(., "rp$", "Pearson") %>%
      str_replace(., "AnnualMaxSpell", "AMS") %>%
      str_replace(., "ratiosd", "RSD") %>%
      str_replace(., "RBMean", "RB") %>%
      str_replace(., "BMean", "BM") %>%
      str_replace(., "cmIndex", "CvMI") %>%
      str_replace(., "cmksdiff", "KSS") %>%
      str_replace(., "Sto", "S")

    names2 <- validation.list$Models[[1]]
    validation.list <- validation.list[-length(validation.list)]
    names(validation.list[[df$var[x]]]) <- ylabs
    
    # Quick fix for rmse pr (all NaN)
    if("Pr" %in% names(validation.list)) {validation.list[["Pr"]][["ROCSS"]] <- NULL}
    
    for (y in grep("%)", names(validation.list[[df$var[x]]]))) {
      validation.list[[df$var[x]]][[y]]$Data <- validation.list[[df$var[x]]][[y]]$Data * 100
    }
    
    validation_fix <- sapply(validation.list[[df$var[x]]], fix_index, models = names2, simplify = F) %>%
      reshape2::melt() %>%
      filter(variable == df$models[x]) %>%
      droplevels()
    
    colnames(validation_fix) <- c("Model", "value", "metric")
    
    models_nice <- df$models[x] %>%
      str_remove_all("-0.25|-le_0.3-le_0.3|_0.3|RUE|ALSE") %>%
      str_remove("T-") 
    
    validation_fix$Model <- models_nice 
    validation_fix$Model_info <- models_nice %>% 
      paste0(., ", R=", df$Rank[x], "\nV=", df$var_model[x], ", L=", df$loss_var_name[x])
    
    levels(validation_fix$Model_info) <- models_nice %>% 
      paste0(., ", R=", df$Rank[x], "\nV=", df$var_model[x], ", L=", df$loss_var_name[x])
    
    validation_fix$metric <- factor(validation_fix$metric, levels = ylabs)
    validation_fix$var_group <- df$var[x]
    validation_fix$var <- df$var[x]
    validation_fix$var_model <- df$var_model[x]
    validation_fix$loss_var_name <- df$loss_var_name[x]
    validation_fix$loss_model_name <- df$loss_model_name[x]
    validation_fix$sce <- names(loss_names_all)[as.numeric(df$sce[x])]
    validation_fix$Rank <- df$Rank[x]
    return(validation_fix)
  }, simplify = F) %>% do.call(rbind.data.frame, .)
  
  df_filter <- df_filter %>% 
    mutate(Model = fct_relevel(Model,  unique(df_filter$Model) %>% 
                                 str_split_fixed(., "-", 5) %>% 
                                 as_tibble %>%
                                 arrange(., V1, V2, desc(V3)) %>% 
                                 unite(., Model, V1:V5, sep = "-", remove = TRUE) %>% 
                                 as.vector %>% unlist(use.names = FALSE))) #%>%
  # df_filter$metric %>% unique()
  
  df_2 <- df_filter %>%
    filter(metric %in% c("RMSE", "Spearman", "KSS", "BM", "Bp98", "Pearson", "RSD", "Bp02", "Bp98S"))  %>%
    mutate(across(metric, str_replace, 'Spearman|Pearson', '\u03C1')) %>%
    mutate(metric = fct_relevel(metric, c("RMSE", "\u03C1", "KSS", "RSD", "BM", "Bp02", "Bp98", "Bp98S"))) %>%
    group_by(Model, metric, var) %>%
    mutate(Rank_y = (quantile(value, 0.025))) %>%
    mutate(across(loss_model_name, str_replace, "bernouilliGammaLoss", "BG"),
           across(loss_model_name, str_replace, "gammaLoss", "Gamma"),
           across(loss_model_name, str_replace, "lognormalLoss", "LN"),
           across(loss_model_name, str_replace, "bernouilliLognormalLoss", "BLN"),
           across(loss_model_name, str_replace, "gaussianLoss", "Gauss"),
           across(loss_model_name, str_replace, "rmse", "RMSE")) %>%
    mutate(loss_model_name = ifelse(var_model == "all", "All", loss_model_name)) %>%
    mutate(loss_rank = paste0(loss_model_name, " [", Rank, "]"))
    
  levels(df_2$loss_model_name)
  df_2$loss_model_name %>% unique

  cnn_plot <- ggplot(df_2, aes(x = loss_rank, y = value, color = Model)) + 
    facet_nested( metric ~ var , scales = "free", independent = "y") +
    stat_summary(fun.data = calc_stat, geom = "boxplot", width = 0.6, lwd = 0.35) + # , size=0.33
    theme_light(base_size = 9, base_family = "Helvetica") +
    scale_y_continuous(breaks= pretty_breaks(n=3)) +
    guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_text(angle = 45, vjust = 1, hjust=1, size=7),
      axis.text.y = element_text(size = 7),
      axis.title.y = element_blank(),
      strip.background = element_rect(fill = "white"),
      strip.text = element_text(color = "black", margin = margin(0, 0, 0.25, 0.25, unit = "mm"), size = 8),
      strip.text.y = element_text(color = "black", margin = margin(0, 0, 0.25, 0.4, unit = "mm"), size = 7),
      legend.key.size = unit(3, "mm"),
      legend.box.margin = margin(-3.75, 0, -1.25, 0, unit = "mm"),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.spacing.y = unit(1.5, "mm"),
      legend.text = element_text(margin = margin(0, 0, 0, 0, unit = "mm")),
      panel.spacing.x = unit(0.75, "mm"),
      panel.spacing.y = unit(2, "mm"),
      plot.margin = margin(-.25, 0, -0.25, 0, unit = "mm")
    )
  nm2plot <- length(unique(df_2$Model))
  if (nm2plot <= 20 && nm2plot >= 10) {
    cnn_plot <- cnn_plot + scale_color_ucscgb()
  } else if (nm2plot <= 10) {
    cnn_plot <- cnn_plot + scale_color_jco()
  } 
  if (nm2plot == 12) {
    cnn_plot <- cnn_plot + scale_color_manual(values = c(
      "#0073c2", "#EFC000", "#A73030", "#868686",
               "#641ea4", "#76CD26", "#E67300", "#1929C8",
               "#cd2926", "#3c3c3c", "#1B6F1B", "#82491E"
    ))
  }
  # Saved as png because rho is not shown in pdf
  ggsave(
    plot = cnn_plot, filename = paste0("./boxplots/validation_", plot_name),
    height = plot_height, width = 175, units = "mm", dpi = 1200
  )
  
  return(paste0("Check plot in: ", path, "/boxplots"))
}
