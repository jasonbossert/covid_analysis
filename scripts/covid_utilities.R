queen_knn_connect <- function(geometries) {
  
  queen_nb <- poly2nb(geometries, row.names=geometries$geoid, queen=TRUE)
  
  count = card(queen_nb)
  if(!any(count==0)){
    return(queen_nb)
  }
  
  coords <- st_centroid(st_geometry(geometries), of_largest_polygon=TRUE)
  nnbs = knearneigh(coords, k=1)$nn
  
  no_edges_from = which(count==0)
  for(i in no_edges_from){
    queen_nb[[i]] = nnbs[i]
  }
  return(queen_nb)
}

spatial_corr <- function(df, neighbors, vari, weeks, nrep = 100) {
  sapply(weeks, function(i) moran.mc(filter(df, week == i)[[vari]], neighbors, nrep)$statistic)
}

chloropleth <- function(geoms, data, var, map_title, legend_title, fill_breaks) {
  
  left_join(geoms, data) %>%
    filter(state_fips != "02" & state_fips != "15") %>%
    tm_shape() +
    tm_fill(var,
            breaks = fill_breaks,
            title = legend_title) +
    tm_borders(col = "grey", lwd = .5) +
    tm_layout(title = map_title,
              title.size = 1.1,
              title.position = c("center", "top"),
              inner.margins = c(0.06, 0.10, 0.10, 0.08),
              frame = FALSE)
}

summarize_df <- function(df, reductions) {
  df %>%
    select_if(is.numeric) %>%
    summarise(across(everything(), reductions)) %>%
    pivot_longer(everything(),
                 names_to = c("variable", "summary"),
                 names_sep = "_(?!.*_)",
                 values_to = "value") %>%
    pivot_wider(names_from = summary, values_from = value)
}

aic_2 <- function(resid, k) {
  n <- length(resid)
  2*k + n * (log(2*pi*sum(resid^2) / n) + 1)
}

bic_2 <- function(resid, k) {
  n <- length(resid)
  log(n)*k + n * (log(2*pi*sum(resid^2) / n) + 1)
}

rmse <- function(resid) {
  sqrt(mean((resid)^2))
}

adj_rmse <- function(resid, k) {
  n <- length(resid)
  sqrt(sum((resid)^2) / (n - k - 1))
}

mae <- function(resid) {
  mean(abs(resid))
}

adj_mae <- function(resid, k) {
  n <- length(resid)
  sum(abs(resid)) / (n - k - 1)
}

calc_metrics <- function(df, k, name) {
  resid <- df$deaths_resid
  return(list("name" = name,
              "AIC" = aic_2(resid, k), 
              "BIC" = bic_2(resid, k), 
              "RMSE" = rmse(resid), 
              "Adj_RMSE" = adj_rmse(resid, k), 
              "MAE" = mae(resid), 
              "Adj_MAE" = adj_mae(resid, k)))
}

make_add_formula_from_varnums <- function(vars, data, response) {
  var_names <- colnames(data)[vars]
  regressors = paste(var_names, collapse = " + ")
  formula(paste(response, regressors, sep = " ~ "))
}

fmt_df_to_mat <- function(df, design_formula) {
  X <- model.matrix(design_formula, data = df)
  preproc_X <- preProcess(X, method = c('center', 'scale'))
  X_trans <- predict(preproc_X, X)
  return(X_trans)
}

build_3d_design_matrix <- function(data, split_val, design_formula) {
  dfs <- split(data, data[[split_val]])
  res <- sapply(dfs, function(x) fmt_df_to_mat(x, design_formula), simplify = "array")
  aperm(res, c(1, 3, 2))
}

build_2d_df_matrix <- function(data, split_val, target_val) {
  dfs <- split(data[[target_val]], data[[split_val]])
  res <- sapply(dfs, function(x) as.matrix(x), simplify = "array")
  drop(res)
}

filter_weeks_states <- function(df, states, weeks) {
  df %>%
    filter(week %in% weeks) %>%
    filter(state_fips %in% states)
}

asinh <- function(arg) {
  log(arg + sqrt(arg^2 + 1))
}

write_stan_chunk <- function(path, block) {
  blocks <- c("functions", "data", "transformed data", "parameters", "transformed parameters", "model", "generated quantities")
  if (!(block %in% blocks)) {
    stop("Specified Stan block is not a valid Stan program block.")
  }
  
  target_idx <- which(str_detect(blocks, block))
  
  my_string <- readLines(path)
  idxs <- c(0, 0, 0, 0, 0, 0, 0)
  
  for (i in 1:length(blocks)) {
    idxs[i] <- which(str_detect(my_string, blocks[i]))[1]
  }
  
  if (is.na(idxs[target_idx])) {
    stop("Stan program does not have specified block.")
  }
  
  i1 <- idxs[target_idx]
  if (target_idx < 7)
    i2 <- idxs[target_idx + 1] - 1
  else
    i2 <- length(my_string)
  
  writeLines(my_string[i1:i2])
}

