# CIF Feature Selection + XGBoost with Hyperparameter Tuning
# Structure: 7 Repeats x 10 Outer x 5 Inner
# Tuning: nrounds, max_depth, eta (AUC criterion)

library(party)
library(readxl)
library(caret)
library(pROC)
library(xgboost)

# Load data
df <- read_excel("final_ov.xlsx")

y <- factor(ifelse(df$`2 group class` == "Cancer", 1, 0), levels = c("0", "1"))
y_num <- as.numeric(y) - 1
feature_cols <- grep("^hsa-", names(df), value = TRUE)
X <- as.matrix(df[, feature_cols])
colnames(X) <- make.names(colnames(X))

# Settings
n_repeats <- 7
n_outer <- 10
n_inner <- 5
mtry_cif <- 50

# XGB grid (matching Python sklearn grid)
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1)
)

# Storage
results_xgb <- data.frame()

# Metric function
calc_metrics <- function(actual, pred_class, pred_prob) {
  actual_num <- as.numeric(actual) - 1
  pred_num <- as.numeric(pred_class) - 1
  
  tp <- sum(pred_num == 1 & actual_num == 1)
  tn <- sum(pred_num == 0 & actual_num == 0)
  fp <- sum(pred_num == 1 & actual_num == 0)
  fn <- sum(pred_num == 0 & actual_num == 1)
  
  auc <- as.numeric(suppressMessages(pROC::auc(actual_num, pred_prob)))
  acc <- (tp + tn) / (tp + tn + fp + fn)
  sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  spec <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  ppv <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  npv <- ifelse((tn + fn) > 0, tn / (tn + fn), 0)
  f1 <- ifelse((ppv + sens) > 0, 2 * ppv * sens / (ppv + sens), 0)
  
  return(c(AUC=auc, Acc=acc, Sens=sens, Spec=spec, PPV=ppv, NPV=npv, F1=f1))
}

set.seed(42)

for (r in 1:n_repeats) {
  cat(sprintf("\n-- Repeat %d/%d --\n", r, n_repeats))
  
  folds <- createFolds(y, k = n_outer, list = TRUE)
  
  for (i in 1:n_outer) {
    cat(sprintf("Outer fold %d/%d ... ", i, n_outer))
    
    # Outer split
    test_idx <- folds[[i]]
    trainval_idx <- setdiff(1:length(y), test_idx)
    
    X_trainval <- X[trainval_idx, ]
    y_trainval <- y[trainval_idx]
    y_trainval_num <- y_num[trainval_idx]
    X_test <- X[test_idx, ]
    y_test <- y[test_idx]
    y_test_num <- y_num[test_idx]
    
    # Train/Val split (80/20)
    set.seed(42 + r * 100 + i)
    val_idx <- createDataPartition(y_trainval, p = 0.2, list = FALSE)
    X_val <- X_trainval[val_idx, ]
    y_val <- y_trainval[val_idx]
    X_train <- X_trainval[-val_idx, ]
    y_train <- y_trainval[-val_idx]
    y_train_num <- as.numeric(y_train) - 1
    
    # CIF Feature Selection on train only
    train_df <- data.frame(y = y_train, X_train)
    
    cif_model <- cforest(
      y ~ .,
      data = train_df,
      controls = cforest_unbiased(ntree = 500, mtry = mtry_cif)
    )
    
    varimp_cif <- varimp(cif_model, conditional = FALSE)
    selected_features <- names(varimp_cif[varimp_cif > 0])
    n_selected <- length(selected_features)
    
    if (n_selected == 0) {
      selected_features <- names(sort(varimp_cif, decreasing = TRUE)[1:50])
      n_selected <- 50
    }
    
    # Subset (no scaling for XGB)
    X_train_sel <- X_train[, selected_features, drop = FALSE]
    X_trainval_sel <- X_trainval[, selected_features, drop = FALSE]
    X_test_sel <- X_test[, selected_features, drop = FALSE]
    
    # Inner CV: tune XGB hyperparameters (AUC criterion)
    best_nrounds <- 100
    best_max_depth <- 3
    best_eta <- 0.1
    best_inner_auc <- 0
    
    inner_folds <- createFolds(y_train, k = n_inner, list = TRUE)
    
    for (g in 1:nrow(xgb_grid)) {
      nr <- xgb_grid$nrounds[g]
      md <- xgb_grid$max_depth[g]
      et <- xgb_grid$eta[g]
      
      auc_inner <- c()
      
      for (j in 1:n_inner) {
        inner_val_idx <- inner_folds[[j]]
        X_tr <- as.matrix(X_train_sel[-inner_val_idx, ])
        y_tr <- y_train_num[-inner_val_idx]
        X_vl <- as.matrix(X_train_sel[inner_val_idx, ])
        y_vl <- y_train[inner_val_idx]
        
        dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
        dval <- xgb.DMatrix(data = X_vl)
        
        params <- list(
          objective = "binary:logistic",
          max_depth = md,
          eta = et,
          eval_metric = "logloss"
        )
        
        model <- xgb.train(params = params, data = dtrain, nrounds = nr, verbose = 0)
        pred_prob <- predict(model, dval)
        auc <- as.numeric(suppressMessages(pROC::auc(y_vl, pred_prob)))
        auc_inner <- c(auc_inner, auc)
      }
      
      mean_auc <- mean(auc_inner)
      if (mean_auc > best_inner_auc) {
        best_inner_auc <- mean_auc
        best_nrounds <- nr
        best_max_depth <- md
        best_eta <- et
      }
    }
    
    # Final model on trainval
    dtrain_full <- xgb.DMatrix(data = as.matrix(X_trainval_sel), label = y_trainval_num)
    dtest <- xgb.DMatrix(data = as.matrix(X_test_sel))
    
    params_final <- list(
      objective = "binary:logistic",
      max_depth = best_max_depth,
      eta = best_eta,
      eval_metric = "logloss"
    )
    
    final_model <- xgb.train(params = params_final, data = dtrain_full,
                             nrounds = best_nrounds, verbose = 0)
    xgb_prob <- predict(final_model, dtest)
    xgb_pred <- factor(ifelse(xgb_prob >= 0.5, 1, 0), levels = c("0", "1"))
    xgb_metrics <- calc_metrics(y_test, xgb_pred, xgb_prob)
    
    results_xgb <- rbind(results_xgb, data.frame(
      Repeat = r, Outer_Fold = i, t(xgb_metrics),
      N_Selected = n_selected,
      Best_nrounds = best_nrounds,
      Best_max_depth = best_max_depth,
      Best_eta = best_eta
    ))
    
    cat(sprintf("N=%d, nrounds=%d, depth=%d, eta=%.2f, AUC=%.3f, Sens=%.3f, Spec=%.3f\n",
                n_selected, best_nrounds, best_max_depth, best_eta,
                xgb_metrics["AUC"], xgb_metrics["Sens"], xgb_metrics["Spec"]))
  }
}

# Summary
cat("\n-- CIF -> XGB (Tuned) --\n")
cat(sprintf("AUC:  %.3f ± %.3f\n", mean(results_xgb$AUC), sd(results_xgb$AUC)))
cat(sprintf("Acc:  %.3f ± %.3f\n", mean(results_xgb$Acc), sd(results_xgb$Acc)))
cat(sprintf("Sens: %.3f ± %.3f\n", mean(results_xgb$Sens), sd(results_xgb$Sens)))
cat(sprintf("Spec: %.3f ± %.3f\n", mean(results_xgb$Spec), sd(results_xgb$Spec)))
cat(sprintf("PPV:  %.3f ± %.3f\n", mean(results_xgb$PPV), sd(results_xgb$PPV)))
cat(sprintf("NPV:  %.3f ± %.3f\n", mean(results_xgb$NPV), sd(results_xgb$NPV)))
cat(sprintf("F1:   %.3f ± %.3f\n", mean(results_xgb$F1), sd(results_xgb$F1)))
cat(sprintf("|S-S|: %.3f\n", abs(mean(results_xgb$Sens) - mean(results_xgb$Spec))))

cat("\nBest params:\n")
cat("nrounds:\n"); print(table(results_xgb$Best_nrounds))
cat("max_depth:\n"); print(table(results_xgb$Best_max_depth))
cat("eta:\n"); print(table(results_xgb$Best_eta))

write.csv(results_xgb, "cif_XGB_tuned_results.csv", row.names = FALSE)
cat("\nSaved: cif_XGB_tuned_results.csv\n")