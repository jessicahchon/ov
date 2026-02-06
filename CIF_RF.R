# CIF Feature Selection + RF with Hyperparameter Tuning
# Structure: 7 Repeats x 10 Outer x 5 Inner
# Tuning: ntree, maxnodes (max_depth proxy), nodesize (AUC criterion)

library(party)
library(readxl)
library(caret)
library(pROC)
library(randomForest)

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

# RF grid (matching Python sklearn grid)
# max_depth 5 -> maxnodes 32, 10 -> 1024, None -> NULL
rf_grid <- expand.grid(
  ntree = c(100, 200, 300),
  maxnodes = c(32, 1024, -1),  # -1 = NULL (no limit)
  nodesize = c(1, 5, 9)
)

# Storage
results_rf <- data.frame()

# Metric function
calc_metrics <- function(actual, pred_class, pred_prob) {
  actual_num <- as.numeric(actual) - 1
  pred_num <- as.numeric(pred_class) - 1
  
  tp <- sum(pred_num == 1 & actual_num == 1)
  tn <- sum(pred_num == 0 & actual_num == 0)
  fp <- sum(pred_num == 1 & actual_num == 0)
  fn <- sum(pred_num == 0 & actual_num == 1)
  
  auc <- as.numeric(pROC::auc(actual_num, pred_prob))
  acc <- (tp + tn) / (tp + tn + fp + fn)
  sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  spec <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  ppv <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  npv <- ifelse((tn + fn) > 0, tn / (tn + fn), 0)
  f1 <- ifelse((ppv + sens) > 0, 2 * ppv * sens / (ppv + sens), 0)
  
  return(c(AUC=auc, Acc=acc, Sens=sens, Spec=spec, PPV=ppv, NPV=npv, F1=f1))
}

# Helper: train RF with given params
train_rf <- function(X_train, y_train, ntree, maxnodes, nodesize) {
  if (maxnodes == -1) {
    randomForest(x = X_train, y = y_train,
                 ntree = ntree, nodesize = nodesize)
  } else {
    randomForest(x = X_train, y = y_train,
                 ntree = ntree, maxnodes = maxnodes, nodesize = nodesize)
  }
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
    X_test <- X[test_idx, ]
    y_test <- y[test_idx]
    
    # Train/Val split (80/20)
    set.seed(42 + r * 100 + i)
    val_idx <- createDataPartition(y_trainval, p = 0.2, list = FALSE)
    X_val <- X_trainval[val_idx, ]
    y_val <- y_trainval[val_idx]
    X_train <- X_trainval[-val_idx, ]
    y_train <- y_trainval[-val_idx]
    
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
    
    # Subset to selected features (no scaling for RF)
    X_train_sel <- X_train[, selected_features, drop = FALSE]
    X_trainval_sel <- X_trainval[, selected_features, drop = FALSE]
    X_test_sel <- X_test[, selected_features, drop = FALSE]
    
    # Inner CV: tune RF hyperparameters (AUC criterion)
    best_ntree <- 100
    best_maxnodes <- -1
    best_nodesize <- 1
    best_inner_auc <- 0
    
    inner_folds <- createFolds(y_train, k = n_inner, list = TRUE)
    
    for (g in 1:nrow(rf_grid)) {
      nt <- rf_grid$ntree[g]
      mn <- rf_grid$maxnodes[g]
      ns <- rf_grid$nodesize[g]
      
      auc_inner <- c()
      
      for (j in 1:n_inner) {
        inner_val_idx <- inner_folds[[j]]
        X_tr <- X_train_sel[-inner_val_idx, ]
        y_tr <- y_train[-inner_val_idx]
        X_vl <- X_train_sel[inner_val_idx, ]
        y_vl <- y_train[inner_val_idx]
        
        model <- train_rf(X_tr, y_tr, nt, mn, ns)
        pred_prob <- predict(model, X_vl, type = "prob")[, 2]
        auc <- as.numeric(pROC::auc(y_vl, pred_prob))
        auc_inner <- c(auc_inner, auc)
      }
      
      mean_auc <- mean(auc_inner)
      if (mean_auc > best_inner_auc) {
        best_inner_auc <- mean_auc
        best_ntree <- nt
        best_maxnodes <- mn
        best_nodesize <- ns
      }
    }
    
    # Final model on trainval
    final_model <- train_rf(X_trainval_sel, y_trainval, best_ntree, best_maxnodes, best_nodesize)
    
    rf_prob <- predict(final_model, X_test_sel, type = "prob")[, 2]
    rf_pred <- factor(ifelse(rf_prob >= 0.5, 1, 0), levels = c("0", "1"))
    rf_metrics <- calc_metrics(y_test, rf_pred, rf_prob)
    
    results_rf <- rbind(results_rf, data.frame(
      Repeat = r, Outer_Fold = i, t(rf_metrics),
      N_Selected = n_selected,
      Best_ntree = best_ntree,
      Best_maxnodes = best_maxnodes,
      Best_nodesize = best_nodesize
    ))
    
    cat(sprintf("N=%d, ntree=%d, maxnodes=%d, nodesize=%d, AUC=%.3f, Sens=%.3f, Spec=%.3f\n",
                n_selected, best_ntree, best_maxnodes, best_nodesize,
                rf_metrics["AUC"], rf_metrics["Sens"], rf_metrics["Spec"]))
  }
}

# Summary
cat("\n-- CIF -> RF (Tuned) --\n")
cat(sprintf("AUC:  %.3f ± %.3f\n", mean(results_rf$AUC), sd(results_rf$AUC)))
cat(sprintf("Acc:  %.3f ± %.3f\n", mean(results_rf$Acc), sd(results_rf$Acc)))
cat(sprintf("Sens: %.3f ± %.3f\n", mean(results_rf$Sens), sd(results_rf$Sens)))
cat(sprintf("Spec: %.3f ± %.3f\n", mean(results_rf$Spec), sd(results_rf$Spec)))
cat(sprintf("PPV:  %.3f ± %.3f\n", mean(results_rf$PPV), sd(results_rf$PPV)))
cat(sprintf("NPV:  %.3f ± %.3f\n", mean(results_rf$NPV), sd(results_rf$NPV)))
cat(sprintf("F1:   %.3f ± %.3f\n", mean(results_rf$F1), sd(results_rf$F1)))
cat(sprintf("|S-S|: %.3f\n", abs(mean(results_rf$Sens) - mean(results_rf$Spec))))

cat("\nBest params distribution:\n")
cat("ntree:\n"); print(table(results_rf$Best_ntree))
cat("maxnodes:\n"); print(table(results_rf$Best_maxnodes))
cat("nodesize:\n"); print(table(results_rf$Best_nodesize))

write.csv(results_rf, "cif_RF_tuned_results.csv", row.names = FALSE)
cat("\nSaved: cif_RF_tuned_results.csv\n")