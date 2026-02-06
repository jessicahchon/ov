# CIF Feature Selection + LR with Hyperparameter Tuning
# Structure: 7 Repeats x 10 Outer x 5 Inner
# Tuning: alpha (l1_ratio) + lambda via cv.glmnet (AUC criterion)

library(party)
library(readxl)
library(caret)
library(pROC)
library(glmnet)

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
alpha_grid <- c(0.0, 0.05, 0.1, 0.3)

# Storage
results_lr <- data.frame()

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
    
    # Subset to selected features
    X_train_sel <- X_train[, selected_features, drop = FALSE]
    X_trainval_sel <- X_trainval[, selected_features, drop = FALSE]
    X_test_sel <- X_test[, selected_features, drop = FALSE]
    
    # Scale
    scaler <- preProcess(as.data.frame(X_train_sel), method = c("center", "scale"))
    X_train_sc <- as.matrix(predict(scaler, as.data.frame(X_train_sel)))
    
    scaler_full <- preProcess(as.data.frame(X_trainval_sel), method = c("center", "scale"))
    X_trainval_sc <- as.matrix(predict(scaler_full, as.data.frame(X_trainval_sel)))
    X_test_sc <- as.matrix(predict(scaler_full, as.data.frame(X_test_sel)))
    
    # Inner CV: tune alpha + lambda (AUC criterion)
    best_alpha <- 0
    best_lambda <- 0.01
    best_inner_auc <- 0
    
    for (a in alpha_grid) {
      cv_fit <- cv.glmnet(
        X_train_sc, y_train_num,
        family = "binomial",
        alpha = a,
        nfolds = n_inner,
        type.measure = "auc"
      )
      
      # Best AUC for this alpha
      max_auc <- max(cv_fit$cvm)
      
      if (max_auc > best_inner_auc) {
        best_inner_auc <- max_auc
        best_alpha <- a
        best_lambda <- cv_fit$lambda.min
      }
    }
    
    # Final model on trainval
    lr_model <- glmnet(
      X_trainval_sc, y_trainval_num,
      family = "binomial",
      alpha = best_alpha,
      lambda = best_lambda
    )
    
    lr_prob <- as.numeric(predict(lr_model, X_test_sc, type = "response"))
    lr_pred <- factor(ifelse(lr_prob >= 0.5, 1, 0), levels = c("0", "1"))
    lr_metrics <- calc_metrics(y_test, lr_pred, lr_prob)
    
    results_lr <- rbind(results_lr, data.frame(
      Repeat = r, Outer_Fold = i, t(lr_metrics),
      N_Selected = n_selected, Best_alpha = best_alpha, Best_lambda = best_lambda
    ))
    
    cat(sprintf("N=%d, alpha=%.2f, lambda=%.4f, AUC=%.3f, Sens=%.3f, Spec=%.3f\n",
                n_selected, best_alpha, best_lambda, lr_metrics["AUC"],
                lr_metrics["Sens"], lr_metrics["Spec"]))
  }
}

# Summary
cat("\n-- CIF -> LR (Tuned) --\n")
cat(sprintf("AUC:  %.3f ± %.3f\n", mean(results_lr$AUC), sd(results_lr$AUC)))
cat(sprintf("Acc:  %.3f ± %.3f\n", mean(results_lr$Acc), sd(results_lr$Acc)))
cat(sprintf("Sens: %.3f ± %.3f\n", mean(results_lr$Sens), sd(results_lr$Sens)))
cat(sprintf("Spec: %.3f ± %.3f\n", mean(results_lr$Spec), sd(results_lr$Spec)))
cat(sprintf("PPV:  %.3f ± %.3f\n", mean(results_lr$PPV), sd(results_lr$PPV)))
cat(sprintf("NPV:  %.3f ± %.3f\n", mean(results_lr$NPV), sd(results_lr$NPV)))
cat(sprintf("F1:   %.3f ± %.3f\n", mean(results_lr$F1), sd(results_lr$F1)))
cat(sprintf("|S-S|: %.3f\n", abs(mean(results_lr$Sens) - mean(results_lr$Spec))))

cat("\nBest alpha distribution:\n")
print(table(results_lr$Best_alpha))

write.csv(results_lr, "cif_LR_tuned_results.csv", row.names = FALSE)
cat("\nSaved: cif_LR_tuned_results.csv\n")