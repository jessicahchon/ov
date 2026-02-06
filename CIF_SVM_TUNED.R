# CIF Feature Selection + SVM with Hyperparameter Tuning
# Structure: 7 Repeats x 10 Outer x 5 Inner
# Tuning: C, kernel, gamma (AUC criterion)

library(party)
library(readxl)
library(caret)
library(pROC)
library(e1071)

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

# SVM grid (matching Python sklearn grid)
svm_grid <- expand.grid(
  C = c(0.1, 1.0, 10.0),
  kernel = c("radial", "linear"),
  gamma = c("scale", "auto"),
  stringsAsFactors = FALSE
)
# Remove gamma for linear kernel (not used)
# Keep all combos, gamma is ignored for linear internally

# Storage
results_svm <- data.frame()

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

# Helper: compute gamma value
get_gamma <- function(gamma_str, n_features) {
  if (gamma_str == "scale") {
    return(1 / n_features)
  } else {
    return(1 / n_features)
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
    
    # Subset + Scale (SVM needs scaling)
    X_train_sel <- X_train[, selected_features, drop = FALSE]
    X_trainval_sel <- X_trainval[, selected_features, drop = FALSE]
    X_test_sel <- X_test[, selected_features, drop = FALSE]
    
    scaler <- preProcess(as.data.frame(X_train_sel), method = c("center", "scale"))
    X_train_sc <- as.matrix(predict(scaler, as.data.frame(X_train_sel)))
    
    scaler_full <- preProcess(as.data.frame(X_trainval_sel), method = c("center", "scale"))
    X_trainval_sc <- as.matrix(predict(scaler_full, as.data.frame(X_trainval_sel)))
    X_test_sc <- as.matrix(predict(scaler_full, as.data.frame(X_test_sel)))
    
    # Inner CV: tune SVM hyperparameters (AUC criterion)
    best_C <- 1.0
    best_kernel <- "radial"
    best_gamma_str <- "scale"
    best_inner_auc <- 0
    
    inner_folds <- createFolds(y_train, k = n_inner, list = TRUE)
    
    for (g in 1:nrow(svm_grid)) {
      sv_C <- svm_grid$C[g]
      sv_kernel <- svm_grid$kernel[g]
      sv_gamma_str <- svm_grid$gamma[g]
      
      # Compute gamma value
      if (sv_gamma_str == "scale") {
        sv_gamma <- 1 / (n_selected * var(as.vector(X_train_sc)))
      } else {
        sv_gamma <- 1 / n_selected
      }
      
      auc_inner <- c()
      
      for (j in 1:n_inner) {
        inner_val_idx <- inner_folds[[j]]
        X_tr <- X_train_sc[-inner_val_idx, ]
        y_tr <- y_train[-inner_val_idx]
        X_vl <- X_train_sc[inner_val_idx, ]
        y_vl <- y_train[inner_val_idx]
        
        if (sv_kernel == "linear") {
          model <- svm(x = X_tr, y = y_tr, kernel = "linear",
                       cost = sv_C, probability = TRUE)
        } else {
          model <- svm(x = X_tr, y = y_tr, kernel = "radial",
                       cost = sv_C, gamma = sv_gamma, probability = TRUE)
        }
        
        pred_obj <- predict(model, X_vl, probability = TRUE)
        pred_prob <- attr(pred_obj, "probabilities")[, "1"]
        auc <- as.numeric(suppressMessages(pROC::auc(y_vl, pred_prob)))
        auc_inner <- c(auc_inner, auc)
      }
      
      mean_auc <- mean(auc_inner)
      if (mean_auc > best_inner_auc) {
        best_inner_auc <- mean_auc
        best_C <- sv_C
        best_kernel <- sv_kernel
        best_gamma_str <- sv_gamma_str
      }
    }
    
    # Final model on trainval
    if (best_gamma_str == "scale") {
      best_gamma <- 1 / (n_selected * var(as.vector(X_trainval_sc)))
    } else {
      best_gamma <- 1 / n_selected
    }
    
    if (best_kernel == "linear") {
      final_model <- svm(x = X_trainval_sc, y = y_trainval, kernel = "linear",
                         cost = best_C, probability = TRUE)
    } else {
      final_model <- svm(x = X_trainval_sc, y = y_trainval, kernel = "radial",
                         cost = best_C, gamma = best_gamma, probability = TRUE)
    }
    
    pred_obj <- predict(final_model, X_test_sc, probability = TRUE)
    svm_prob <- attr(pred_obj, "probabilities")[, "1"]
    svm_pred <- factor(ifelse(svm_prob >= 0.5, 1, 0), levels = c("0", "1"))
    svm_metrics <- calc_metrics(y_test, svm_pred, svm_prob)
    
    results_svm <- rbind(results_svm, data.frame(
      Repeat = r, Outer_Fold = i, t(svm_metrics),
      N_Selected = n_selected,
      Best_C = best_C,
      Best_kernel = best_kernel,
      Best_gamma = best_gamma_str
    ))
    
    cat(sprintf("N=%d, C=%.1f, kernel=%s, gamma=%s, AUC=%.3f, Sens=%.3f, Spec=%.3f\n",
                n_selected, best_C, best_kernel, best_gamma_str,
                svm_metrics["AUC"], svm_metrics["Sens"], svm_metrics["Spec"]))
  }
}

# Summary
cat("\n-- CIF -> SVM (Tuned) --\n")
cat(sprintf("AUC:  %.3f ± %.3f\n", mean(results_svm$AUC), sd(results_svm$AUC)))
cat(sprintf("Acc:  %.3f ± %.3f\n", mean(results_svm$Acc), sd(results_svm$Acc)))
cat(sprintf("Sens: %.3f ± %.3f\n", mean(results_svm$Sens), sd(results_svm$Sens)))
cat(sprintf("Spec: %.3f ± %.3f\n", mean(results_svm$Spec), sd(results_svm$Spec)))
cat(sprintf("PPV:  %.3f ± %.3f\n", mean(results_svm$PPV), sd(results_svm$PPV)))
cat(sprintf("NPV:  %.3f ± %.3f\n", mean(results_svm$NPV), sd(results_svm$NPV)))
cat(sprintf("F1:   %.3f ± %.3f\n", mean(results_svm$F1), sd(results_svm$F1)))
cat(sprintf("|S-S|: %.3f\n", abs(mean(results_svm$Sens) - mean(results_svm$Spec))))

cat("\nBest params:\n")
cat("C:\n"); print(table(results_svm$Best_C))
cat("kernel:\n"); print(table(results_svm$Best_kernel))
cat("gamma:\n"); print(table(results_svm$Best_gamma))

write.csv(results_svm, "cif_SVM_tuned_results.csv", row.names = FALSE)
cat("\nSaved: cif_SVM_tuned_results.csv\n")