# CIF Feature Selection + Nested CV
# Structure: 7 Repeats x 10 Outer x 5 Inner
# Validation split included

library(party)
library(readxl)
library(caret)
library(pROC)

# Load data
df <- read_excel("final_ov.xlsx")

y <- factor(ifelse(df$`2 group class` == "Cancer", 1, 0), levels = c("0", "1"))
feature_cols <- grep("^hsa-", names(df), value = TRUE)
X <- df[, feature_cols]
data_full <- data.frame(y = y, X)
names(data_full) <- make.names(names(data_full))

# Settings
n_repeats <- 7
n_outer <- 10
n_inner <- 5
mtry_grid <- c(30, 50, 100)

# Storage
results <- data.frame()
all_selected_features <- list()

set.seed(42)

for (r in 1:n_repeats) {
  cat(sprintf("\n-- Repeat %d/%d --\n", r, n_repeats))
  
  outer_folds <- createFolds(data_full$y, k = n_outer, list = TRUE)
  
  for (i in 1:n_outer) {
    cat(sprintf("Outer fold %d/%d ... ", i, n_outer))
    
    # Outer split: trainval (90%) / test (10%)
    test_idx <- outer_folds[[i]]
    trainval_idx <- setdiff(1:nrow(data_full), test_idx)
    
    trainval_data <- data_full[trainval_idx, ]
    test_data <- data_full[test_idx, ]
    
    # Split trainval: train (80%) / val (20%)
    set.seed(42 + r * 100 + i)
    val_split <- createDataPartition(trainval_data$y, p = 0.2, list = FALSE)
    val_data <- trainval_data[val_split, ]
    train_data <- trainval_data[-val_split, ]
    
    # Inner CV for mtry tuning (on train only)
    best_mtry <- mtry_grid[1]
    best_auc <- 0
    
    inner_folds <- createFolds(train_data$y, k = n_inner, list = TRUE)
    
    for (mtry in mtry_grid) {
      auc_inner <- c()
      
      for (j in 1:n_inner) {
        inner_val_idx <- inner_folds[[j]]
        train_inner <- train_data[-inner_val_idx, ]
        val_inner <- train_data[inner_val_idx, ]
        
        # CIF model
        model <- cforest(
          y ~ ., 
          data = train_inner,
          controls = cforest_unbiased(ntree = 500, mtry = mtry)
        )
        
        # Predict
        pred_prob <- treeresponse(model, newdata = val_inner)
        pred_prob <- sapply(pred_prob, function(x) x[2])
        
        auc <- as.numeric(pROC::auc(val_inner$y, pred_prob))
        auc_inner <- c(auc_inner, auc)
      }
      
      mean_auc <- mean(auc_inner)
      if (mean_auc > best_auc) {
        best_auc <- mean_auc
        best_mtry <- mtry
      }
    }
    
    # CIF on train data with best mtry
    cif_train <- cforest(
      y ~ ., 
      data = train_data,
      controls = cforest_unbiased(ntree = 500, mtry = best_mtry)
    )
    
    # Get variable importance
    varimp_cif <- varimp(cif_train, conditional = FALSE)
    
    # Select features with importance > 0
    selected_features <- names(varimp_cif[varimp_cif > 0])
    n_selected <- length(selected_features)
    
    if (n_selected == 0) {
      selected_features <- names(sort(varimp_cif, decreasing = TRUE)[1:50])
      n_selected <- 50
      cat("WARNING: No features > 0, using top 50. ")
    }
    
    # Store selected features
    all_selected_features[[length(all_selected_features) + 1]] <- selected_features
    
    # Subset data to selected features
    train_sel <- train_data[, c("y", selected_features)]
    val_sel <- val_data[, c("y", selected_features)]
    trainval_sel <- trainval_data[, c("y", selected_features)]
    test_sel <- test_data[, c("y", selected_features)]
    
    # Final CIF model on trainval
    final_model <- cforest(
      y ~ ., 
      data = trainval_sel,
      controls = cforest_unbiased(ntree = 500, mtry = min(best_mtry, n_selected))
    )
    
    # Predict on test
    pred_prob <- treeresponse(final_model, newdata = test_sel)
    pred_prob <- sapply(pred_prob, function(x) x[2])
    pred_class <- ifelse(pred_prob >= 0.5, "1", "0")
    
    # Metrics
    actual <- test_sel$y
    tp <- sum(pred_class == "1" & actual == "1")
    tn <- sum(pred_class == "0" & actual == "0")
    fp <- sum(pred_class == "1" & actual == "0")
    fn <- sum(pred_class == "0" & actual == "1")
    
    auc <- as.numeric(pROC::auc(actual, pred_prob))
    acc <- (tp + tn) / (tp + tn + fp + fn)
    sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    spec <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
    ppv <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
    npv <- ifelse((tn + fn) > 0, tn / (tn + fn), 0)
    f1 <- ifelse((ppv + sens) > 0, 2 * ppv * sens / (ppv + sens), 0)
    
    results <- rbind(results, data.frame(
      Repeat = r,
      Outer_Fold = i,
      AUC = auc,
      Acc = acc,
      Sens = sens,
      Spec = spec,
      PPV = ppv,
      NPV = npv,
      F1 = f1,
      N_Selected = n_selected,
      Best_mtry = best_mtry
    ))
    
    cat(sprintf("N=%d, mtry=%d, AUC=%.3f, Sens=%.3f, Spec=%.3f\n", 
                n_selected, best_mtry, auc, sens, spec))
  }
}

# Summary
cat("\n-- Final Results --\n")
cat(sprintf("AUC: %.3f ± %.3f\n", mean(results$AUC), sd(results$AUC)))
cat(sprintf("Acc: %.3f ± %.3f\n", mean(results$Acc), sd(results$Acc)))
cat(sprintf("Sens: %.3f ± %.3f\n", mean(results$Sens), sd(results$Sens)))
cat(sprintf("Spec: %.3f ± %.3f\n", mean(results$Spec), sd(results$Spec)))
cat(sprintf("PPV: %.3f ± %.3f\n", mean(results$PPV), sd(results$PPV)))
cat(sprintf("NPV: %.3f ± %.3f\n", mean(results$NPV), sd(results$NPV)))
cat(sprintf("F1: %.3f ± %.3f\n", mean(results$F1), sd(results$F1)))
cat(sprintf("N_Selected: %.1f ± %.1f\n", mean(results$N_Selected), sd(results$N_Selected)))
cat(sprintf("|Sens - Spec|: %.3f\n", abs(mean(results$Sens) - mean(results$Spec))))

# Save
write.csv(results, "cif_nested_cv_results.csv", row.names = FALSE)
cat("\nSaved: cif_nested_cv_results.csv\n")
