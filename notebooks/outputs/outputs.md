LogReg (SMOTE): 2,035 candidates x 5 folds = 10,175 fits
Fitting 5 folds for each of 2035 candidates, totalling 10175 fits
LogReg (SMOTE): done in 5.3 min | best CV F2 = 0.646
LogReg (SMOTE): best params = {'clf__C': 0.03, 'clf__class_weight': {0: 1, 1: 4}, 'clf__l1_ratio': 0.5, 'clf__penalty': 'elasticnet', 'smote': 'passthrough'}

RandomForest: 576 candidates x 5 folds = 2,880 fits
Fitting 5 folds for each of 576 candidates, totalling 2880 fits
RandomForest: done in 4.5 min | best CV F2 = 0.580
RandomForest: best params = {'class_weight': {0: 1, 1: 8}, 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 300}

XGBoost: 192 candidates x 5 folds = 960 fits
Fitting 5 folds for each of 192 candidates, totalling 960 fits
XGBoost: done in 11.6 min | best CV F2 = 0.573
XGBoost: best params = {'colsample_bytree': 0.8, 'gamma': 1, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 300, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': np.float64(5.189473684210526), 'subsample': 1.0}

LightGBM: 256 candidates x 5 folds = 1,280 fits
Fitting 5 folds for each of 256 candidates, totalling 1280 fits
LightGBM: done in 24.1 min | best CV F2 = 0.560
LightGBM: best params = {'colsample_bytree': 0.5, 'learning_rate': 0.05, 'max_depth': -1, 'min_child_samples': 10, 'n_estimators': 300, 'num_leaves': 15, 'reg_alpha': 1, 'reg_lambda': 5, 'scale_pos_weight': np.float64(5.189473684210526), 'subsample': 0.7}