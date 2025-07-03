import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

class PredictionModel:

    def __init__(self, test_size=0.25, signals=[]):
        # Extract features and targets from env.signals
        features = []
        gravity_targets = []  # Severity value (regression)
        severity_targets = []  # Severity class (classification)
        victim_ids = []
        for signal in signals:
            vid, sp, dp, qp, pf, rf, gr, lb = signal
            features.append([sp, dp, qp, pf, rf])
            gravity_targets.append(gr)
            severity_targets.append(lb)
            victim_ids.append(vid)

        # Convert to numpy arrays
        X = np.array(features)
        y_gravity = np.array(gravity_targets)
        y_severity = np.array(severity_targets)

        # Split into train and test sets
        X_train, X_test, y_gravity_train, y_gravity_test, y_severity_train, y_severity_test, train_ids, test_ids = train_test_split(
            X, y_gravity, y_severity, victim_ids, test_size=test_size, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train models
        best_classifier = self.train_classifier(X_train_scaled, y_severity_train)
        best_regressor = self.train_regressor(X_train_scaled, y_gravity_train)

        self.test_ids = test_ids
        self.classifier = best_classifier
        self.regressor = best_regressor

        # Write true values for test set to file_target.txt
        with open('file_target.txt', 'w') as f_target:
            for vid, gr, sev in zip(test_ids, y_gravity_test, y_severity_test):
                f_target.write(f"{vid},{gr:.2f},{sev}\n")

    def train_classifier(self, X_train, y_train):
        """Train and select the best classifier using GridSearchCV."""
        # DecisionTreeClassifier
        #dt_params = {
        #    'criterion': ['gini', 'entropy'],
        #    'max_depth': [2, 4, 8],
        #    'min_samples_leaf': [4, 8]
        #}

        dt_params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 4, 8],
            'min_samples_leaf': [8, 12]
        }

        dt_clf = GridSearchCV(
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            dt_params,
            cv=3,
            scoring='f1_weighted'
        )
        dt_clf.fit(X_train, y_train)

        # MLPClassifier
        mlp_params = {
            'hidden_layer_sizes': [(111, 55), (100, 50), (50, 25)],
            'alpha': [0.01, 0.001],
            'learning_rate': ['adaptive', 'constant']
        }
        mlp_clf = GridSearchCV(
            MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1, max_iter=2222),
            mlp_params,
            cv=3,
            scoring='f1_weighted'
        )
        mlp_clf.fit(X_train, y_train)

        # Select the best classifier
        if dt_clf.best_score_ > mlp_clf.best_score_:
            return dt_clf.best_estimator_#, 'DecisionTreeClassifier'
        return mlp_clf.best_estimator_#, 'MLPClassifier'

    def train_regressor(self, X_train, y_train):
        """Train and select the best regressor using GridSearchCV."""
        # DecisionTreeRegressor
        dt_params = {
            'max_depth': [2, 4, 8, 16],
            'min_samples_leaf': [0.01, 0.1, 0.25]
        }
        dt_reg = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            dt_params,
            cv=3,
            scoring='neg_mean_squared_error'
        )
        dt_reg.fit(X_train, y_train)

        # MLPRegressor
        mlp_params = {
            'hidden_layer_sizes': [(100, 55), (50, 25)],
            'alpha': [0.01, 0.001],
            'learning_rate': ['adaptive', 'constant']
        }
        mlp_reg = GridSearchCV(
            MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1, max_iter=2222),
            mlp_params,
            cv=3,
            scoring='neg_mean_squared_error'
        )
        mlp_reg.fit(X_train, y_train)

        # Select the best regressor (higher neg_mse is better)
        if dt_reg.best_score_ > mlp_reg.best_score_:
            return dt_reg.best_estimator_#, 'DecisionTreeRegressor'
        return mlp_reg.best_estimator_#, 'MLPRegressor'

