import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

import pickle

import os
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from colorama import init, Fore
init()

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

        # Scale features
        self.scaler = StandardScaler()

        # Handle test_size=0 for training on the full dataset
        if test_size == 0:
            X_train = X
            X_train_scaled = self.scaler.fit_transform(X)
            y_gravity_train = y_gravity
            y_severity_train = y_severity
            train_ids = victim_ids
            X_test = None
            X_test_scaled = None
            #y_severity_test = None
            test_ids = None
        else:
            # Split into train and test sets
            X_train, X_test, y_gravity_train, y_gravity_test, y_severity_train, y_severity_test, train_ids, test_ids = train_test_split(
                X, y_gravity, y_severity, victim_ids, test_size=test_size, random_state=42
            )
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

        # Train models
        best_classifier = self.train_classifier(X_train_scaled, y_severity_train)
        best_regressor = self.train_regressor(X_train_scaled, y_gravity_train)

        self.test_ids = test_ids
        self.classifier = best_classifier
        self.regressor = best_regressor
        # Only write file_target.txt if test_ids exists (not for training-only case)
        if test_ids is not None:
            # Write true values for test set to file_target.txt
            with open('file_target.txt', 'w') as f_target:
                for vid, gr, sev in zip(test_ids, y_gravity_test, y_severity_test):
                    f_target.write(f"{vid},{gr:.2f},{sev}\n")


        # EXPLICABILIDADE
        if X_test_scaled is not None:

            # Feature names for interpretability
            feature_names = ['sp', 'dp', 'qp', 'pf', 'rf']

            # Select one instance per severity class (1, 2, 3, 4)
            selected_indices = []
            for cls in [1, 2, 3, 4]:
                idx = np.where(y_severity_test == cls)[0][0]  # Pick the first instance of each class
                selected_indices.append(idx)

            # Create SHAP explainers
            # Use a background dataset (summarized with k-means for efficiency)
            background = shap.kmeans(X_test_scaled, 11)  # clusters for faster computation
            explainer_clf = shap.KernelExplainer(self.classifier.predict_proba, background)
            explainer_reg = shap.KernelExplainer(self.regressor.predict, background)

            # Compute SHAP values for the selected instances
            shap_values_clf = explainer_clf.shap_values(X_test_scaled[selected_indices])
            shap_values_reg = explainer_reg.shap_values(X_test_scaled[selected_indices])

            # Ensure output directory exists
            os.makedirs('shap_plots', exist_ok=True)

            # Explain predictions for each instance
            for i, idx in enumerate(selected_indices):
                instance = X_test_scaled[idx]
                true_class = y_severity_test[idx]
                pred_class = self.classifier.predict([instance])[0]
                pred_gravity = self.regressor.predict([instance])[0]
                
                print(f"\nInstance {idx} (Victim ID: {int(signals[idx][0])}):")
                print(f"True Class: {true_class}, Predicted Class: {pred_class}")
                print(f"True Gravity: {y_gravity_test[idx]:.2f}, Predicted Gravity: {pred_gravity:.2f}")
                
                # Classifier explanation: Focus on the predicted class
                print(f"{Fore.YELLOW}Classifier Explanation (Top 3 Features):{Fore.RESET}")
                shap_values_for_pred_class = shap_values_clf[i][:, pred_class - 1]  # SHAP values for predicted class
                sorted_indices = np.argsort(np.abs(shap_values_for_pred_class))[::-1]  # Sort by absolute value
                for j in sorted_indices[:3]:  # Top 3 features
                    print(f"  {feature_names[j]}: SHAP value = {shap_values_for_pred_class[j]:.4f}")
                
                # Regressor explanation
                print(f"{Fore.YELLOW}Regressor Explanation (Top 3 Features):{Fore.RESET}")
                shap_values_reg_instance = shap_values_reg[i]
                sorted_indices_reg = np.argsort(np.abs(shap_values_reg_instance))[::-1]  # Sort by absolute value
                for j in sorted_indices_reg[:3]:  # Top 3 features
                    print(f"  {feature_names[j]}: SHAP value = {shap_values_reg_instance[j]:.4f}")

                # Generate and save SHAP force plot for classifier
                plt.figure(figsize=(10, 5))  # Increased figure size
                force_plot_clf = shap.force_plot(
                    explainer_clf.expected_value[pred_class - 1],
                    shap_values_for_pred_class,
                    feature_names=feature_names,
                    matplotlib=True
                )
                plt.title(f"SHAP Force Plot - Classifier (Instance {idx}, Class {pred_class})", pad=21)
                plt.tight_layout()  # Adjust layout to prevent cropping
                plt.savefig(f'shap_plots/force_clf_instance_{idx}.png', bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # Generate and save SHAP force plot for regressor
                force_plot_reg = shap.force_plot(
                    explainer_reg.expected_value,
                    shap_values_reg_instance,
                    feature_names=feature_names,
                    matplotlib=True
                )
                plt.title(f"SHAP Force Plot - Regressor (Instance {idx}, Gravity {pred_gravity:.2f})", pad=21)
                plt.tight_layout()
                plt.savefig(f'shap_plots/force_reg_instance_{idx}.png', bbox_inches='tight', pad_inches=0.1)
                plt.close()

            # Generate and save summary plot for all instances
            shap.summary_plot(shap_values_clf, X_test_scaled[selected_indices], feature_names=feature_names)
            plt.title("SHAP Summary Plot - Classifier (All Instances)", pad=21)
            plt.tight_layout()
            plt.savefig('shap_plots/summary_clf.png', bbox_inches='tight', pad_inches=0.1)
            plt.close()

            shap.summary_plot(shap_values_reg, X_test_scaled[selected_indices], feature_names=feature_names)
            plt.title("SHAP Summary Plot - Regressor (All Instances)", pad=21)
            plt.savefig('shap_plots/summary_reg.png', bbox_inches='tight', pad_inches=0.1)
            plt.close()



    def train_classifier(self, X_train, y_train):
        """Train and select the best classifier using GridSearchCV."""
        # DecisionTreeClassifier
        #dt_params = {
        #    'criterion': ['gini', 'entropy'],
        #    'max_depth': [2, 4, 8],
        #    'min_samples_leaf': [4, 8]
        #}

        #dt_params = {
        #    'criterion': ['gini', 'entropy'],
        #    'max_depth': [2, 4, 8],
        #    'min_samples_leaf': [8, 12]
        #}

        dt_params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 10],
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
            #'hidden_layer_sizes': [(111, 55), (100, 50), (50, 25)],
            'hidden_layer_sizes': [(10,), (50,), (30, 10)],
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


        print(f"DT_Classifier: {dt_clf.best_score_} - {dt_clf.best_estimator_}")
        print(f"MLP_Classifier: {mlp_clf.best_score_} - {mlp_clf.best_estimator_}")


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

        print(f"DT_Regressor: {dt_reg.best_score_} - {dt_reg.best_estimator_}")
        print(f"MLP_Regressor: {mlp_reg.best_score_} - {mlp_reg.best_estimator_}")

        # Select the best regressor (higher neg_mse is better)
        if dt_reg.best_score_ > mlp_reg.best_score_:
            return dt_reg.best_estimator_#, 'DecisionTreeRegressor'
        return mlp_reg.best_estimator_#, 'MLPRegressor'

    def save_model(self, filepath="trained_model.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath="trained_model.pkl"):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
