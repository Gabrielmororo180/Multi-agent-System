import os
import csv

from vs.environment import Env
from prediction_model import PredictionModel

IDX_GRAVITY = 6
IDX_SEVERITY = 7

def predict_severity_and_class(model, signals):
    """ @TODO to be replaced by a classifier and a regressor to calculate the class of severity and the severity values.
        This method should add the vital signals(vs) of the self.victims dictionary with these two values.

        This implementation assigns random values to both, severity value and class"""

    with open('file_predict.txt', 'w') as f_predict:

        for signal in signals:

            vic_id = signal[0]

            # Step 1: Extract features (first 5 vital signals: sp, dp, qp, pf, rf)
            features = signal

            # Step 2: Scale features using the model's scaler
            features_scaled = model.scaler.transform([features[1:6]])

            # Step 3: Predict severity value using the regressor
            severity_value = model.regressor.predict(features_scaled)[0]

            # Step 4: Predict severity class using the classifier
            severity_class = model.classifier.predict(features_scaled)[0]

            # Step 5: Write predictions to file for test set victims (if test_ids available)
            if model.test_ids is None or vic_id in model.test_ids:
                f_predict.write(f"{vic_id},{severity_value:.2f},{severity_class}\n")

def read_signals(vitals_file):
    signals = []
    with open(vitals_file, 'r') as f_vitals:
        for vitals in f_vitals:
            signal = [float(vital) for vital in vitals[:-1].split(',')]
            signal[0] = int(signal[0])
            signal[-1] = int(signal[-1])
            signals.append(signal)

    return signals


if __name__ == "__main__":

    folder_train = os.path.join("datasets", "data_4000v")
    folder_test = os.path.join("datasets", "data_800v")

    current_folder = os.path.abspath(os.getcwd())
    data_train = os.path.abspath(os.path.join(current_folder, folder_train))
    data_test = os.path.abspath(os.path.join(current_folder, folder_test))

    # Parte 2 e 3: classificação e regressão
    test_size = input("Test size(0.0):")
    if test_size == '': test_size = 0.0
    else:   
        try:
            test_size = float(test_size)
        except ValueError:
            print("Valor invalido.")
            quit()

    vitals_file = os.path.join(data_train, "env_vital_signals.txt")
    train_signals = read_signals(vitals_file)

    model = PredictionModel(test_size=test_size, signals=train_signals)

    vitals_file = os.path.join(data_test, "env_vital_signals.txt")
    test_signals = read_signals(vitals_file)

    # Generate file_target.txt from 800v dataset true values
    with open('file_target.txt', 'w') as f_target:
        for signal in test_signals:
            vic_id = signal[0]
            gravity = signal[6]  # Assuming gravity is at index 6
            severity_class = signal[7]  # Assuming severity class is at index 7
            f_target.write(f"{vic_id},{gravity:.2f},{severity_class}\n")

    predict_severity_and_class(model, test_signals)

    model.save_model("trained_model.pkl")