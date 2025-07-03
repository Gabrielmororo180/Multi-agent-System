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
            if vic_id in model.test_ids:
                f_predict.write(f"{vic_id},{severity_value:.2f},{severity_class}\n")


if __name__ == "__main__":

    data_folder_name = os.path.join("datasets", "data_4000v")

    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    vitals_file = os.path.join(data_folder,"env_vital_signals.txt")

    # Parte 2 e 3: classificação e regressão
    test_size = input("Test size(0.25):")
    if test_size == '': test_size = 0.25
    else:   
        try:
            test_size = float(test_size)
        except ValueError:
            print("Valor invalido.")
            quit()

    signals = []
    with open(vitals_file, 'r') as f_vitals:
        for vitals in f_vitals:
            signal = [float(vital) for vital in vitals[:-1].split(',')]
            signal[0] = int(signal[0])
            signal[-1] = int(signal[-1])
            signals.append(signal)

    model = PredictionModel(test_size, signals)
    predict_severity_and_class(model, signals)