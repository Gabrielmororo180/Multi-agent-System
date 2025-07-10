import sys
import os
import csv

## importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer
from prediction_model import PredictionModel

#USE_MODEL = False
USE_MODEL = True 

def load_external_signals(filepath):
    print(f"Carregando dados de treinamento")
    signals = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                formatted_row = [
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    int(row[7])
                ]
                signals.append(formatted_row)
        return signals
    except Exception as e:
        print(f"ERRO: {e}")

def main(data_folder_name, config_ag_folder_name):
   
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
    
    # Instantiate the environment
    env = Env(data_folder)
    
    with open('file_target.txt', 'w', newline='') as f_target:
        # Usar o módulo csv é uma boa prática para evitar erros de formatação
        import csv
        writer = csv.writer(f_target)
        for signal in env.signals:
            # Formato: id, gravity_value, gravity_class
            # Extrai o ID (índice 0), valor de gravidade (índice 6) e classe (índice 7)
            vic_id = signal[0]
            gravity_value = signal[6]
            gravity_class = signal[7]
            writer.writerow([vic_id, f"{gravity_value:.2f}", gravity_class])
        
    model = PredictionModel.load_model("trained_model.pkl")

    # Instantiate master_rescuer
    # This agent unifies the maps and instantiate other 3 agents
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4, model, USE_MODEL)   # 4 is the number of explorer agents

    # Explorer needs to know rescuer to send the map 
    # that's why rescuer is instatiated before
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        Explorer(env, explorer_file, master_rescuer, exp)

    # Run the environment simulator
    env.run()
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        #data_folder_name = os.path.join("datasets", "data_42v_20x20")
        #data_folder_name = os.path.join("datasets", "data_132v_100x80")
        data_folder_name = os.path.join("datasets", "data_430v_94x94")
        #data_folder_name = os.path.join("datasets", "data_4000v")

        config_ag_folder_name = os.path.join("cfg_1")

        tlim = input("TLIM: ")
        for i in range(1, 5):
            with open(f"cfg_1/explorer_{i}_config.txt", 'r') as f:
                lines = f.readlines()
                for l, line in enumerate(lines):
                    #print(f"l: {l} - line: {line}")
                    if "TLIM" in line:
                        lines[l] = "TLIM " + tlim + "\n"
                        with open(f"cfg_1/explorer_{i}_config.txt", 'w') as f:
                            f.writelines(lines)
        
    main(data_folder_name, config_ag_folder_name)