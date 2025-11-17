# Generating_dataset.py 
from Feature_extraction import Feature_extraction
import time
import warnings
warnings.filterwarnings('ignore')
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Process
import numpy as np
import pandas as pd

def split_pcap(pcap_file, output_dir, size_mb):
    print(">>>> 1. Splitting the .pcap file.")
    try:
        subprocess.run([
            "tcpdump", "-r", pcap_file,
            "-w", os.path.join(output_dir, "split_temp"),
            "-C", str(size_mb)
        ], check=True)
    except FileNotFoundError:
        print("tcpdump n'est pas installé. Veuillez l'installer pour continuer.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de tcpdump : {e}")
        exit(1)

def pcap_eval_wrapper(subpcap_file, destination_path):
    fe = Feature_extraction()
    fe.pcap_evaluation(subpcap_file, destination_path)

def convert_subfiles_to_csv(subfiles, split_directory, destination_directory, n_threads):
    print(">>>> 2. Converting .pcap subfiles to .csv files.")
    subfiles_threadlist = np.array_split(subfiles, int(len(subfiles) / n_threads) + 1)
    
    for f_list in tqdm(subfiles_threadlist):
        n_processes = min(len(f_list), n_threads)
        processes = []
        for i in range(n_processes):
            f = f_list[i]
            subpcap_file = os.path.join(split_directory, f)
            destination_path = os.path.join(destination_directory, f.split('.')[0])
            p = Process(target=pcap_eval_wrapper, args=(subpcap_file, destination_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

def merge_csv_files(csv_directory, final_csv_path):
    print(">>>> 4. Merging .csv files.")
    csv_files = os.listdir(csv_directory)
    mode = 'w'
    for f in tqdm(csv_files):
        try:
            d = pd.read_csv(os.path.join(csv_directory, f))
            d.to_csv(final_csv_path, header=mode == 'w', index=False, mode=mode)
            mode = 'a'
        except Exception as e:
            print(f"Erreur dans la fusion de {f} : {e}")

def clean_directory(directory, extension=""):
    print(f">>>> Cleaning files in: {directory}")
    for f in tqdm(os.listdir(directory)):
        if extension and not f.endswith(extension):
            continue
        try:
            os.remove(os.path.join(directory, f))
        except Exception as e:
            print(f"Erreur lors de la suppression de {f} : {e}")

if __name__ == '__main__':
    start = time.time()
    print("========== CIC IoT feature extraction ==========")

   # base_dir = os.path.expanduser("~/Documents/Internship/pcap2csv")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.normpath(os.path.join(base_dir, "filename.pcap"))
    subfiles_size = 10  # MB
    split_directory = os.path.join(base_dir, "split_temp")
    destination_directory = os.path.join(base_dir, "output")
    final_csv_path = os.path.join(base_dir, "final_dataset.csv")
    n_threads = 8

    # Assure les dossiers existent
    for directory in [split_directory, destination_directory]:
        os.makedirs(directory, exist_ok=True)

    print(f"Lecture du fichier : {file}")
    split_pcap(file, split_directory, subfiles_size)

    subfiles = os.listdir(split_directory)
    convert_subfiles_to_csv(subfiles, split_directory, destination_directory, n_threads)

    # Vérification du résultat
    output_files = os.listdir(destination_directory)
    if len(subfiles) != len(output_files):
        print(f"Attention : {len(subfiles)} fichiers .pcap mais {len(output_files)} .csv générés.")
    else:
        print("Tous les fichiers convertis avec succès.")

    clean_directory(split_directory)
    merge_csv_files(destination_directory, final_csv_path)
    clean_directory(destination_directory)

    print(f"Terminé : {file} en {round(time.time() - start, 2)} secondes.")
