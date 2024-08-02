import subprocess
import os
import pickle as pkl
# Fisher: 11
data = 'Fisher'
dim = 11
data_folder = "./Datasets/Fisher"

base_args = ("--freq 'm' --is_training 1 --enc_in " + str(dim) +
            " --dec_in " + str(dim) + " --c_out " + str(dim) + 
            " --model_id 'test' --model '{model}' --data '{data}'" +
            " --features 'M' --max_level 3 --root_path " + data_folder)

seq_lengths = [72,144,288]
models = ['harmony']

csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    data_path = csv_file
    for seq_len in seq_lengths:
        for model in models:
            print(f"Testing {csv_file} with model {model} and seq_len {seq_len}")
            args = base_args.format(model=model, data=data) + f" --seq_len {seq_len} --data_path '{data_path}'"
            print(args)
            command = f"python run.py {args}"
            subprocess.run(command, shell=True)
print("End Testing")
