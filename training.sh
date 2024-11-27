#!/bin/bash --login
python -u train_elivagar_circuits.py --num_qubits 8 --save_dir q8_e16_p20_64 --size 64 --dataset PA --circs_dir ./experiments/q8_s16_p20 --device_name ibm_cleveland --encoding_type angle  --num_circs 10 --num_epochs 200 --batch_size 64 --learning_rate 0.01
