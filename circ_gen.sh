#!/bin/bash --login
python generate_device_aware_circuits.py --num_qubits 8 --num_params 20 --num_embeds 16 --num_meas_qubit 8 --target_dataset PA --encoding_type angle --num_circs 250 --device_name ibm_cleveland --save_dir ./experiments/q8_s16_p20 --temp 0.5 --add_rotations
