import argparse
import os
import numpy as np
import pickle as pkl

from qiskit_ibm_runtime import QiskitRuntimeService
from elivagar.circuits.device_aware import generate_device_aware_gate_circ

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='mnist_2', help='which dataset the circuits will be used with')
    parser.add_argument('--num_qubits', type=int, default=None, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_circs', type=int, default=2500, help='the number of circuits to generate')
    parser.add_argument('--num_embeds', type=int, default=None, help='the number of embedding gates in each circuit')
    parser.add_argument('--num_params', type=int, default=None, help='the number of parameters in each circuit')
    parser.add_argument('--device_name', type=str, default='ibm_brisbane', help='the name of the IBM device the circuits will be run on')
    parser.add_argument('--param_focus', type=float, default=2, help='the amount of preference given to parametrized gates over nonparametrized ones')
    parser.add_argument('--save_dir', type=str, default='./', help='the folder to save the generated circuits in')
    parser.add_argument('--add_rotations', action='store_true', help='whether to add RX / RY / RZ gates to the basis gate list or not')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='the number of qubits to be measured')
    parser.add_argument('--num_trial_mappings', type=int, default=100, help='the number of trial qubit mappings to consider')
    parser.add_argument('--temp', type=float, default=0.1, help='the temperature of the softmax distributions to sample from')
    parser.add_argument('--braket_device_properties_path', type=str, default=None, help='the path to the brakte device properties pkl file')
    parser.add_argument('--encoding_type', default=None, help='the encoding type to use for the data')
    args = parser.parse_args()
    
    if args.target_dataset is None and (args.num_embeds is None or args.num_params is None or args.num_qubits is None):
        raise ValueError('Either provide the target dataset name or the number of qubits, parameters / embeds in each circuit.')
        
    if args.device_name is None and args.braket_device_properties_path is None:
        raise ValueError('Device name cannot be None, please enter a valid device name.')
     
    print('Generating circuits with the following hyperparameters:') 
    print(f'num_qubits: {args.num_qubits}, num_params: {args.num_params}, num_embeds: {args.num_embeds}, num_meas_qubits: {args.num_meas_qubits}')
    if args.braket_device_properties_path is None:
        service = QiskitRuntimeService()
        backend = service.backend(args.device_name)
        dev_properties = None
    
    for i in range(args.num_circs):
        curr_circ_dir = os.path.join(args.save_dir, f'circ_{i + 1}')
        
        if not os.path.exists(curr_circ_dir):
            os.makedirs(curr_circ_dir)
            
        ent_prob = np.random.sample()
        
        circ_gates, gate_params, inputs_bounds, weights_bounds, selected_mapping, meas_qubits = generate_device_aware_gate_circ(
            backend, args.num_qubits, args.num_embeds, args.num_params, ent_prob, args.add_rotations, args.param_focus,
            args.num_meas_qubits, args.num_trial_mappings, args.temp, braket_device_properties=dev_properties
        )
    
        results = {'gates': circ_gates, 'gate_params': gate_params, 
                'inputs_bounds': inputs_bounds, 'weights_bounds': weights_bounds,
                'qubit_mapping': selected_mapping, 'meas_qubits': meas_qubits}

        # Save to a pickle file
        with open(f"{curr_circ_dir}/result.pkl", "wb") as f:
            pkl.dump(results, f)
    
if __name__ == '__main__':
    main()
