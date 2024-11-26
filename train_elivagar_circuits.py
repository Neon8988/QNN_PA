import argparse
import os
import numpy as np
import pennylane as qml

from elivagar.training.train_circuits_from_predictors import train_elivagar_circuits

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='which dataset to train the circuits on')
    parser.add_argument('--circ_prefix', default='circ', help='the common prefix for all the circuit folder names')
    parser.add_argument('--circs_dir', default='./', help='the folder where all the circuits are stored')
    parser.add_argument('--num_qubits', default=None, type=int, help='number of qubits used by the generated circuits')
    parser.add_argument('--num_meas_qubits', type=int, default=None, help='number of qubits to measure in each circuit')
    parser.add_argument('--num_epochs', default=300, type=int, help='number of epochs to train the generated circuits for')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size for training')
    parser.add_argument('--device_name', default=None, help='the device noise model the CNR scores were computed using')
    parser.add_argument('--num_runs_per_circ', default=5, help='number of training runs for each circuit')
    parser.add_argument('--encoding_type', default='angle', help='the encoding type to use for the data')
    parser.add_argument('--num_data_reps', default=None, help='the number of times to re-encode the data')
    parser.add_argument('--noise_importance', type=float, default=0.5, help='the relative importance of circuit CNR scores')
    parser.add_argument('--num_data_for_rep_cap', default=None, type=int, help='number of data points used to compute representational capacity')
    parser.add_argument('--num_params_for_rep_cap', default=None, type=int, help='number of parameter vectors used to compute representational capacity')
    parser.add_argument('--num_cdcs', default=None, type=int, help='number of cdcs used to compute CNR scores')
    parser.add_argument('--num_circs', default=None, type=int, help='number of circuits evaluated')
    parser.add_argument('--num_candidates_per_circ', default=5, type=int,help='number of candidate circuits to evaluate for every circuit trained')
    parser.add_argument('--save_dir', default=None, help='folder to save the generated circuits, trained models, and training results in')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')    
    parser.add_argument('--size',  default=16, help='the number of features ')
    parser.add_argument('--sub_encoder',  default=4, help='the number of sub_encoder ')
    args = parser.parse_args()
    
    if args.dataset is None:
        raise ValueError('Dataset cannot be None, please enter a valid dataset.')
     
    print('num_qubits:', args.num_qubits, 'num_meas_qubits:', args.num_meas_qubits, 'num_data_reps:', args.num_data_reps, 'size:', args.size)
    
    circ_list =[i for i in range(1,args.num_circs+1)]
    train_elivagar_circuits(args.save_dir,args.circs_dir, circ_list,args.dataset, args.encoding_type, 
                            args.num_data_reps,args.batch_size,args.num_qubits,  args.num_epochs, 
                            args.learning_rate,args.size,args.sub_encoder)
    
if __name__ == '__main__':
    main()
