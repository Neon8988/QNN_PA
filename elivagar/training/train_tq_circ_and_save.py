import pickle as pkl
import torch
import os
import numpy as np
from quantumnat.quantize import PACTActivationQuantizer

from elivagar.circuits.arbitrary import get_circ_params, generate_true_random_gate_circ
from elivagar.circuits.create_circuit import TQCirc
from elivagar.training.train_circ_np import train_tq_model, TQMseLoss

def train_tq_circ_and_save_results(circ_dir, train_data_loader,num_qubits, 
                                    epochs,quantize=False,noise_strength=0.05, 
                                    tt_input_size=None,tt_ranks=None,  
                                   tt_output_size=None, learning_rate=0.01,
                                    ):
    """
    Train the TQ circuit in the directory passed in, and save the trained loss and accuracy values, as well as the trained model(s).
    """
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    circ_gates, gate_params, inputs_bounds, weights_bounds,qubit_mapping = get_circ_params(circ_dir)

    model = TQCirc(
        circ_gates, gate_params, inputs_bounds, weights_bounds,
        num_qubits, False, quantize, noise_strength,
        tt_input_size, tt_ranks, tt_output_size).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = lambda x, y: ((x - y) ** 2).mean()

        
    quantizer = PACTActivationQuantizer(
        model, precision=4, alpha=1.0, backprop_alpha=False,
        device=device, lower_bound=-5, upper_bound=5
    )
    
    if quantize:
        model.set_noise_injection(True)
    
    losses = []
    best_loss = 1e+10
    for epoch in range(epochs):
        quantizer.register_hook()
        for step, (x, y) in enumerate(train_data_loader):
            x = x.to(torch.float).to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            losses.append(loss.detach().item())

            opt.step()
            if loss < best_loss:
                path=os.path.join(circ_dir, 'best.pt')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, path)
                best_loss = loss
            losses.append(loss.detach().item()) 
            print(f'Epoch {epoch + 1} | Loss: {loss.item()}') 
              
        quantizer.remove_hook()

    model.set_noise_injection(False)
    return min(losses)