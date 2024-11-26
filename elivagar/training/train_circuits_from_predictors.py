
import torch
import numpy as np
import os
import pickle
import torch.nn as nn
from quantumnat.quantize import PACTActivationQuantizer
from elivagar.circuits.arbitrary import get_circ_params
from elivagar.circuits.create_circuit import TQCirc
from elivagar.training.train_tq_circ_and_save import train_tq_circ_and_save_results
from elivagar.metric_computation.compute_composite_scores import compute_composite_scores_for_circs
from torch.nn.functional import mse_loss
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch import Tensor
from elivagar.utils.datasets import load_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler




def train_elivagar_circuits(save_dir,circ_dir,circ_list, dataset, embed_type, num_data_reps,batch_size,num_qubits,num_epochs, 
                             learning_rate=0.01,size=16,sub_encoder=4,quantize=False,noise_strength=0.05, 
                                    tt_input_size=None,tt_ranks=None,  
                                   tt_output_size=None,use_test=True):
    """
    Compute the composite scores for all the circuits in a directory and train the top-ranked circuits.
    """

    x_train, y_train, x_test, y_test,scaler_y=load_dataset(dataset,embed_type, num_data_reps,size)
    train_data = torch.utils.data.TensorDataset(x_train,y_train)
    test_data = torch.utils.data.TensorDataset(x_test,y_test)
    creteria=nn.L1Loss()
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=y_test.shape[0], shuffle=False)
    maes=[]
    os.makedirs(save_dir, exist_ok=True)
    for i in circ_list:
        path_dir=os.path.join(circ_dir, f'circ_{i}')
        
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
        circ_gates, gate_params, inputs_bounds, weights_bounds,qubit_mapping = get_circ_params(path_dir)

        module_list=[TQCirc(
            circ_gates, gate_params, inputs_bounds, weights_bounds,
            num_qubits, False, quantize, noise_strength,
            tt_input_size, tt_ranks, tt_output_size).to(device) for i in range(sub_encoder)]
        input_size=num_qubits*sub_encoder
        model=HybridModel(module_list, input_size, 1).to(device)
        

        print(f'Number of parameters in the model: {sum(p.numel() for p in model.parameters())}')
        
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
   
        # quantizer = PACTActivationQuantizer(
        #     model, precision=4, alpha=1.0, backprop_alpha=False,
        #     device=device, lower_bound=-5, upper_bound=5
        # )
        
        # if quantize:
        #     model.set_noise_injection(True)
        
        losses = []
        best_loss = 1e+10
        for epoch in range(num_epochs):
            #quantizer.register_hook()
            model.train()
            for step, (x, y) in enumerate(train_data_loader):

                x = x.to(torch.float).to(device)
                y = y.to(device)

                out = model(x).reshape(-1,1)

                loss = creteria(out, y)
                opt.zero_grad()
                loss.backward()
                losses.append(loss.detach().item())

                opt.step()
                if loss < best_loss:
                    path=os.path.join(save_dir, f'circ_{i}.pt')
                    torch.save( model.state_dict(), path)
                    best_loss = loss
                losses.append(loss.detach().item()) 
            if epoch % 10 == 0:
                print(f'Epoch {epoch} | Loss: {loss.item()}') 
               
        if use_test:
            model.load_state_dict(torch.load(path))
            pred_tests=[]
            true_tests=[]
            trash_pred=[]
            trash_true=[]
            model.eval()
            with torch.no_grad():
                for x, y in test_data_loader:
                    x = x.to(torch.float).to(device)
                    y = y.to(device)
                    trash_true.extend(list(y.detach().numpy().reshape(-1)))
                    preds = model(x).reshape(-1,1)
                    loss = creteria(preds, y)
                    trash_pred.extend(list(preds.detach().numpy().reshape(-1)))
                    y_pred_re=scaler_y.inverse_transform(preds.detach().numpy())
                    y_true_re=scaler_y.inverse_transform(y.detach().numpy())
                    pred_tests.extend(list(y_pred_re.reshape(-1)))
                    true_tests.extend(list(y_true_re.reshape(-1)))
 
            mae=mean_absolute_error(true_tests,pred_tests)
            maes.append(mae)
            r2=r2_score(true_tests,pred_tests)
            rmse=root_mean_squared_error(true_tests,pred_tests)
            print(f'circuit {i},loss:{loss}| MAE: {mae} | R2: {r2} | RMSE: {rmse}')
    
    min_value = min(maes)
    print(f'Best MAE: {min_value}')





class HybridModel(nn.Module):
    def __init__(self, model_list,in_features, out_features):
        super(HybridModel, self).__init__()
        self.act = nn.Sigmoid()
        self.qcs = nn.ModuleList(model_list)
        self.fc1 = nn.Linear(in_features, in_features//2)
        self.fc2=nn.Linear(in_features//2, in_features//4)
        self.fc3 = nn.Linear(in_features//4, out_features)
        self.act_1=nn.LeakyReLU()
        
    def forward(self, x):
        part_size=x.shape[1]//len(self.qcs)
        sub_features = torch.split(x, part_size, dim=1) 
        x = [qc(sub_feature) for qc, sub_feature in zip(self.qcs, sub_features)]
        x = torch.cat(x, dim=1)
        x = self.act_1(self.fc1(x))
        x = self.act_1(self.fc2(x))
        output = self.act(self.fc3(x))
        return output