dataset_circuit_hyperparams = dict()
gatesets = dict()

dataset_names = ['PA','pneumoniamnist']

circuit_params = [64,40]
circuit_embeds = [31,16]
circuit_qubits = [10,4]
num_data_reps = [1,1]
num_meas_qubits = [10,1]
num_embed_layers_angle_iqp = [1,4]
num_embed_layers_amp = [1 for i in range(2)]
num_var_layers = [1,15]
num_classes = [4,4]

for i in range(len(dataset_names)):
    dataset_circuit_hyperparams[dataset_names[i]] = dict()
    dataset_circuit_hyperparams[dataset_names[i]]['num_params'] = circuit_params[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embeds'] = circuit_embeds[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_qubits'] = circuit_qubits[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_data_reps'] = num_data_reps[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_meas_qubits'] = num_meas_qubits[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_var_layers'] = num_var_layers[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers'] = dict()
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['angle'] = num_embed_layers_angle_iqp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['iqp'] = num_embed_layers_angle_iqp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['amp'] = num_embed_layers_amp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_classes'] = num_classes[i]

gateset_names = ['rxyz_cz', 'rzx_rxx', 'ibm_basis', 'rigetti_aspen_m2_basis', 'oqc_lucy_basis']
gateset_gates = [[['rx', 'ry', 'rz'], ['cz']], [[], ['rzx', 'rxx']], [['rz', 'sx', 'x'], ['cx']], 
                [[], []], [[], []]]

gateset_param_nums = [[[1, 1, 1], [0]], [[], [1, 1]], [[1, 0, 0], [0]], 
                [[], []], [[], []]]

for i in range(len(gateset_names)):
    gatesets[gateset_names[i]] = (gateset_gates[i], gateset_param_nums[i])
