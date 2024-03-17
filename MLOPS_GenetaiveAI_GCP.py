import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from scoring import *
import torch
from collections import defaultdict
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

config_dict = nested_defaultdict()



def _get_network_type(weight_dict):
    if any(x.startswith('s_layer_') for x in weight_dict):
        if any(x.startswith('t_layer_') for x in weight_dict):
            network = 'transfer'
        else:
            network = 'multitask'
    else:
        network = 'baseline'
    return network


def _get_layer_key(layer):
    ordinal = 0 if layer.startswith('s_layer') else 2 if 'output' in layer else 1
    try:
        num = int(layer.split('_')[-1])
    except ValueError:
        num = 0
    return ordinal, num


def _get_layers(weight_dict, network, robot_name):
    if network == 'multitask':
        keys = (k for k in weight_dict if (k.startswith('s_layer_') or k.startswith(robot_name)))
    else:
        keys = weight_dict
    return list(sorted((x.split('/')[0] for x in keys), key=_get_layer_key))


def _relu(x):
    x[x < 0] = 0
    return x


def _linear(x):
    return x


def calc_layer_outputs(weight_dict, bias_dict, inp, robot_name, network=None):
    if network is None:
        network = _get_network_type(weight_dict)
    layers = _get_layers(weight_dict, network, robot_name)
    print(layers)
    wkeys = [l + '/weights' for l in layers]
    bkeys = [l + '/bias' for l in layers]
    activation = [_linear if 'output' in l else _relu for l in layers]
    x = inp
    results = dict()
    for l, w, b, sigma in zip(layers, wkeys, bkeys, activation):
        w = weight_dict[w]
        b = bias_dict[b]
        x = sigma(np.matmul(x, w) + b)
        results[l] = x
    return results


def process_network(data):
    weight_dict = data[0]
    network = _get_network_type(weight_dict)
    bias_dict = data[1]
    robots = list(filter(lambda x: x != 'input', data[2]))

    results = dict()
    for robot in robots:
        if network == 'multitask':
            inp = data[2]['input'][robot]
        else:
            inp = data[2][robot]['input']
        results[robot] = calc_layer_outputs(weight_dict, bias_dict, inp, robot, network)
    return results


# The Metric
def procrustes(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    """
    A_sq_frob = np.sum(A ** 2)
    B_sq_frob = np.sum(B ** 2)
    nuc = np.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc

# The Metric
def procrustes_Torch(A, B):
    """
    Computes Procrustes distance between representations A and B
    """
    A_sq_frob = torch.sum(A ** 2)
    B_sq_frob = torch.sum(B ** 2)
    nuc = torch.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc
# Make sure you have a GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

def lin_cka_dist(A, B):
    """
    Computes Linear CKA distance bewteen representations A and B
    """
    similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = np.linalg.norm(A @ A.T, ord="fro") * np.linalg.norm(
        B @ B.T, ord="fro"
    )
    return 1 - similarity / normalization

# Compute mutual information
def compute_mutual_information(mat_A, mat_B):
    flat_A = mat_A.flatten()
    flat_B = mat_B.flatten()
    mi = mutual_info_score(flat_A, flat_B)
    return mi

# Compute normalized mutual information
def compute_normalized_mutual_information(mat_A, mat_B):
    flat_A = mat_A.flatten()
    flat_B = mat_B.flatten()
    nmi = normalized_mutual_info_score(flat_A, flat_B, average_method="arithmetic")
    return nmi

from scipy.stats import entropy


num_bins=30

def compute_joint_histogram(mat_A, mat_B, num_bins):
    hist, _, _ = np.histogram2d(mat_A.flatten(), mat_B.flatten(), bins=num_bins)
    return hist

def compute_normalized_mutual_information_large(mat_A, mat_B, num_bins=30):
    joint_hist = compute_joint_histogram(mat_A, mat_B, num_bins)
    joint_prob = joint_hist / np.sum(joint_hist)
    
    marginal_prob_A = np.sum(joint_prob, axis=1)
    marginal_prob_B = np.sum(joint_prob, axis=0)

    ent_A = entropy(marginal_prob_A, base=2)
    ent_B = entropy(marginal_prob_B, base=2)
    
    joint_ent = entropy(joint_prob.flatten(), base=2)
    
    mi = ent_A + ent_B - joint_ent
    nmi = mi / np.sqrt(ent_A * ent_B)
    return nmi

def Normal(A):
        # Subtract the mean value from each column
    mean_column = np.mean(A, axis=0)
    A_centered = A - mean_column

    # Compute the Frobenius norm
    frobenius_norm = np.linalg.norm(A_centered, 'fro')

    # Divide by the Frobenius norm to obtain the normalized representation A*
    A_normalized = A_centered / frobenius_norm
    return A_normalized

def Normal_Torch(A):
    # Subtract the mean value from each column
    mean_column = torch.mean(A, axis=0)
    A_centered = A - mean_column

    # Compute the Frobenius norm
    frobenius_norm = torch.linalg.norm(A_centered, 'fro')

    # Divide by the Frobenius norm to obtain the normalized representation A*
    A_normalized = A_centered / frobenius_norm
    return A_normalized
# implement it on the dictionary

with open('F_dict.pkl', 'rb') as f:
     PA_dict = pickle.load(f)


# A_all = process_network(PA_dict['All']['Fixed_BSW_S1'][0])
# B_all = process_network(PA_dict['All']['Down_W_S0'][0])
# C_all = process_network(PA_dict['All']['Down_BSWtoK_S1T1'][0]) 

# plot 

# Keys_A = list(A_all['All'].keys())
# Keys_B = list(B_all['All'].keys())
Keys_All = list(PA_dict['All'].keys())

d_list_ditance_procrustes = []
d_list_ditance_CKA = []
config_dict = nested_defaultdict()
for Index1 in range(len(Keys_All)):
    for Index2 in range(len(Keys_All)):
        if 'Fixed' in Keys_All[Index1] and 'Fixed' in Keys_All[Index2]:# and 'S0' in Keys_All[Index1] and 'S0' in Keys_All[Index2]:
            print("config1=", Keys_All[Index1])
            print("config2=", Keys_All[Index2])
            # for keys1 in PA_dict['All'][Keys_All[Index1]]:
            #     for keys2 in PA_dict['All'][Keys_All[Index2]]:

                        # Keys_A_Output = list(A_all.keys())
                        # Keys_B_Output = list(B_all.keys())
                        # for j in 
            # print("Index=", PA_dict['All'][Keys_All[Index1]][0].keys())
            A = process_network(PA_dict['All'][Keys_All[Index1]][0])
            B = process_network(PA_dict['All'][Keys_All[Index2]][0])
            for keys1 in A.keys():
                for keys2 in B.keys():
                    for Layer in range(2):
                        A_Layer_Keys_list = list(A[keys1].keys())
                        B_Layer_Keys_list = list(B[keys2].keys())

                        ditance_procrustes= procrustes(Normal(A[keys1][A_Layer_Keys_list[Layer]]), Normal(B[keys2][B_Layer_Keys_list[Layer]]))
                        ditance_procrustes = compute_normalized_mutual_information((A[keys1][A_Layer_Keys_list[Layer]]), (B[keys2][B_Layer_Keys_list[Layer]]))
                        ditance_procrustes = compute_normalized_mutual_information_large((A[keys1][A_Layer_Keys_list[Layer]]), (B[keys2][B_Layer_Keys_list[Layer]]))

                        # ditance = compute_mutual_information((A[keys1][A_Layer_Keys_list[Layer]]), (B[keys2][B_Layer_Keys_list[Layer]]))
                        # A_torch = torch.tensor(A[keys1][A_Layer_Keys_list[Layer]], device=device, dtype=torch.float)
                        # B_torch = torch.tensor(B[keys2][B_Layer_Keys_list[Layer]], device=device, dtype=torch.float)

                        # A_torch_N = torch.tensor(A[keys1][A_Layer_Keys_list[Layer]], device=device, dtype=torch.float)
                        # B_torch_N = torch.tensor(B[keys2][B_Layer_Keys_list[Layer]], device=device, dtype=torch.float)

                        

                        # ditance = procrustes_Torch(Normal_Torch(A_torch), Normal_Torch(B_torch))
                        # distance_numpy = ditance.cpu().numpy()

                        # ditance_CKA= lin_cka_dist(Normal(A[keys1][A_Layer_Keys_list[Layer]]), Normal(B[keys2][B_Layer_Keys_list[Layer]]))
                        
                        d_list_ditance_procrustes.append(ditance_procrustes)
                        # d_list_ditance_CKA.append(ditance_CKA)
                        # [Keys_All[Index1] A.keys() # A_Layer_Keys_list 

                        config = Keys_All[Index1] + keys1 + ' ' +  'to' + ' ' + Keys_All[Index2] + keys2
                        config_dict[Keys_All[Index1]][keys1][Keys_All[Index2]][keys2][Layer]['procrustes'] = ditance_procrustes
                        # config_dict[Keys_All[Index1]][keys1][Keys_All[Index2]][keys2][Layer]['CKA'] = ditance_CKA  
                        # markers = 'o' if 'B' in path0 else '^' if 'S' in path0 else '<' if 'K' in path0 else '.' if 'W' in path0 else '>'  
                        plt.scatter(Layer+1, ditance_procrustes, label=config, marker='o')
                        # plt.scatter(Layer+1, ditance_CKA, label=config, marker='+')
                    
print("ditance_procrustes=", ditance_procrustes)
# print("ditance_CKA=", ditance_CKA)
# save the dict
# with open('config_dict.pkl', 'wb') as file:
#     pickle.dump(config_dict, file)
# Plot Histogram

#plt.hist(d_list, bins=5, edgecolor='black')

# Plot scatter plot

# Add labels and a title
plt.xlabel(' Layer ')
plt.ylabel('Distance Values')
# plt.legend(bbox_to_anchor=(1.1, 1.16), loc='upper right', fontsize=5)
# plt.title('Histogram')
# Save the plot to a file
plt.savefig('Distance_All_Layers1ls_Metrics.png')
# Display the histogram
plt.show()

# plt.scatter(list(range(1,5)), d_list)
# plt.xlabel('Layers')
# plt.ylabel('procrustes distance')
# plt.title('Comparing the distance for each layer for Down_BKStoW_S1T1 and Down_W_S0')

# plt.show()

# B=
