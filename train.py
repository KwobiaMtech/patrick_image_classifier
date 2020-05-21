from get_input_args import input_args
from helper_functions import network, train_network
image_path, network_path, arch, cat_to_name, gpu_status, topk, lr, hidden_units, epochs = input_args()


model, criterion, optimizer = network(arch, lr, hidden_units)

train_network(model, optimizer, criterion, gpu_status, arch, epochs)
