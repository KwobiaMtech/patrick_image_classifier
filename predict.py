from get_input_args import input_args
from helper_functions import load_checkpoint, get_flower_name
image_path, network_path, arch, cat_to_name, gpu_status, topk, lr, hidden_units, epochs = input_args()

model = load_checkpoint(network_path)


if model:
    flower_name, probable_names = get_flower_name(model, image_path, gpu_status, cat_to_name, topk)
    print("Flower name is {}".format(flower_name))
    print("\nBelow are probable names from the probable class")
    print(probable_names)
