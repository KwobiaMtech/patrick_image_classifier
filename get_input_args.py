import argparse


def input_args():
    parser = argparse.ArgumentParser(description='check images from cli input taker.')
    parser.add_argument('--image', type=str, default='flowers/test/25/image_06580.jpg')
    parser.add_argument('--network_path', type=str, default='checkpoint.pth')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--arch', type=str, default='alexnet')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', type=str, default=False)
    response = parser.parse_args()

    image_path = response.image
    network_path = response.network_path
    arch = response.arch
    cat_to_name = response.cat_to_name
    gpu_status = response.gpu
    topk = response.topk
    lr = response.lr
    hidden_units = response.hidden_units
    epochs = response.epochs

    return image_path, network_path, arch, cat_to_name, gpu_status, topk,lr,hidden_units,epochs

