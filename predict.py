import json
import torch
import argparse
import numpy as np
from PIL import Image
from torch.autograd import Variable


def load_trained_model(path, device):
    '''
        Loads a trained model from a given path and set's it to run on the specified device (GPU or CPU)
        returns the trained model and data parameters
    '''
    model = torch.load(path)
    data_params = model.data_params
    model.to(device)
    return model, data_params

def process_image(image, data_params):
    '''
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 256, 256
    mean = data_params['mean']
    std = data_params['std']
    width, height = data_params['im_size'], data_params['im_size']
    top = size[0] // 2 - width//2
    left = size[1] //2 - height//2
    
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((left, top, left + width, top + height))
    image_np = np.array(image) / 255
    for i in range(3):
        image_np[:,:,i] = (image_np[:,:,i] - mean[i]) / std[i]
    return np.transpose(image_np, (2,0,1))

def predict(image_path, model, data_params, device, topk=5):
    '''
        Predict the class (or classes) of an image using a trained deep learning model.
        Returns the topk probabilities and respective classes
    '''
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    image = Image.open(image_path)
    image = process_image(image, data_params)
    image = Variable(torch.FloatTensor([image])).to(device)
    model.eval()
    output = model.forward(image)
    probs = torch.exp(output).data.cpu().numpy()[0]
    topk_idx = np.argsort(probs)[-topk:][::-1]
    topk_class = [idx_to_class[x] for x in topk_idx]
    topk_probs = probs[topk_idx]
    return topk_probs, topk_class

def get_device(enable_gpu):
    '''
        Gets the GPU or CPU device
        Returns torch.device
    '''
    if enable_gpu:
        if torch.cuda.is_available():
            print("using GPU\n")
            return torch.device("cuda:0")
        else:
            print("no GPU found, using CPU\n")
    else:
        print("using CPU\n")
    return torch.device("cpu")

def load_label_map(path_to_map):
    '''
        Loads a json file containing labels for integer values
        Returns the dictionary
    '''
    with open('cat_to_name.json', 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Train a model on the data.')

    parser.add_argument('path_to_model', type=str, help='path to the model')
    parser.add_argument('path_to_image', type=str, help='path to image for classification')
    parser.add_argument("--gpu", help="enable gpu for prediction", action="store_true")
    parser.add_argument("--topk", type=int, help="print top k prediction", default=1)
    parser.add_argument("--label_map", type=str, help="json file containing mapping for integer label to string labels", default='cat_to_name.json')
    
    args = parser.parse_args()
    device = get_device(args.gpu)
    topk = args.topk
    if topk <= 0:
        print('topk should be a positive number')
        return

    # load model and parameters
    model, data_params = load_trained_model(args.path_to_model, device)
    path_to_image = args.path_to_image
    cat_to_name = load_label_map(args.label_map)
    
    # Predict probabilites and classes
    probs, classes = predict(path_to_image, model, data_params, device, topk)
    labels = [cat_to_name[x] for x in classes]
    # Print best prediction
    print('Best prediction- {} with probability {}\n'.format(probs[0], labels[0]))
    # Print top K prediction
    print('Top {} prediction-'.format(topk))
    for prob, flower in zip(probs, labels):
        print('{} - {}'.format(flower, prob))

if __name__ == "__main__":
    main()
