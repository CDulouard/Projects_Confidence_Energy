#   conda activate doctr_tf
#   cd ~/Desktop/github/doctr/
#   cp ../../dcotr-documents/step-v0.5.B/recognition_inference_possiblities_tf.py .    
#   python3.8 recognition_inference_possiblities_tf.py --arch crnn_vgg16_bn --dataset FUNSD -j 7 -b 12 

import os

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import multiprocessing as mp
import time
import numpy as np

from copy import deepcopy

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr import datasets
from doctr import transforms as T
from doctr.datasets import VOCABS, DataLoader
from doctr.models import recognition
from doctr.utils.metrics import TextMatch



class SortedQueue:
    def __init__(self,max_length=32):
        self.max_length = max_length
        self.liste_value = []
        self.liste_elements = []
        self.index = -1

    def push(self, value, elements):
        i = 0
        while i<len(self.liste_value):
            if self.liste_value[i]>=value:
                self.liste_value.insert(i,value)
                self.liste_elements.insert(i,elements)
                break
            i+=1
        if len(self.liste_value) == 0:
            self.liste_value.insert(0,value)
            self.liste_elements.insert(0,elements)
        self.liste_value = self.liste_value[:self.max_length]
        self.liste_elements = self.liste_elements[:self.max_length]

    def get_lists(self):
        return self.list_value, self.liste_elements

    def __len__(self):
        return len(self.liste_value)

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index == len(self.liste_value):
            raise StopIteration
        return self.liste_value[self.index], self.liste_elements[self.index]
   
    def is_ended(self):
        for elements in self.liste_elements:
            if elements[-1] != 123 :
                return False
        return True


def logits_possibilities(logits, corrections=4):
    '''
    fonction qui permet de determienr les 2**corrections sequences les plus probables selon les predictions (logits) du modele
    '''
    max_length = 2**corrections
    
    sorted_queue = SortedQueue(max_length)
    array = tf.nn.softmax(logits[0,:], axis=-1)
    p_log_p = np.sum(array * tf.math.log(array))    
    for j in range(array.shape[0]):
        value = -(p_log_p + tf.math.log(array[j])) / 2.
        elements = [j]
        sorted_queue.push(value, elements)

    while not sorted_queue.is_ended():
        sorted_queue2 = SortedQueue(max_length)
        itr = iter(sorted_queue)
        for value, list_elements in itr:
            if elements[-1] == 123 or len(elements)==35:
                sorted_queue2.push(value, elements)
                continue
            else:
                i = len(elements)
                array = tf.nn.softmax(logits[i,:])
                p_log_p = np.sum(array * tf.math.log(array))  
                for j in range(array.shape[0]):
                    value_ = -(p_log_p + tf.math.log(array[j])) / 2.
                    value_ = (value*i + value_)/(i + 1)
                    elements_ = elements+[j] 
                    sorted_queue2.push(value_, elements_)
        sorted_queue = deepcopy(sorted_queue2)

    liste_logits = []
    itr = iter(sorted_queue)
    for value, elements in itr:
        array = np.zeros(logits.shape)
        for index_i, index_j in enumerate(elements):
            array[index_i,index_j]=1
        liste_logits.append(tf.convert_to_tensor(array))
    return liste_logits
    
    
def inference(model, dataloader, batch_transforms):
    val_iter = iter(dataloader)
    
    liste_inference = []
    for images, targets in tqdm(val_iter):
        images_copy = deepcopy(images)
        images = batch_transforms(images)
        out = model(images, targets, return_model_output=True, return_preds=True, training=False)

        for outs in out['out_map']:
            logits = outs
            liste_logits = logits_possibilities(logits, corrections=4)
            liste_words = []
            print(liste_logits)
            for indexx, logits in enumerate(liste_logits):
            
                preds = model.postprocessor((np.expand_dims(logits,0)))
                print(preds,targets[indexx])
                words, _ = zip(*out["preds"])
                liste_words.append(words)
            liste_inference.append({"images": images_copy, 'predictions': liste_words, 'labels': targets})
    return liste_inference

def main(args):
    print(args)

    if not isinstance(args.workers, int):
        args.workers = min(7, mp.cpu_count())

    # Load doctr model
    model = recognition.__dict__[args.arch](
        pretrained=True,
        input_shape=(args.input_size, 4 * args.input_size, 3),
    )

    st = time.time()
    ds = datasets.__dict__[args.dataset](
        train=True,
        download=True,
        recognition_task=True,
        use_polygons=args.regular,
        img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )

    _ds = datasets.__dict__[args.dataset](
        train=False,
        download=True,
        recognition_task=True,
        use_polygons=args.regular,
        img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )
    ds.data.extend([(np_img, target) for np_img, target in _ds.data])

    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        shuffle=False,
    )
    print(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in " f"{len(test_loader)} batches)")

    mean, std = model.cfg["mean"], model.cfg["std"]
    batch_transforms = T.Normalize(mean=mean, std=std)


    print("Running evaluation")
    liste_inference = inference(model, test_loader, batch_transforms)
    
    with open("inference_" + str(args.dataset) + "_" + str(args.arch) + ".pkl", "wb") as f:
        pickle.dump(liste_inference,f)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text recognition (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--arch", type=str, help="text-recognition model to evaluate")
    parser.add_argument("--dataset", type=str, default="FUNSD", help="Dataset to evaluate on")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument("-j", "--workers", type=int, default=7, help="number of workers used for dataloading")
    parser.add_argument("--only_regular", dest="regular", action="store_true", help="test set contains only regular text")
    parser.add_argument("--vocab", type=str, default="french", help="Vocab to be used for evaluation")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
