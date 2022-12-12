import os
import json

import numpy as np
import h5py
import click

import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def append_input(fname, data, label):
    if not os.path.exists(fname):
        mkw = {'chunks': True, 'compression': 'gzip'}
        subshp = tuple(data.shape[1:])
        with h5py.File(fname, 'w') as fd:
            fd.create_dataset(
                'data', shape=(0,) + subshp, dtype='float32', maxshape=(None,) + subshp, **mkw
            )
            fd.create_dataset('label', shape=(0,), dtype='uint16', maxshape=(None,), **mkw)

    with h5py.File(fname, 'a') as fd:
        n = fd['data'].shape[0]
        nnew = data.shape[0]
        fd['data'].resize(n + nnew, axis=0)
        fd['label'].resize(n + nnew, axis=0)

        fd['data'][n:] = data
        fd['label'][n:] = label


def append_attribution(fname, attrib, out, label):
    if not os.path.exists(fname):
        mkw = {'chunks': True, 'compression': 'gzip'}
        subshp = tuple(attrib.shape[1:])
        nout = tuple(out.shape[1:])
        with h5py.File(fname, 'w') as fd:
            fd.create_dataset(
                'attribution', shape=(0,) + subshp, dtype='float32', maxshape=(None,) + subshp, **mkw
            )
            fd.create_dataset(
                'prediction', shape=(0,) + nout, dtype='float32', maxshape=(None,) + nout, **mkw
            )
            fd.create_dataset('label', shape=(0,), dtype='uint16', maxshape=(None,), **mkw)

    with h5py.File(fname, 'a') as fd:
        n = fd['attribution'].shape[0]
        nnew = attrib.shape[0]
        fd['attribution'].resize(n + nnew, axis=0)
        fd['prediction'].resize(n + nnew, axis=0)
        fd['label'].resize(n + nnew, axis=0)

        fd['attribution'][n:] = attrib
        fd['prediction'][n:] = out
        fd['label'][n:] = label

def main():

    input_file = "test-project/input.h5"
    attribution_file = "test-project/attribution.h5"
    label_map_file = "test-project/label-map.json"

    num_classes = 2

    label_map = [
        {
            'index': i,
            'word_net_id': f'{i:08d}',
            'name': f'Class {i:d}',
        } for i in range(num_classes)
    ]
    with open(label_map_file, 'w') as fd:
        json.dump(label_map, fd)

    #cifar_img_path_rel = "lrp_results/backgrounds_imgs"
    handwheel_img_path_rel = "lrp_results/handwheels_imgs"
    #cifar_attrib_path_rel = "lrp_results/backgrounds_attrib"
    handwheel_attrib_path_rel = "lrp_results/handwheels_attrib"

    #cifar_img_path = os.path.join(str(Path().absolute()), cifar_img_path_rel)    
    handwheel_img_path = os.path.join(str(Path().absolute()), handwheel_img_path_rel)   
    #cifar_attrib_path = os.path.join(str(Path().absolute()), cifar_attrib_path_rel)   
    handwheel_attrib_path = os.path.join(str(Path().absolute()), handwheel_attrib_path_rel)   

    #cifar_imgs = sorted(os.listdir(cifar_img_path))
    handwheel_imgs = sorted(os.listdir(handwheel_img_path))
    #cifar_attribs = sorted(os.listdir(cifar_attrib_path))
    handwheel_attribs = sorted(os.listdir(handwheel_attrib_path))

    print(len(handwheel_attribs), " Handwheel attribs")
    print(len(handwheel_imgs), "Handwheel imgs")

    cifar_imgs_list = []
    handwheel_imgs_list = []
    cifar_attribs_list = []
    handwheel_attribs_list = []
    labels_hw = []
    outs = []

    for handwheel_img, handwheel_attrib in zip(handwheel_imgs, handwheel_attribs):
        print(handwheel_img_path + '/' + handwheel_img)
        print(handwheel_attrib_path + '/' + handwheel_attrib)
        img = cv2.imread(handwheel_img_path + '/' + handwheel_img)
        height = 224
        dim = (height, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).transpose()
        attrib = cv2.imread(handwheel_attrib_path + '/' + handwheel_attrib)
        #attrib = cv2.cvtColor(attrib, cv2.COLOR_BGR2RGB)
        norm_image = cv2.normalize(attrib, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        attrib = np.array(norm_image).transpose()
        #attrib = np.expand_dims(attrib, axis=0)
        print(attrib)
        handwheel_imgs_list.append(img)
        handwheel_attribs_list.append(attrib)
        labels_hw.append(0)
        outs.append(0)

    label = np.array([0] * len(handwheel_imgs))
    out = np.random.uniform(0, 0, size=(len(handwheel_imgs), 1))
    print("Label: ", label.shape)
    print("Out: ", out.shape)
    print("imgs: ", np.array(handwheel_imgs_list).shape)
    print("attribs: ",np.array(handwheel_attribs_list).shape)

    append_input(input_file, np.array(handwheel_imgs_list), label)
    append_attribution(attribution_file, np.array(handwheel_attribs_list), out, label)

    labels_cifar = []
    outs = []

    ## SAVE TO JSON
    hw_json_object = json.dumps(handwheel_attribs, indent = 4)

    with open('./metrics/hw_data.json', "w") as outfile:
        outfile.write(hw_json_object)
    '''
    for cifar_img, cifar_attrib in zip(cifar_imgs, cifar_attribs):
        print(cifar_img_path + '/' + cifar_img)
        print(cifar_attrib_path + '/' + cifar_attrib)
        img = cv2.imread(cifar_img_path + '/' + cifar_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).transpose()
        attrib = cv2.imread(cifar_attrib_path + '/' + cifar_attrib)
        #attrib = cv2.cvtColor(attrib, cv2.COLOR_BGR2RGB)
        #norm_image = cv2.normalize(attrib, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        attrib = np.array(attrib).transpose()
        #attrib = np.expand_dims(attrib, axis=0)
        print(attrib)
        cifar_imgs_list.append(img)
        cifar_attribs_list.append(attrib)
        labels_cifar.append(1)
        outs.append(1)

    label = np.array([1] * len(cifar_imgs))
    out = np.random.uniform(1, 1, size=(len(cifar_imgs), 1))
    print("Label: ", label.shape)
    print("Out: ", out.shape)
    print("imgs: ", np.array(cifar_imgs_list).shape)
    print("attribs: ",np.array(cifar_attribs_list).shape)

    append_input(input_file, np.array(cifar_imgs_list), label)
    append_attribution(attribution_file, np.array(cifar_attribs_list), out, label)
    '''
if __name__ == '__main__':
    main()
