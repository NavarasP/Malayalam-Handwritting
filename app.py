import os
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory, send_file, flash
from flask import jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import urllib
from scipy import ndimage
import csv
import sys
from fastai import *
from fastai.vision import *
# from fastai.metrics import accuracy
import json
import spell_correct
# from fastai.vision.all import defaults
# from fastai.learner import load_learner 
# from fastai.vision.all import *
from fastai.layers import FlattenedLoss

import subprocess

torch.nn.Module.dump_patches = True


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER_PNG = os.path.join(os.getcwd(),'output/png')
OUTPUT_FOLDER_PNG_x64 = os.path.join(os.getcwd(),'output/png_x64')
OUTPUT_FOLDER_SVG = os.path.join(os.getcwd(),'output/svg')
OUTPUT_FOLDER_FONT = os.path.join(os.getcwd(),'output/fonts')
OUTPUT_FOLDER_METADATA = os.path.join(os.getcwd(),'output/font_metadata')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
import torch  # Missing import for setting default device
#  # Ensure this is imported for loading models
import dill


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')


import numpy as np
import dill

def predict_alphabets(image_path):
    print(image_path)
    defaults.device = torch.device("cpu")
    alphabets = ['്', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'െ', 'ൃ', 'െ', 'ൌ', 'ം', 'അ', 'ആ', 'ഇ', 'ഉ', 'ഋ', 'എ', 'ഏ', 'ഒ', 'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ', 'ത', 'ഫ', 'ദ', 'ധ', 'ന', 'പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'യ', 'ര',
                 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ', 'ക്ക', 'ക്ഷ', 'ങ്ക', 'ങ്ങ', 'ച്ച', 'ഞ്ച', 'ഞ്ഞ', 'ട്ട', 'ണ്ട', 'ണ്ണ', 'ത്ത', 'ദ്ധ', 'ന്ത', 'ന്ദ', 'ന്ന', 'പ്പ', 'മ്പ', 'മ്മ', 'യ്യ', 'ല്ല', 'ള്ള', 'വ്വ', '്യ', '്ര', '്വ']
    labels = {'12': 20, '54': 59, '79': 81, '26': 33, '83': 85, '11': 2, '18': 26, '55': 6, '42': 48, '14': 22, '36': 42, '63': 67, '57': 61, '60': 64, '70': 73, '31': 38, '65': 69, '0': 1, '73': 76, '51': 56, '17': 25, '84': 9, '1': 10, '64': 68, '38': 44, '47': 52, '23': 30, '68': 71, '80': 82, '75': 78, '30': 37, '72': 75, '82': 84, '43': 49, '62': 66, '49': 54, '77': 8, '5': 14, '34': 40, '16': 24, '25': 32, '29': 36,
              '39': 45, '21': 29, '74': 77, '46': 51, '44': 5, '27': 34, '35': 41, '37': 43, '58': 62, '9': 18, '4': 13, '41': 47, '40': 46, '81': 83, '67': 70, '59': 63, '69': 72, '52': 57, '78': 80, '15': 23, '33': 4, '24': 31, '22': 3, '76': 79, '10': 19, '3': 12, '50': 55, '28': 35, '6': 15, '19': 27, '2': 11, '7': 16, '53': 58, '32': 39, '61': 65, '8': 17, '13': 21, '45': 50, '71': 74, '44': 5, '48': 53, '56': 60, '20': 28, '66': 7}
    
    

    
    ens = ["alexnet_non_pretrained_bs32.pkl", "resnet_18_non-pre_trained.pkl"]
    ens_test_preds = []
    for mod in ens:
        learn = load_learner(path='.', file=mod)

        img = open_image(image_path)
        pred_class, pred_idx, outputs = learn.predict(img)
        ens_test_preds.append(np.array(outputs))
    ens_preds = np.mean(ens_test_preds, axis=0)
    
    return (alphabets[int(labels[str(np.argmax(ens_preds))]) - 1])


def _edge_detect(im):
    return np.max(np.array([_sobel_detect(im[:, :, 0]), _sobel_detect(im[:, :, 1]), _sobel_detect(im[:, :, 2])]), axis=0)


def _sobel_detect(channel):  # sobel edge detection
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def sort_words(boxes):
    for i in range(1, len(boxes)):
        key = boxes[i]
        j = i - 1
        while(j >= 0 and key[2] < boxes[j][2]):
            boxes[j+1] = boxes[j]
            j -= 1
        boxes[j+1] = key
    return boxes


def sort_boxes(boxes):
    lines = []
    new_lines = []
    tmp_box = boxes[0]
    lines.append(tmp_box)
    for box in boxes[1:]:
        if((box[0] + (box[1] - box[0])/2) < tmp_box[1]):
            lines.append(box)
            tmp_box = box
        else:
            new_lines.append(sort_words(lines))
            lines = []
            tmp_box = box
            lines.append(box)
    new_lines.append(sort_words(lines))
    return(new_lines)


def sort_labels(label_boxes):
    for i in range(1, len(label_boxes)):
        key = label_boxes[i]
        j = i - 1
        while(j >= 0 and key[1][2] < label_boxes[j][1][2]):
            label_boxes[j+1] = label_boxes[j]
            j -= 1
        label_boxes[j+1] = key
    return label_boxes


def scale_boxes(new_img, old_img, coords):
    coords[0] = int((coords[0]) * new_img.shape[0] /
                    old_img.shape[0])  # new top left
    coords[1] = int((coords[1]) * new_img.shape[1] /
                    old_img.shape[1])  # new bottom left
    coords[2] = int((coords[2] + 1) * new_img.shape[0] /
                    old_img.shape[0]) - 1  # new top right
    coords[3] = int((coords[3] + 1) * new_img.shape[1] /
                    old_img.shape[1]) - 1  # new bottom right
    return coords


def clipping_image(new):
    colsums = np.sum(new, axis=0)
    linessum = np.sum(new, axis=1)
    colsums2 = np.nonzero(0-colsums)
    linessum2 = np.nonzero(0-linessum)

    x = linessum2[0][0]  # top left
    xh = linessum2[0][linessum2[0].shape[0]-1]  # bottom left
    y = colsums2[0][0]  # top right
    yw = colsums2[0][colsums2[0].shape[0]-1]  # bottom right

    imgcrop = new[x:xh, y:yw]  # crop the image

    return imgcrop, [x, xh, y, yw]


def padding_resizing_image(img, reduce_size):
    # add 2px padding to image
    img = cv2.copyMakeBorder(img, 2, 2, 0, 0, cv2.BORDER_CONSTANT)
    try:
        # resize the image to 32*32
        img = cv2.resize(np.uint8(img), (reduce_size, reduce_size))
    except:
        return img
    finally:
        return img


def segmentation(img, sequence, origimg=None, wordNo=None, filename=None):
    if(sequence == "word"):  # resize to find the words
        width = 940
        height = int(img.shape[0] * (width / img.shape[1]))
        sigma = 18
    elif(sequence == "character"):  # resize to find the characters
        width = img.shape[1]  # 1280
        height = img.shape[0]  # int(img.shape[0] * (width / img.shape[1]))
        sigma = 0

    img = cv2.resize(img, (width, height))
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)  # apply gaussian blur

    if(sequence == "word"):
        blurred = _edge_detect(blurred)  # edge detect in blurred image (words)
        # Otsu's thresholding with Binary
        ret, img = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Morphological processing - Black&White
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                               np.ones((15, 15), np.uint8))

    elif(sequence == "character"):
        # Otsu's thresholding with Binary Inverted
        ret, img = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    num_labels, labels_im = cv2.connectedComponents(
        img)  # find the connected components

    if(sequence == "word"):
        boxes = []  # for storing the coordinates of the bounding boxes
        for i in range(1, num_labels):
            # select the images with label
            new, nr_objects = ndimage.label(labels_im == i)
            # clipping the image to the edges
            new, new_coord = clipping_image(new)
            if(not(new.shape[0] < 10 or new.shape[1] < 10)):
                boxes.append(new_coord)

    if(sequence == "character"):
        boxes = []
        label_box = []
        for i in range(1, num_labels):
            # select the images with label
            new, nr_objects = ndimage.label(labels_im == i)
            # clipping the image to the edges
            new, new_coord = clipping_image(new)
            if(not(new.shape[0] < 10 or new.shape[1] < 10)):
                label_box.append([i, new_coord])
        label_box = sort_labels(label_box)  # sort the words
        chNo = 0
        for box in label_box:
            ch_img, nr_objects = ndimage.label(labels_im == box[0])
            ch_img, new_coord = clipping_image(ch_img)
            cropped_image = padding_resizing_image(ch_img, 32)
            cropped_image_x64 = padding_resizing_image(ch_img, 64)
            try:
                dst = os.path.join(OUTPUT_FOLDER_PNG, filename,
                                   str(wordNo)+"_"+str(chNo)+".png")
                dst_x64 = os.path.join(
                    OUTPUT_FOLDER_PNG_x64, filename, str(wordNo)+"_"+str(chNo)+".png")
                plt.imsave(dst, cropped_image, cmap=cm.gray)
                plt.imsave(dst_x64, cropped_image_x64, cmap=cm.gray)
            except:
                pass
            finally:
                pass
            chNo += 1
    return img, boxes


def img_to_seg(img, filename):
    # change image width to 1280px
    img = cv2.resize(img, (1280, int(img.shape[0] * (1280 / img.shape[1]))))
    kernel = np.ones((5, 5), np.uint8)  # kernel for opening
    # opening image to remove minor noise
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    new_img, boxes = segmentation(img, "word")  # line segmentation
    boxes.sort()
    boxes = sort_boxes(boxes)
    scaled_boxes = []  # coordinates values scaled to img
    for box in boxes:
        for box in box:
            box = scale_boxes(img, new_img, box)
            scaled_boxes.append(box)
    wordNo = 0
    for scaled_box in scaled_boxes:
        img_gray = cv2.cvtColor(
            img[scaled_box[0]:scaled_box[1], scaled_box[2]:scaled_box[3]], cv2.COLOR_BGR2GRAY)
        img_new, _ = segmentation(
            img_gray, "character", None, wordNo, filename.split(".")[0])
        wordNo += 1
    return "1", wordNo


@app.route('/')
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename, file=sys.stdout)
            os.makedirs(os.path.join(OUTPUT_FOLDER_PNG,
                                     filename.split(".")[0]), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_FOLDER_SVG,
                                     filename.split(".")[0]), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_FOLDER_PNG_x64,
                                     filename.split(".")[0]), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_FOLDER_METADATA,
                                     filename.split(".")[0]), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_FOLDER_FONT,
                                     filename.split(".")[0]), exist_ok=True)
            img = cv2.cvtColor(cv2.imread(os.path.join(
                UPLOAD_FOLDER, filename)), cv2.COLOR_BGR2RGB)  # input image
            x, wordNo = img_to_seg(img, filename)

    if x == "1":
        metadata = {
            "props": {
                "ascent": 800,
                "descent": 200,
                "em": 1000,
                "family": "Example"
            },
            "input": "",
            "output": [""],
            "glyphs": {}
        }
        metadata["input"] = str("../svg/" + filename.split(".")[0])
        metadata["output"][0] = "../fonts/" + filename.split(".")[0]+".ttf"
        alpha = {}  # dictionary with predicted alphabet and image name
        char_filenames = {}
        hybrid = ['ക്ക', 'ക്ഷ', 'ങ്ക', 'ങ്ങ', 'ച്ച', 'ഞ്ച', 'ഞ്ഞ', 'ട്ട', 'ണ്ട', 'ണ്ണ', 'ത്ത', 'ദ്ധ',
                  'ന്ത', 'ന്ദ', 'ന്ന', 'പ്പ', 'മ്പ', 'മ്മ', 'യ്യ', 'ല്ല', 'ള്ള', 'വ്വ', '്യ', '്ര', '്വ']
        hybrid_dict = {'ക്ക': '0x1000c', 'ക്ഷ': '0x10030', 'ങ്ക': '0x10074', 'ങ്ങ': '0x10078', 'ച്ച': '0x1007d', 'ഞ്ച': '0x100a4', 'ഞ്ഞ': '0x100af', 'ട്ട': '0x100b6', 'ണ്ട': '0x100d6', 'ണ്ണ': '0x100e8', 'ത്ത': '0x100f2', 'ദ്ധ': '0x10132',
                       'ന്ത': '0x10156', 'ന്ദ': '0x10163', 'ന്ന': '0x10171', 'പ്പ': '0x10194', 'മ്പ': '0x101f9', 'മ്മ': '0x10200', 'യ്യ': '0x1022a', 'ല്ല': '0x10257', 'ള്ള': '0x1030b', 'വ്വ': '0x10263', '്യ': '0x10003', '്ര': '0x10002', '്വ': '0x10006'}
        for i in range(wordNo):
            alpha[str(i)] = {}
        for f in os.listdir(os.path.join(OUTPUT_FOLDER_PNG, filename.split(".")[0])):
            character = str(predict_alphabets(os.path.join(
                OUTPUT_FOLDER_PNG, filename.split(".")[0], f)))
            alpha[str(f.split("_")[0])][str(
                f.split("_")[1].split(".")[0])] = character
            if(not(character in hybrid)):
                char_filenames[str(character.encode("unicode_escape")).split(
                    "\\")[-1].split("\'")[0].replace('u0', '0x')] = f.split(".")[0] + ".svg"
                subprocess.call(["convert", os.path.join(OUTPUT_FOLDER_PNG_x64, filename.split(".")[
                                0], f), "-negate", os.path.join(OUTPUT_FOLDER_SVG, filename.split(".")[0], f.split(".")[0]+".svg")])
            else:
                char_filenames[str(hybrid_dict[character])
                               ] = f.split(".")[0] + ".svg"
                subprocess.call(["convert", os.path.join(OUTPUT_FOLDER_PNG_x64, filename.split(".")[
                                0], f), "-negate", os.path.join(OUTPUT_FOLDER_SVG, filename.split(".")[0], f.split(".")[0]+".svg")])
        metadata["glyphs"] = char_filenames
        print(metadata)
        with open(os.path.join(OUTPUT_FOLDER_METADATA, filename.split(".")[0]+str(".json")), 'w') as fp:
            json.dump(metadata, fp)
        print(char_filenames)
        print(alpha)
        sentence = ""
        words_list = []
        for word in range(0, len(alpha)):
            string = ""
            for ch in range(len(alpha[str(word)])):
                string += alpha[str(word)][str(ch)]
            words_list.append(spell_correct.correction(string))
            sentence += string
            sentence += " "
        print(words_list)
        print(sentence)
        # subprocess.call(
        #     ["./svgs2ttf.py", str(os.path.join(OUTPUT_FOLDER_METADATA, filename.split(".")[0]+".json"))])
        # subprocess.call(["fontforge", "-script", "svgs2ttf.py", os.path.join(OUTPUT_FOLDER_METADATA, filename.split(".")[0]+".json")])

    # return jsonify(character = alpha)
    return render_template('result.html', messages={'string': sentence, 'image': str(filename), 'font': str(filename.split(".")[0])+'.ttf', 'possible_words': words_list})


@app.route('/upload/<filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/download/<filename>')
def fonts(filename):
    return send_file(os.path.join(OUTPUT_FOLDER_FONT, filename), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
