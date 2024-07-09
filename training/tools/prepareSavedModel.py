import os
import sys
import json
import shutil
import numpy as np
import argparse

from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
import tensorflow as tf
from keras.models import load_model
import keras.backend as K


def main(filepath):

    #This takes a model saved in the .h5 format and prepares it as a .pb, for use in CMSSW
    #Note, this needs to be run on the GPU with the same version of keras as you trained with
    K.set_learning_phase(0)

    # model = load_model('BEST_model.h5')
    # modelPath = '/uscms/home/msabbott/submitBEST/training/models/nnBEST_nopt/flattened_2018/BEST_model_flattened_2018.h5'
    # modelPath = '../models/nnBEST_nopt/flattened_2018/BEST_model_flattened_2018.h5'
    modelPath = filepath
    model = load_model(modelPath)

    print (model.inputs)
    print (model.outputs)


    if "2015" in filepath:
        year = "2015"
    elif "2016" in filepath:
        year = "2016"
    elif "2017" in filepath:
        year = "2017"
    elif "2018" in filepath:
        year = "2018"
    else:
        print("ERROR: year not interpreted correctly from file. ")
        return

    #output_str = [  str(model.outputs).split("'")[1].split(":")[0]       ]
    #outputs = output_str
    with K.get_session() as sess:
        # outputs = ["dense_4/Softmax"]
        # outputs = ["dense_8/Softmax"]
        # outputs = ["dense_12/Softmax"]
        #outputs = ["dense_16/Softmax"]
        output_str = [  str(model.outputs).split("'")[1].split(":")[0]       ]
        outputs = output_str
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)
        tf.train.write_graph(constant_graph, 'GraphExport/', "constantgraph_%s.pb"%year, as_text=False)
        if os.path.exists('BuilderExport/'): shutil.rmtree('BuilderExport/')
        builder = tf.saved_model.builder.SavedModelBuilder('BuilderExport/')
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()

    #os.system("cp GraphExport/constantgraph_%s.pb /uscms_data/d3/cannaert/analysis/CMSSW_10_6_30/src/data/BEST_models/"%year)
    #print("Model copied to /uscms_data/d3/cannaert/analysis/CMSSW_10_6_30/src/data/BEST_models/")
    #print("BE CAREFUL: if you change analysis folder locations, this path will also need to be changed.")

    #print("REMEMBER: if you change the BES variables and their scaling you need to copy the BESTScalerParameters ... text files to $ANHOME.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script with file path option")    
    parser.add_argument("--filepath", help="Path to the file")

    args = parser.parse_args()
    
    main(args.filepath)


