#!/usr/bin/env python

    ##------------------------------------------------------------------------------##
    #                                                                                #
    #   Our end goal is to use those encodings to improve, among other things,       #
    #   caption generation.                                                          #
    #                                                                                #
    #   So, let's say we want to predict what's happening in video DVS_n, n being    # 
    #   the n-ieme segment in our movie. We assume we know what happened before,     #
    #   from DVS_0 to DVS_n-1.                                                       #
    #   What we want to do is use the context given by m previous sentences,         #
    #   either ground truth or predicted, to make a prediction for the current       #
    #   sentence. Then we use this prediction as another set of features in our      # 
    #   video model.                                                                 #
    #                                                                                #   
    #   Therefore, for each sentence, the context is built only with the previous    #
    #   sentences. And then we compute the predicted encoding.                       #
    #                                                                                #
    #   Each entry in the dictionary V_n is the predicted value for the video n      #
    #   considering the n-m previous one. No overlap, no seeing into the future      #
    #   (or even see the present). Can be used directly.                             #
    #                                                                                #
    ##------------------------------------------------------------------------------##
    
import argparse
import cPickle
import traceback
import logging
import time
import sys
import math

import os
import numpy
import codecs
import search
import utils

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state

logger = logging.getLogger(__name__)

def build_text_context(sentenceID, dictionary, nb_sentences_back):
    
    ##------------------------------------------------------------------------------##
    #   We build here the context.                                                   #   
    #   SentenceID corresponds to the last sentence you encounter                    #
    #   The context must be in chronological order, so the sentence your looking at  #
    #   is the last element.                                                         #   
    ##------------------------------------------------------------------------------##

    context  = []
    
    childID     = dictionary[sentenceID][1]
    
    if childID == "None":
        print "----> First segment. No context available."
        return "no_context"
    
    else:
        parentID    = sentenceID
        newSentence = ""

        for i in range(nb_sentences_back):
            childID     = dictionary[parentID][1]
            if childID != "None":
                newSentence = dictionary[childID][0]
                context.append(newSentence)
                parentID    = childID
            else:
                print "----> Reached the beginning."
                print "----> Context (sentence, non chronological): ", context
                context = context[::-1]
                print "----> Context (sentence, chronological): ", context
                return context

        print "----> Context (sentence, non chronological): ", context
        context = context[::-1]                                                         
        print "----> Context (sentence, chronological): ", context
        return context

def build_context(model, sentenceID, dictionary, nb_sentences_back):

    ##------------------------------------------------------------------------------##
    #           Here we convert the context from words to numbers.                   #
    #           We build that the vector needed to compute the encoding.             #
    ##------------------------------------------------------------------------------##

    context_text     = build_text_context(sentenceID, dictionary, nb_sentences_back)
    joined_context   = []
    reversed_context = []

    if context_text == "no_context":
            joined_context   = [0]
            reversed_context = [0]  #[model.eos_sym]
    
    else:
        for context_id, context_sentences in enumerate(context_text):
            if len(context_text) == 0:
                joined_context = [model.eos_sym]
            else:
                joined_context += [model.eos_sym]
                sentence_ids = model.words_to_indices(context_sentences.split())
            
                joined_context   += sentence_ids + [model.eos_sym]
                reversed_context += [model.eos_sym] + sentence_ids[::-1] + [model.eos_sym] 
                # The reversed context doesn't change the order of the sentences.
                # Only the words inside each sentence are reversed.
    
    return joined_context, reversed_context


def compute_encoding(model, encodingFunc, context, reversed_context, max_length):
    
    ##------------------------------------------------------------------------------##
    #       Here we actually compute the encoding for a given sentence and its       #
    #       context. The needed value is hs. Check dialog_enc.py for details of      #
    #       its computation.                                                         #
    ##------------------------------------------------------------------------------##
    
    # Needed for legacy reasons. Still needed to run the encoding function
    semantic_info            = numpy.ones((1,1), dtype= 'int32')  
    
    # The encoding function need a 2d input.
    updated_context          = numpy.array(context, ndmin=2, dtype= "int32")
    updated_reversed_context = numpy.array(reversed_context, ndmin=2, dtype="int32")
    
    print "----> Updated context:", updated_context
    print "----> Updated reversed context:", updated_reversed_context
    
    # We transpose to get a column vector. Otherwise, the encoding function thinks you have a bunch of one word sentence.
    # In that case, you always ends with the sames 3 vectors. ( Why three? Who knows...) 
    updated_context          = updated_context.T
    updated_reversed_context = updated_reversed_context.T
    
    encoder_states = encodingFunc(updated_context, updated_reversed_context, max_length, semantic_info)
    hs             = encoder_states[1]    # The encoder returns h and hs (in this order) we want the latter.
    
    # The shape of your encoding should be in the end (1, N). N being usually in the thousands. 
    print "----> hs shape before ", hs.shape
    print "----> hs[:, -1, :] shape ", numpy.shape(hs[:,-1,:])
    print "----> hs[-1, :, :] shape ", numpy.shape(hs[-1,:,:])
    print "----> hs[:, :, -1] shape ", numpy.shape(hs[:,:,-1])
    
    # Get embedding at the last end-of-sentence / end-of-utterance token
    # This is necessary if we padd context with zeros at the end.
    # Otherwise, it won't do anything :)
    last_eos_index = -1
    for i in range(len(context)):
        if context[i] == model.eos_sym:
             last_eos_index = i

    return hs[last_eos_index, :, :]
    #return hs[-1, :,:]


def compute_encoding_trunc(model, encodingFunc, context, reversed_context, max_length):
    
    ##------------------------------------------------------------------------------##
    #                  Alternative way to compute the encodings.                     #
    #           Truncate the context if it's longer than some set value.             #
    #                  May disappear or be merged in the future.                     #
    ##------------------------------------------------------------------------------##
    
    context_length           = len(context)      
    semantic_info            = numpy.zeros((1,1), dtype= 'int32')  # Once again, legacy.

    if  context_length < max_length:
        print "----> Smaller than max length"
        updated_context          = numpy.zeros(context_length, dtype = 'int32') 
        updated_reversed_context = numpy.zeros(context_length, dtype='int32')
        
        for idx in range(context_length): 
           updated_context[idx]          = context[idx]
           updated_reversed_context[idx] = reversed_context[idx]
    else:
        
        # If context is longer the max context, truncate it and force the end-of-utterance token at the end
        print "----> Longer than max length"
        
        updated_context          = numpy.zeros(max_length, dtype = 'int32') #max_length, dtype='int32')
        updated_reversed_context = numpy.zeros(max_length, dtype='int32')
        
        for idx in range(max_length - 1): 
            updated_context[idx]          = context[idx]
            updated_reversed_context[idx] = reversed_context[idx]
        
        updated_context[max_length - 1]          = model.eos_sym
        updated_reversed_context[max_length - 1] = model.eos_sym
    
    updated_context          = numpy.array(context, ndmin=2, dtype= "int32")
    updated_reversed_context = numpy.array(reversed_context, ndmin=2, dtype="int32")
   
    # Again, we transpose to get a column vector...
    updated_context          = updated_context.T
    updated_reversed_context = updated_reversed_context.T
    
    encoder_states = encodingFunc(updated_context, updated_reversed_context, max_length, semantic_info)
    hs             = encoder_states[1] 
    
    return hs[-1, :, :]

def get_encoding(model, encoding_func, sentenceID, sentenceDict, max_length, nb_sent_back): # **kwargs):
  
    ##------------------------------------------------------------------------------##
    #               Compute the encoding for a single sentence.                      #
    ##------------------------------------------------------------------------------##
    
    context, reversed_context = build_context(model, sentenceID, sentenceDict, nb_sent_back)
    encoding = compute_encoding(model, encoding_func, context, reversed_context, max_length)
    
    return encoding

def get_all_encodings(model, encoding_func, sentenceDict, max_length, nb_sent_back, outputName):

    ##------------------------------------------------------------------------------##
    #   Compute the encodings for the whole set contained in sentenceDict.           #                                           
    ##------------------------------------------------------------------------------##
    
    
    encodingDict = {}
    joined_context = []
    joined_reversed_context = []
    sentenceNb = 1               # Count the number of processed sentences. Gives an idea to where you are in the batch.
    
    for keys in sentenceDict:
        
        print "\n----> Sentence Number ", sentenceNb
        print "----> Sentence Name ", keys

        encodingDict[keys] =  get_encoding(model, encoding_func, keys, sentenceDict, max_length, nb_sent_back)
    
        sentenceNb += 1
        #print "begin",  encodingDict[keys][0,:50]
        #print "middle", encodingDict[keys][0,1000:1050]
        #print "end", encodingDict[keys][0,1950:]

    print "----> Dummping the encodings..."
    cPickle.dump(encodingDict, open(outputName + ".pkl", "w"))
    print "\tL----> Done."

    return encodingDict

def init(path):
    
    ##------------------------------------------------------------------------------##
    #                Compile and load the model.                                     #          
    #                Compile the encoder.                                            #                                           
    ##------------------------------------------------------------------------------##

    state = prototype_state()
    state_path   = path  + "_state.pkl"
    model_path   = path  + "_model.npz"
    
    with open(state_path) as src:
        state.update(cPickle.load(src))
    
    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    model = DialogEncoderDecoder(state) 
    
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    
    encoding_function = model.build_encoder_function()
    
    return model, encoding_function

def main(**kwargs):

    args = parse_args()

    sentenceDict = cPickle.load(open(args.sentenceDict, "rb"))

    model, encoding_function = init(args.model_path)
    
    if args.batch:
        print "---> Computing encoding in batch..."
        print args.sentenceDict
        get_all_encodings(model, encoding_function, sentenceDict, args.max_length, args.nback, args.output_name)
        print "\tL----> All done."

    if args.one_sentence:
        
        print "---> Computing encoding for " , args.sentenceID
        encoding = get_hidden_state(model, encoding_function, args.sentence_ID, sentenceDict, args.max_length, args.nback)
        print "\tL----> Done."
        
        if args.output_name is None:
            name = args.sentenceID
        else:
            name = args.output_name
        
        print "----> Dummping the encodings..."
        encoding_pkl = cPickle.dump(encoding, open(name + ".pkl", "wb"))
        print "\tL----> Done."


def parse_args():
    
    parser = argparse.ArgumentParser("Compute encodings from model, for different context lenghts.")
    
    parser.add_argument("--model_path",   type = str, help="Path to the model prefix (without _model.npz or _state.pkl).")
    parser.add_argument("--sentenceDict", help="Path to the dictionnary. Contain each sentence ID as keys and the sentence and previous ID as values.")
    parser.add_argument("--output_name",  type = str, help="Name of the output file.")
    
    parser.add_argument("--one_sentence", action = "store_true", help="Compute encodings for one sentence only.")
    parser.add_argument("--batch",        action = "store_true", help="Compute encodings in batch.")
    
    parser.add_argument("--one_set",      action = "store_true", help="Compute encodings in batch, for one set of context.")
    parser.add_argument("--one_model",    action = "store_true", help="Compute encodings in batch, for one particular model.")
    parser.add_argument("--one_nback",    action = "store_true", help="Compute encodings in batch, for one particular context lenght.")
    
    parser.add_argument("--all_sets",     action = "store_true", help="Compute encodings in batch, for training, validation and test sets.")
    parser.add_argument("--all_models",   action = "store_true", help="Compute encodings in batch, for different model.")
    
    parser.add_argument("--nback",        type = int,            default = 4,   help="Number of sentences back to use to build context. The current sentence doesn't count in nback and is not included in your context.")
    parser.add_argument("--full_context", action = "store_true", help="Get the full lenght of the context. For each sentence, the context go back until reaching the begining of the movie.")
    parser.add_argument("--max_length",   type = int,            default = 200, help="Max number of words in the context.")
    
    parser.add_argument("--sentence_ID",  type = str,            help="ID of the sentence you want to get the encoding. Basically, the name of the DVS or video.")
        
    parser.add_argument("--verbose",      action="store_true",   default=False, help="Be verbose")

    return parser.parse_args()

if __name__ == "__main__":
    
    main()
    
    # Example of sentence ID.
    # ID = "3010_BIG_MOMMAS_LIKE_FATHER_LIKE_SON_00.03.27.568-00.03.35.097"
