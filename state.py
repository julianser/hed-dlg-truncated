from collections import OrderedDict

def prototype_state():
    state = {} 
     
    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    state['oov'] = '<unk>'
    state['len_sample'] = 40
    
    # These are end-of-sequence marks
    state['start_sym_sentence'] = '<s>'
    state['end_sym_sentence'] = '</s>'

    # This is obsolete
    #state['end_sym_triple'] = '</t>' 
    
    state['unk_sym'] = 0
    state['eot_sym'] = 3
    state['eos_sym'] = 2
    state['sos_sym'] = 1
    
    # Maxout requires qdim = 2x rankdim
    state['maxout_out'] = False
    state['deep_out'] = True
    state['dcgm_encoder'] = False

    # ----- ACTIV ---- 
    state['sent_rec_activation'] = 'lambda x: T.tanh(x)'
    state['triple_rec_activation'] = 'lambda x: T.tanh(x)'
    
    state['decoder_bias_type'] = 'all' # first, or selective 

    state['sent_step_type'] = 'gated'
    state['triple_step_type'] = 'gated' 

    # if turned on two utterances encoders (one forward and one backward) will be used, otherwise only a forward utterance encoder is used
    state['bidirectional_utterance_encoder'] = False
    # if turned on the L2 average pooling of the utterance encoders is used, otherwise the last state of the dialogue encoder(s) is used
    state['encode_with_l2_pooling'] = False

    state['direct_connection_between_encoders_and_decoder'] = False
    
    # ----- SIZES ----
    # Dimensionality of hidden layers
    state['qdim'] = 512
    # Dimensionality of triple hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank approximation
    state['rankdim'] = 256

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 5
    state['cost_threshold'] = 1.003

    # Initialization configuration
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = ''
    state['fix_pretrained_word_embeddings'] = False
     
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  
    # Maximum sequence length / trim batches
    state['seqlen'] = 80
    # Batch size
    state['bs'] = 80
    # Sort by length groups of  
    state['sort_k_batches'] = 20
   
    # Maximum number of iterations
    state['max_iters'] = 10
    # Modify this in the prototype
    state['save_dir'] = './'
    
    # ----- TRAINING PROCESS -----
    # Frequency of training error reports (in number of batches)
    state['train_freq'] = 10
    # Validation frequency
    state['valid_freq'] = 5000
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    # ----- EVALUATION PROCESS -----
    state['track_extrema_validation_samples'] = True # If set to true will print the extrema (lowest and highest log-likelihood scoring) validation samples
    state['track_extrema_samples_count'] = 100 # Set of extrema samples to track
    state['print_extrema_samples_count'] = 5 # Number of extrema samples to print (chosen at random from the extrema sets)

    state['compute_mutual_information'] = True # If true, the empirical mutural information will be calculcated on the validation set


    return state

def dcgm_test():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_triples'] = "./tests/data/ttrain.triples.pkl"
    state['test_triples'] = "./tests/data/ttest.triples.pkl"
    state['valid_triples'] = "./tests/data/tvalid.triples.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"
    
    # Handle bleu evaluation
    state['bleu_evaluation'] = "./tests/bleu/bleu_evaluation"
    state['bleu_context_length'] = 2

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    state['fix_pretrained_word_embeddings'] = True
    
    # Validation frequency
    state['valid_freq'] = 50
    
    # Varia
    state['prefix'] = "dcgm_testmodel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_out'] = True

    state['dcgm_encoder'] = True 
     
    # If out of memory, modify this!
    state['bs'] = 20
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all' #'selective' 
    
    state['qdim'] = 50 
    # Dimensionality of triple hidden layer 
    state['sdim'] = 100
    # Dimensionality of low-rank approximation
    state['rankdim'] = 10
    return state

def prototype_test():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_triples'] = "./tests/data/ttrain.triples.pkl"
    state['test_triples'] = "./tests/data/ttest.triples.pkl"
    state['valid_triples'] = "./tests/data/tvalid.triples.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    # Paths for semantic information 
    state['train_semantic'] = "./tests/data/ttrain.semantic.pkl"
    state['test_semantic'] = "./tests/data/ttest.semantic.pkl"
    state['valid_semantic'] = "./tests/data/tvalid.semantic.pkl"
    state['semantic_information_dim'] = 2
    state['bootstrap_from_semantic_information'] = False
    state['semantic_information_start_weight'] = 0.95
    state['semantic_information_decrease_rate'] = 0.001
    
    # Handle bleu evaluation
    state['bleu_evaluation'] = "./tests/bleu/bleu_evaluation"
    state['bleu_context_length'] = 2

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    state['fix_pretrained_word_embeddings'] = True
    
    # Validation frequency
    state['valid_freq'] = 50
    
    # Variables
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_out'] = True

    state['sent_step_type'] = 'gated'
    state['triple_step_type'] = 'gated' 
    #state['bidirectional_utterance_encoder'] = True 
    #state['encode_with_l2_pooling'] = True
    #state['direct_connection_between_encoders_and_decoder'] = False

    # If out of memory, modify this!
    state['bs'] = 20
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all' #'selective' 
    
    state['qdim'] = 50 
    # Dimensionality of triple hidden layer 
    state['sdim'] = 100
    # Dimensionality of low-rank approximation
    state['rankdim'] = 10
    return state

def prototype_moviedic():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_triples'] = "Data/Training.triples.pkl"
    state['test_triples'] = "Data/Test.triples.pkl"
    state['valid_triples'] = "Data/Validation.triples.pkl"
    state['dictionary'] = "Data/Training.dict.pkl" 
    state['save_dir'] = "Output" 

    # Paths for semantic information.
    # When bootstrapping from semantic information, the cost function optimized is 
    # a linear interpolation between the cross-entropy of genre prediction,
    # and the cross-entropy of utterance decoder.
    state['train_semantic'] = "Data/Training.genres.pkl"
    state['test_semantic'] = "Data/Test.genres.pkl"
    state['valid_semantic'] = "Data/Validation.genres.pkl"
    state['semantic_information_dim'] = 16
    state['bootstrap_from_semantic_information'] = True

    # Sets the initial weight of the semantic bootstrapping.
    # A weight of 0.95 means that the semantic bootstrapping will dominate in the beginning,
    #  which is a reasonable way to regularize the model.
    state['semantic_information_start_weight'] = 0.95

    # This parameter controls the (linear) decay rate of the semantic information bootstrapping.
    # Reasonable values for this parameter is between 0.0001-0.0008, because after seeing the 
    #  entire moviescript corpus once or twice, the semantic bootstrapping will be zero.
    state['semantic_information_decrease_rate'] = 0.0004 

    # Handle bleu evaluation
    state['bleu_evaluation'] = "Data/Mini_Validation_Shuffled_Dataset.txt"
    state['bleu_context_length'] = 2

    # Handle pretrained word embeddings. Using this requires rankdim=15
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = 'Data/Word2Vec_Emb.pkl' 
    state['fix_pretrained_word_embeddings'] = True
    
    # Validation frequency
    state['valid_freq'] = 2500
    
    # Varia
    state['prefix'] = "MovieScriptModel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = True
    state['deep_out'] = True
     
    # If out of memory, modify this!
    state['bs'] = 40
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective' 

    # Increase sequence length to fit movie dialogues better
    state['seqlen'] = 160

    state['qdim'] = 600
    # Dimensionality of triple hidden layer 
    state['sdim'] = 1200
    # Dimensionality of low-rank approximation
    state['rankdim'] = 300
    return state


