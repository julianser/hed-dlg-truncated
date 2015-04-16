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
    state['end_sym_triple'] = '</t>'
    
    state['unk_sym'] = 0
    state['eot_sym'] = 3
    state['eos_sym'] = 2
    state['sos_sym'] = 1
    
    state['maxout_out'] = False
    state['deep_out'] = True

    # ----- ACTIV ---- 
    state['sent_rec_activation'] = 'lambda x: T.tanh(x)'
    state['triple_rec_activation'] = 'lambda x: T.tanh(x)'
    
    state['decoder_bias_type'] = 'all' # first, or selective 

    state['sent_step_type'] = 'gated'
    state['triple_step_type'] = 'gated' 
    
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
    return state

def prototype_test():
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
    
    # Validation frequency
    state['valid_freq'] = 500
    
    # Varia
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_out'] = True
     
    # If out of memory, modify this!
    state['bs'] = 80
    state['use_nce'] = True
    state['decoder_bias_type'] = 'selective' 
    
    state['qdim'] = 50 
    # Dimensionality of triple hidden layer 
    state['sdim'] = 100
    # Dimensionality of low-rank approximation
    state['rankdim'] = 25
    return state


