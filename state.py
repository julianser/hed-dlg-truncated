from collections import OrderedDict
import cPickle

def prototype_state():
    state = {} 
     
    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    # Out-of-vocabulary token string
    state['oov'] = '<unk>'
    
    # These are end-of-sequence marks
    state['end_sym_sentence'] = '</s>'

    # Special tokens need to be hardcoded, because model architecture may adapt depending on these
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = 2 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = 3 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = 4 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = 5 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = 6 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = 7 # voice over symbol <voice_over>
    state['off_screen_sym'] = 8 # off screen symbol <off_screen>
    state['pause_sym'] = 9 # pause symbol <pause>

    # Training examples will be split into subsequences of size max_grad_steps each.
    # Gradients will be computed on the subsequence, and the last hidden state of all RNNs will
    # be used to initialize the hidden state of the RNNs in the next subsequence.
    state['max_grad_steps'] = 80

    # If this flag is on, the hidden state between RNNs in subsequences is always initialized to zero.
    # Basically, set this flag on if you want to reset all RNN hidden states between 'max_grad_steps' time steps
    state['reset_hidden_states_between_subsequences'] = False

    # If on, will use NCE (Noise-Contrastive Estimation) to train model.
    # This is significantly faster for large vocabularies (e.g. more than 20K words), 
    # but experiments show that this degrades performance.
    state['use_nce'] = False
    # If on, the maxout activation function will be applied to the utterance decoders output unit.
    # This requires qdim_decoder = 2x rankdim
    state['maxout_out'] = False
    # If on, a two-layer MLPs will applied on the utterance decoder hidden state before 
    # outputting the distribution over words.
    state['deep_out'] = True

    # If on, there will be an extra MLP between utterance and dialogue encoder
    state['deep_dialogue_input'] = False

    # ----- ACTIVATION FUNCTIONS ---- 
    # Default and recommended setting is: tanh.
    # The utterance encoder and utterance decoder activation function
    state['sent_rec_activation'] = 'lambda x: T.tanh(x)'
    # The dialogue encoder activation function
    state['dialogue_rec_activation'] = 'lambda x: T.tanh(x)'
    
    # Determines how to input the utterance encoder and dialogue encoder into the utterance decoder RNN hidden state:
    #  - 'first': initializes first hidden state of decoder using encoders
    #  - 'all': initializes first hidden state of decoder using encoders, 
    #            and inputs all hidden states of decoder using encoders
    #  - 'selective': initializes first hidden state of decoder using encoders, 
    #                 and inputs all hidden states of decoder using encoders.
    #                 Furthermore, a gating function is applied to the encoder input 
    #                 to turn off certain dimensions if necessary.
    #
    # Experiments show that 'all' is most effective.
    state['decoder_bias_type'] = 'all' 

    # Define the gating function for the three RNNs.
    state['utterance_encoder_gating'] = 'GRU' # Supports 'None' and 'GRU'
    state['dialogue_encoder_gating'] = 'GRU' # Supports 'None' and 'GRU'
    state['utterance_decoder_gating'] = 'GRU' # Supports 'None', 'GRU' and 'LSTM'

    # If on, two utterances encoders (one forward and one backward) will be used, otherwise only a forward utterance encoder is used
    state['bidirectional_utterance_encoder'] = False

    # If on, there will be a direct connection between utterance encoder and utterane decoder RNNs
    state['direct_connection_between_encoders_and_decoder'] = False

    # If on, there will be an extra MLP between utterance encoder and utterance decoder
    state['deep_direct_connection'] = False

    # If on, the model will collaps to a standard RNN:
    # 1) The utterance+dialogue encoder input to the utterance decoder will be zero.
    # 2) The utterance decoder will never be reset
    # Note this model will always be initialized with a hidden state equal to zero
    state['collaps_to_standard_rnn'] = False

    # If on, the utterance decoder will never be reset. 
    # If off, the utterance decoder will be reset at the end of every utterance.
    #state['never_reset_decoder'] = False

    # If on, the utterance decoder will be reset after each end-of-utterance token
    # This replaces the previous configuration variable 'never_reset_decoder', by being
    # its opposite (negation).
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    # If on, the utterance encoder will be reset after each end-of-utterance token
    state['reset_utterance_encoder_at_end_of_utterance'] = True

    # ----- SIZES ----
    # Dimensionality of hidden layers
    # Dimensionality of (word-level) utterance encoder hidden state
    state['qdim_encoder'] = 512
    # Dimensionality of (word-level) utterance decoder (RNN which generates output) hidden state
    state['qdim_decoder'] = 512
    # Dimensionality of (utterance-level) dialogue hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank word embedding approximation
    state['rankdim'] = 256

    # ----- LATENT VARIABLES WITH VARIATIONAL LEARNING -----
    # If on, a Gaussian latent variable is added at the beginning of each utterance.
    # The utterance decoder will be conditioned on this latent variable,
    # and training will be done using the variational lower bound. 
    # See, for example, the variational auto-encoder by Kingma et al.
    state['add_latent_gaussian_per_utterance'] = False
    # If on, will condition the latent variable on the dialogue encoder
    state['condition_latent_variable_on_dialogue_encoder'] = False
    # If on, will condition the latent variable on the DCGM (mean pooling over words) encoder.
    # This will replace the conditioning on the utterance encoder.
    state['condition_latent_variable_on_dcgm_encoder'] = False
    # Dimensionality of Gaussian latent variable (under the assumption that covariance matrix is diagonal)
    state['latent_gaussian_per_utterance_dim'] = 10
    # If on, the latent Gaussian variable at time t will be affected linearly by the distribution (sufficient statistics) of the latent variable at time t-1. This is different from an actual linear state space model (Kalman filter), since effective latent variables at time t are independent of all other latent variables, given the observed utterances. However, it's useful, because it avoids forward propagating noise (which would make the training procedure more difficult than it already is!).
    state['latent_gaussian_linear_dynamics'] = False


    # If on, batch normalization will be applied to the MLP computing the prior and posterior of the latent variable.
    # The normalization is done both w.r.t. to the batch and temporal dimension. 
    # THIS SEEMS TO WORK TERRIBLE. I RECOMMEND TO DISABLE THIS AND NOT TRAIN WITH BATCH NORMALIZATION.
    state['train_latent_gaussians_with_batch_normalization'] = False
    # The (diagonal) covariance matrix is scaled by this value.
    # Initial diagonal covariance matrix will be a softplus function times this value.
    # By setting it to a high number (e.g. 1 or 10), the KL divergence will be relatively low at the beginning of
    # training.
    state['scale_latent_variable_variances'] = 10
    # If on, the utterance decoder will ONLY be conditioned on the Gaussian latent variable.
    state['condition_decoder_only_on_latent_variable'] = False

    # If on, the KL-divergence term weight for the Gaussian latent variable will be annealed from zero to one.
    state['train_latent_gaussians_with_kl_divergence_annealing'] = False
    # The KL-divergence term weight is increased by this amount for every training batch.
    # It is truncated to one. For example, 1.0/60000.0 means that at iteration 60000 the model
    # will assign weight one to the KL-divergence term (and thus be maximizing the true variational bound).
    state['kl_divergence_annealing_rate'] = 1.0/60000.0

    # If enabled, the previous token input to the decoder RNN is replaced with the 'unk' token at random.
    state['decoder_drop_previous_input_tokens'] = False
    # The rate at which the previous token input to the decoder is kept (i.e. not set to 'unk').
    # Setting this to zero effectively disables teacher-forcing in the model.
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    # If enabled, the log-likelihood of each token will be multiplied by (1 + beta * weight(w)) / max(1, beta), 
    # where weight(w) are weights of each words given by the values in the dictionary 
    # 'weight_token_loglikelihoods_dictionary'. Setting beta=10.0 appears to work.
    state['weight_token_loglikelihoods'] = False
    state['weight_token_loglikelihoods_dictionary'] = {}
    state['weight_token_loglikelihoods_beta'] = 10.0

    # Initialization configuration
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = ''
    state['fix_pretrained_word_embeddings'] = False

    # If on, will fix the parameters of the utterance encoder and dialogue encoder RNNs,
    # as well as the word embeddings. NOT APPLICABLE when collaps_to_standard_rnn is on.
    state['fix_encoder_parameters'] = False

    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    # Learning rate. The rate 0.0002 seems to work well across many tasks with adam.
    # Alternatively, the learning rate can be adjusted down (e.g. 0.00004) 
    # to at the end of training to help the model converge well.
    state['lr'] = 0.0002

    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003

    # Batch size
    state['bs'] = 80
    # Sort by length groups of  
    state['sort_k_batches'] = 20
   
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

def prototype_test():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    # Paths for semantic information 
    state['train_semantic'] = "./tests/data/ttrain.semantic.pkl"
    state['test_semantic'] = "./tests/data/ttest.semantic.pkl"
    state['valid_semantic'] = "./tests/data/tvalid.semantic.pkl"
    state['semantic_information_dim'] = 2

    # If secondary_train_dialogues is specified the model will simultaneously be trained on a second dataset.
    # Each batch (document) will be chosen from the secondary dataset with probability secondary_proportion.
    #state['secondary_train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    #state['secondary_proportion'] = 0.5

    state['max_grad_steps'] = 20
    
    # Handle bleu evaluation
    state['bleu_evaluation'] = "./tests/bleu/bleu_evaluation"
    state['bleu_context_length'] = 2

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    state['fix_pretrained_word_embeddings'] = False
    
    # Validation frequency
    state['valid_freq'] = 50

    state['collaps_to_standard_rnn'] = False
    
    # Variables
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_out'] = True
    state['deep_dialogue_input'] = True

    state['utterance_encoder_gating'] = 'GRU'
    state['dialogue_encoder_gating'] = 'GRU'
    state['utterance_decoder_gating'] = 'GRU'
    state['bidirectional_utterance_encoder'] = True 
    state['direct_connection_between_encoders_and_decoder'] = True

    # If out of memory, modify this!
    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all' # 'none', 'all' or 'selective' 
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 10
    # Dimensionality of low-rank approximation
    state['rankdim'] = 10
    return state

def prototype_test_variational():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    # Paths for semantic information 
    state['train_semantic'] = "./tests/data/ttrain.semantic.pkl"
    state['test_semantic'] = "./tests/data/ttest.semantic.pkl"
    state['valid_semantic'] = "./tests/data/tvalid.semantic.pkl"
    state['semantic_information_dim'] = 2

    state['max_grad_steps'] = 20
    
    # Handle bleu evaluation
    #state['bleu_evaluation'] = "./tests/bleu/bleu_evaluation"
    #state['bleu_context_length'] = 2

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    state['fix_pretrained_word_embeddings'] = True
    
    # Validation frequency
    state['valid_freq'] = 50

    state['collaps_to_standard_rnn'] = False
    
    # Variables
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_out'] = True
    state['deep_dialogue_input'] = True
    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = True

    state['utterance_encoder_gating'] = 'GRU'
    state['dialogue_encoder_gating'] = 'GRU'
    state['utterance_decoder_gating'] = 'GRU'

    state['bidirectional_utterance_encoder'] = True
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 5
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['condition_latent_variable_on_dcgm_encoder'] = True
    state['train_latent_gaussians_with_batch_normalization'] = False
    state['train_latent_gaussians_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['latent_gaussian_linear_dynamics'] = True

    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    # If out of memory, modify this!
    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all' # 'none', 'all' or 'selective' 
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 10
    # Dimensionality of low-rank approximation
    state['rankdim'] = 10
    return state


def prototype_movies():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "Data/Training.dialogues.pkl"
    state['test_dialogues'] = "Data/Test.dialogues.pkl"
    state['valid_dialogues'] = "Data/Validation.dialogues.pkl"
    state['dictionary'] = "Data/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    # If secondary_train_dialogues is specified the model will simultaneously be trained on a second dataset.
    # Each batch (document) will be chosen from the secondary dataset with probability secondary_proportion.
    #state['secondary_train_dialogues'] = "Data/OpenSubtitles.dialogues.pkl"
    #state['secondary_proportion'] = 0.5

    # Paths for semantic information.
    # The genre labels are incorrect right now...
    #state['train_semantic'] = "Data/Training.genres.pkl"
    #state['test_semantic'] = "Data/Test.genres.pkl"
    #state['valid_semantic'] = "Data/Validation.genres.pkl"
    #state['semantic_information_dim'] = 16

    # Gradients will be truncated after 80 steps. This seems like a fair start.
    state['max_grad_steps'] = 80

    # Handle bleu evaluation
    #state['bleu_evaluation'] = "Data/Mini_Validation_Shuffled_Dataset.txt"
    #state['bleu_context_length'] = 2

    # Handle pretrained word embeddings.
    # These need to be recomputed if we want them for the 20K vocabulary.
    #state['initialize_from_pretrained_word_embeddings'] = True
    #state['pretrained_word_embeddings_file'] = 'Data/Word2Vec_Emb.pkl' 
    #state['fix_pretrained_word_embeddings'] = True
    
    # Validation frequency
    state['valid_freq'] = 2500
    
    state['prefix'] = "MovieScriptModel_" 
    state['updater'] = 'adam'
    
    # Model architecture
    state['bidirectional_utterance_encoder'] = True
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 20
    state['deep_dialogue_input'] = True
    state['deep_out'] = True
 
    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective' 

    state['qdim_encoder'] = 600
    state['qdim_decoder'] = 600
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 300
    # Dimensionality of low-rank approximation
    state['rankdim'] = 300

    # If enabled, the log-likelihood of each token will be multiplied by the values found in the dictionary 
    # given by the state variable 'weight_token_loglikelihoods_dictionary'
    state['weight_token_loglikelihoods'] = False
    state['weight_token_loglikelihoods_dictionary'] = {}

    return state



def prototype_twitter():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    # Gradients will be truncated after 80 steps. This seems like a fair start.
    state['max_grad_steps'] = 80
    
    # Validation frequency
    state['valid_freq'] = 5000
    
    state['prefix'] = "TwitterModel_" 
    state['updater'] = 'adam'
    
    # Model architecture
    state['bidirectional_utterance_encoder'] = True
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 20
    state['deep_dialogue_input'] = True
    state['deep_out'] = True
 
    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective' 
    state['direct_connection_between_encoders_and_decoder'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0001


    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank approximation
    state['rankdim'] = 500

    return state

def prototype_twitter_lstm():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    # Gradients will be truncated after 80 steps. This seems like a fair start.
    state['max_grad_steps'] = 80
    
    # Validation frequency
    state['valid_freq'] = 5000
    
    state['prefix'] = "TwitterModel_" 
    state['updater'] = 'adam'
    
    # Model architecture
    state['bidirectional_utterance_encoder'] = False
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 20
    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['collaps_to_standard_rnn'] = True
 
    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective' 
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0001


    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 10
    # Dimensionality of low-rank approximation
    state['rankdim'] = 400

    return state

def prototype_twitter_variational():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    # Gradients will be truncated after 80 steps. This seems like a fair start.
    state['max_grad_steps'] = 80
    
    # Validation frequency
    state['valid_freq'] = 5000
    
    state['prefix'] = "TwitterModel_" 
    state['updater'] = 'adam'
    
    # Model architecture
    state['bidirectional_utterance_encoder'] = True
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 20
    state['deep_dialogue_input'] = True
    state['deep_out'] = True
 
    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective' 
    state['direct_connection_between_encoders_and_decoder'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = False

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_variable_variances'] = 0.1
    state['lr'] = 0.0001
    state['train_latent_gaussians_with_batch_normalization'] = False
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['condition_latent_variable_on_dcgm_encoder'] = True
    state['condition_decoder_only_on_latent_variable'] = True
    state['train_latent_gaussians_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank approximation
    state['rankdim'] = 400

    return state


def prototype_weighted_movies():
    state = prototype_state()

    # Fill your paths here!
    # Fill your paths here!
    #state['train_dialogues'] = "../Data/Training.dialogues.pkl"
    state['train_dialogues'] = "../Data/OpenSubtitles.dialogues.pkl"
    state['test_dialogues'] = "../Data/Test.dialogues.pkl"
    state['valid_dialogues'] = "../Data/Validation.dialogues.pkl"
    state['dictionary'] = "../Data/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    # Gradients will be truncated after 80 steps. This seems like a fair start.
    state['max_grad_steps'] = 80

    # Validation frequency
    state['valid_freq'] = 5000

    state['prefix'] = "HREDWeightedModel_"
    state['updater'] = 'adam'

    # Model architecture
    state['bidirectional_utterance_encoder'] = True
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 20
    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0001


    state['qdim_encoder'] = 2000
    state['qdim_decoder'] = 4000
    # Dimensionality of dialogue hidden layer
    state['sdim'] = 1000
    # Dimensionality of low-rank approximation
    state['rankdim'] = 500

    state['weight_token_loglikelihoods'] = True
    state['weight_token_loglikelihoods_dictionary'] = cPickle.load(open('../Data/wrd2weight.pkl', 'r'))

    return state

def prototype_weighted_twitter():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    # Gradients will be truncated after 80 steps. This seems like a fair start.
    state['max_grad_steps'] = 80
    
    # Validation frequency
    state['valid_freq'] = 5000
    
    state['prefix'] = "TwitterModel_" 
    state['updater'] = 'adam'
    
    # Model architecture
    state['bidirectional_utterance_encoder'] = True
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 20
    state['deep_dialogue_input'] = True
    state['deep_out'] = True
 
    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective' 
    state['direct_connection_between_encoders_and_decoder'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0001


    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank approximation
    state['rankdim'] = 400


    state['weight_token_loglikelihoods'] = True
    state['weight_token_loglikelihoods_dictionary'] = cPickle.load(open('../TwitterData/wrd2weight.pkl', 'r'))

    return state
