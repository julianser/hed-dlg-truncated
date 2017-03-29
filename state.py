from collections import OrderedDict
import cPickle
import os

def prototype_state():
    state = {} 

    # ----- CONSTANTS -----
    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    # Out-of-vocabulary token string
    state['oov'] = '<unk>'
    
    # These are end-of-sequence marks
    state['end_sym_utterance'] = '</s>'

    # Special tokens need to be defined here, because model architecture may adapt depending on these
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


    # ----- MODEL ARCHITECTURE -----
    # If this flag is on, the hidden state between RNNs in subsequences is always initialized to zero.
    # Set this to reset all RNN hidden states between 'max_grad_steps' time steps
    state['reset_hidden_states_between_subsequences'] = False

    # If this flag is on, the maxout activation function will be applied to the utterance decoders output unit.
    # This requires qdim_decoder = 2x rankdim
    state['maxout_out'] = False

    # If this flag is on, a two-layer MLPs will applied on the utterance decoder hidden state before 
    # outputting the distribution over words.
    state['deep_out'] = True

    # If this flag is on, there will be an extra MLP between utterance and dialogue encoder
    state['deep_dialogue_input'] = False

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

    # If this flag is on, two utterances encoders (one forward and one backward) will be used,
    # otherwise only a forward utterance encoder is used.
    state['bidirectional_utterance_encoder'] = False

    # If this flag is on, there will be a direct connection between utterance encoder and utterane decoder RNNs.
    state['direct_connection_between_encoders_and_decoder'] = False

    # If this flag is on, there will be an extra MLP between utterance encoder and utterance decoder.
    state['deep_direct_connection'] = False

    # If this flag is on, the model will collaps to a standard RNN:
    # 1) The utterance+dialogue encoder input to the utterance decoder will be zero
    # 2) The utterance decoder will never be reset
    # Note this model will always be initialized with a hidden state equal to zero.
    state['collaps_to_standard_rnn'] = False

    # If this flag is on, the utterance decoder will be reset after each end-of-utterance token.
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    # If this flag is on, the utterance encoder will be reset after each end-of-utterance token.
    state['reset_utterance_encoder_at_end_of_utterance'] = True


    # ----- HIDDEN LAYER DIMENSIONS -----
    # Dimensionality of (word-level) utterance encoder hidden state
    state['qdim_encoder'] = 512
    # Dimensionality of (word-level) utterance decoder (RNN which generates output) hidden state
    state['qdim_decoder'] = 512
    # Dimensionality of (utterance-level) context encoder hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank word embedding approximation
    state['rankdim'] = 256


    # ----- LATENT VARIABLES WITH VARIATIONAL LEARNING -----
    # If this flag is on, a Gaussian latent variable is added at the beginning of each utterance.
    # The utterance decoder will be conditioned on this latent variable,
    # and the model will be trained using the variational lower bound. 
    # See, for example, the variational auto-encoder by Kingma et al. (2013).
    state['add_latent_gaussian_per_utterance'] = False
    # This flag will condition the latent variable on the dialogue encoder
    state['condition_latent_variable_on_dialogue_encoder'] = False
    # This flag will condition the latent variable on the DCGM (mean pooling over words) encoder.
    # This will replace the conditioning on the utterance encoder.
    # If the flag is false, the latent variable will be conditioned on the utterance encoder RNN.
    state['condition_latent_variable_on_dcgm_encoder'] = False
    # Dimensionality of Gaussian latent variable, which has diagonal covariance matrix.
    state['latent_gaussian_per_utterance_dim'] = 10
    # If this flag is on, the latent Gaussian variable at time t will be affected linearly
    # by the distribution (sufficient statistics) of the latent variable at time t-1.
    # This is different from an actual linear state space model (Kalman filter),
    # since effective latent variables at time t are independent of all other latent variables,
    # given the observed utterances. However, it's useful, because it avoids forward propagating noise
    # which would make the training procedure more difficult than it already is.
    # Although it has nice properties (matrix preserves more information since it is full rank, and if its eigenvalues are all positive the linear dynamics are just rotations in space),
    # it appears to make training very unstable!
    state['latent_gaussian_linear_dynamics'] = False

    # This is a constant by which the diagonal covariance matrix is scaled.
    # By setting it to a high number (e.g. 1 or 10),
    # the KL divergence will be relatively low at the beginning of training.
    state['scale_latent_variable_variances'] = 10
    # If this flag is on, the utterance decoder will ONLY be conditioned on the Gaussian latent variable.
    state['condition_decoder_only_on_latent_variable'] = False

    # If this flag is on, the KL-divergence term weight for the Gaussian latent variable
    # will be slowly increased from zero to one.
    state['train_latent_gaussians_with_kl_divergence_annealing'] = False
    # The KL-divergence term weight is increased by this parameter for every training batch.
    # It is truncated to one. For example, 1.0/60000.0 means that at iteration 60000 the model
    # will assign weight one to the KL-divergence term
    # and thus only be maximizing the true variational bound from iteration 60000 and onward.
    state['kl_divergence_annealing_rate'] = 1.0/60000.0

    # If this flag is enabled, previous token input to the decoder RNN is replaced with 'unk' tokens at random.
    state['decoder_drop_previous_input_tokens'] = False
    # The rate at which the previous tokesn input to the decoder is kept (not set to 'unk').
    # Setting this to zero effectively disables teacher-forcing in the model.
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    # Initialization configuration
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = ''
    state['fix_pretrained_word_embeddings'] = False

    # If this flag is on, the model will fix the parameters of the utterance encoder and dialogue encoder RNNs,
    # as well as the word embeddings. NOTE: NOT APPLICABLE when the flag 'collaps_to_standard_rnn' is on.
    state['fix_encoder_parameters'] = False


    # ----- TRAINING PROCEDURE -----
    # Choose optimization algorithm (adam works well most of the time)
    state['updater'] = 'adam'
    # If this flag is on, NCE (Noise-Contrastive Estimation) will be used to train model.
    # This is significantly faster for large vocabularies (e.g. more than 20K words), 
    # but experiments show that this degrades performance.
    state['use_nce'] = False
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    # Learning rate. The rate 0.0002 seems to work well across many tasks with adam.
    # Alternatively, the learning rate can be adjusted down (e.g. 0.00004) 
    # to at the end of training to help the model converge well.
    state['lr'] = 0.0002
    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003
    # Batch size. If out of memory, modify this!
    state['bs'] = 80
    # Sort by length groups of  
    state['sort_k_batches'] = 20
    # Training examples will be split into subsequences.
    # This parameter controls the maximum size of each subsequence.
    # Gradients will be computed on the subsequence, and the last hidden state of all RNNs will
    # be used to initialize the hidden state of the RNNs in the next subsequence.
    state['max_grad_steps'] = 80
    # Modify this in the prototype
    state['save_dir'] = './'
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
    
    # Fill paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    state['max_grad_steps'] = 20
    
    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    state['fix_pretrained_word_embeddings'] = False
    
    state['valid_freq'] = 50

    state['collaps_to_standard_rnn'] = False
    
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

    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all'
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    state['sdim'] = 10
    state['rankdim'] = 10

    return state

def prototype_test_variational():
    state = prototype_state()
    
    # Fill paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    state['max_grad_steps'] = 20

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    state['fix_pretrained_word_embeddings'] = True
    
    state['valid_freq'] = 5

    state['collaps_to_standard_rnn'] = False
    
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
    state['condition_latent_variable_on_dcgm_encoder'] = False
    state['train_latent_gaussians_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['latent_gaussian_linear_dynamics'] = True

    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all'
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    state['sdim'] = 10
    state['rankdim'] = 10

    return state


# Twitter LSTM RNNLM model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016).
def prototype_twitter_lstm():
    state = prototype_state()
    
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    state['max_grad_steps'] = 80
    
    state['valid_freq'] = 5000
    
    state['prefix'] = "TwitterModel_" 
    state['updater'] = 'adam'
    
    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['collaps_to_standard_rnn'] = True
 
    state['bs'] = 80 
    state['decoder_bias_type'] = 'all'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0001


    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 400

    return state




# Twitter HRED model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016).
# Note, similar or better performance might be reached by setting:
#
#   state['decoder_bias_type'] = 'all'
#   state['reset_utterance_encoder_at_end_of_utterance'] = True
#   state['utterance_decoder_gating'] = 'LSTM'
#
def prototype_twitter_HRED():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0001

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    return state


# Twitter HRED model, where context biases decoder using standard MLP.
def prototype_twitter_HRED_StandardBias():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = True
    state['lr'] = 0.0002

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    return state



# Twitter VHRED model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016). Note, this model was pretrained as the HRED model with state 'prototype_twitter_HRED'!
def prototype_twitter_VHRED():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True



    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0001

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'


    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100


    state['scale_latent_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_gaussians_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


# Twitter VHRED model, where context biases decoder using standard MLP.
# Note, this model should be pretrained as HRED model.
def prototype_twitter_VHRED_StandardBias():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True



    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = True
    state['lr'] = 0.0002

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'


    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100


    state['scale_latent_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_gaussians_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


# Ubuntu LSTM RNNLM model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016).
def prototype_ubuntu_LSTM():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['collaps_to_standard_rnn'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['lr'] = 0.0002

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM' # Supports 'None', 'GRU' and 'LSTM'

    return state



# Ubuntu HRED model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016).
def prototype_ubuntu_HRED():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['bs'] = 80

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = True
    state['utterance_decoder_gating'] = 'LSTM'

    state['lr'] = 0.0002

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 1000
    state['rankdim'] = 300

    return state



# Ubuntu VHRED model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016). Note, this model was pretrained as the HRED model with state 'prototype_ubuntu_HRED'!
def prototype_ubuntu_VHRED():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_input'] = True
    state['deep_out'] = True

    state['bs'] = 80

    state['reset_utterance_decoder_at_end_of_utterance'] = True
    state['reset_utterance_encoder_at_end_of_utterance'] = True
    state['utterance_decoder_gating'] = 'LSTM'

    state['lr'] = 0.0002

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 1000
    state['rankdim'] = 300

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_gaussians_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state

