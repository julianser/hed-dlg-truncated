"""
Dialog hierarchical encoder-decoder code.
The code is inspired from nmt encdec code in groundhog
but we do not rely on groundhog infrastructure.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni, Iulian Vlad Serban")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
logger = logging.getLogger(__name__)

from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv3d2d import *
from collections import OrderedDict

from model import *
from utils import *

import operator

# Theano speed-up
theano.config.scan.allow_gc = False
#

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

class EncoderDecoderBase():
    def __init__(self, state, rng, parent):
        self.rng = rng
        self.parent = parent
        
        self.state = state
        self.__dict__.update(state)
        
        self.dialogue_rec_activation = eval(self.dialogue_rec_activation)
        self.sent_rec_activation = eval(self.sent_rec_activation)
         
        self.params = []

class UtteranceEncoder(EncoderDecoderBase):
    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in'+self.name))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh'+self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_hh'+self.name))
        
        if self.sent_step_type == "gated":
            self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r'+self.name))
            self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z'+self.name))
            self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_r'+self.name))
            self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_z'+self.name))
            self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_z'+self.name))
            self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_r'+self.name))

    def approx_embedder(self, x):
        return self.W_emb[x]

    def plain_sent_step(self, x_t, m_t, *args):
        do_pool = False
        if len(args) > 1:
            do_pool = True

        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        hr_tm1 = m_t * h_tm1
        h_t = self.sent_rec_activation(T.dot(x_t, self.W_in) + T.dot(hr_tm1, self.W_hh) + self.b_hh)

        if do_pool:
            # perform pooling with a moving average
            h_tm1_pooled = next(args)
            pool_len = next(args)

            new_pool_len = pool_len * m_t + 1
            h_t_pooled = ((new_pool_len - 1) * h_tm1_pooled + h_t**2) / new_pool_len

            # return state and pooled state
            return [h_t, h_t_pooled, new_pool_len]
        else:
            # Return hidden state only
            return [h_t]

    def gated_sent_step(self, x_t, m_t, *args):
        do_pool = False
        if len(args) >= 3:
            do_pool = True

        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x') 
         
        hr_tm1 = m_t * h_tm1

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(hr_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(hr_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.sent_rec_activation(T.dot(x_t, self.W_in) + T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
        h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
        
        if do_pool:
            # perform pooling with a moving average
            h_tm1_pooled = next(args)
            pool_len = next(args)

            new_pool_len = pool_len * m_t + 1
            h_t_pooled = ((new_pool_len - 1) * h_tm1_pooled + h_t**2) / new_pool_len

            # return both reset state and non-reset state
            return [h_t, h_t_pooled, new_pool_len, r_t, z_t, h_tilde]
        else:
            # return both reset state and non-reset state
            return [h_t, r_t, z_t, h_tilde]

    def build_encoder(self, x, xmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1

        # if it is not one_step then we initialize everything to previous state or zero  
        if not one_step:
            if prev_state:
                h_0 = prev_state
            else:
                h_0 = T.alloc(np.float32(0), batch_size, self.qdim)

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_h' in kwargs 
            h_0 = kwargs['prev_h']

        xe = self.approx_embedder(x)
        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        # We add ones at the the beginning of the reset vector to align the resets with y_training:
        # for example for 
        # training_x =        </s> a b c </s> d
        # xmask =               0  1 1 1  0   1
        # rolled_xmask =        1  0 1 1  1   0 1
        # Thus, we ensure that the no information in the encoder is carried from input "</s>" to "a",
        # or from "</s>" to "d". 
        # Now, the state at exactly </s> always reflects the previous utterance encoding.
        # Since the dialogue encoder uses xmask, and inputs it when xmask=0, it will input the utterance encoding
        # exactly on the </s> state.

        if xmask.ndim == 2:
            #ones_vector = theano.shared(value=numpy.ones((1, self.bs), dtype='float32'), name='ones_vector')
            ones_vector = T.ones_like(xmask[0,:]).dimshuffle('x', 0)
            rolled_xmask = T.concatenate([ones_vector, xmask], axis=0)
        else:
            ones_scalar = theano.shared(value=numpy.ones((1), dtype='float32'), name='ones_scalar')
            rolled_xmask = T.concatenate([ones_scalar, xmask])


        # Gated Encoder
        if self.sent_step_type == "gated":
            f_enc = self.gated_sent_step
            o_enc_info = [h_0, None, None, None]

        else:
            f_enc = self.plain_sent_step
            o_enc_info = [h_0]


        # Run through all the sentence (encode everything)
        if not one_step: 
            _res, _ = theano.scan(f_enc,
                              sequences=[xe, rolled_xmask],\
                              outputs_info=o_enc_info)
        else: # Make just one step further
            _res = f_enc(xe, rolled_xmask, [h_0])[0]

        # Get the hidden state sequence
        h = _res[0]
        return h

    def __init__(self, state, rng, word_embedding_param, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(word_embedding_param)


class DialogEncoder(EncoderDecoderBase):
    def init_params(self):
        """ Context weights """

        if self.bidirectional_utterance_encoder:
            # With the bidirectional flag, the dialog encoder gets input 
            # from both the forward and backward utterance encoders, hence it is double qdim
            input_dim = self.qdim * 2
        else:
            # Without the bidirectional flag, the dialog encoder only gets input
            # from the forward utterance encoder, which has dim self.qdim
            input_dim = self.qdim

        transformed_input_dim = input_dim
        if self.deep_dialogue_input:
            self.Ws_deep_input = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, input_dim, self.sdim), name='Ws_deep_input'+self.name))
            self.bs_deep_input = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_deep_input'+self.name))
            transformed_input_dim = self.sdim

        
        self.Ws_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in'+self.name))
        self.Ws_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh'+self.name))
        self.bs_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh'+self.name)) 
         


        if self.dialogue_step_type == "gated":
            self.Ws_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_r'+self.name))
            self.Ws_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_z'+self.name))
            self.Ws_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_r'+self.name))
            self.Ws_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_z'+self.name))
            self.bs_z = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_z'+self.name))
            self.bs_r = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_r'+self.name))
    
    def plain_dialogue_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        # If deep input to dialogue encoder is enabled, run h_t through an MLP
        transformed_h_t = h_t
        if self.deep_dialogue_input:
            transformed_h_t = self.dialogue_rec_activation(T.dot(h_t, self.Ws_deep_input) + self.bs_deep_input)

        hs_tilde = self.dialogue_rec_activation(T.dot(transformed_h_t, self.Ws_in) + T.dot(hs_tm1, self.Ws_hh) + self.bs_hh)

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_tilde 
        return hs_t

    def gated_dialogue_step(self, h_t, m_t, hs_tm1):
        # If deep input to dialogue encoder is enabled, run h_t through an MLP
        transformed_h_t = h_t
        if self.deep_dialogue_input:
            transformed_h_t = self.dialogue_rec_activation(T.dot(h_t, self.Ws_deep_input) + self.bs_deep_input)

        rs_t = T.nnet.sigmoid(T.dot(transformed_h_t, self.Ws_in_r) + T.dot(hs_tm1, self.Ws_hh_r) + self.bs_r)
        zs_t = T.nnet.sigmoid(T.dot(transformed_h_t, self.Ws_in_z) + T.dot(hs_tm1, self.Ws_hh_z) + self.bs_z)
        hs_tilde = self.dialogue_rec_activation(T.dot(transformed_h_t, self.Ws_in) + T.dot(rs_t * hs_tm1, self.Ws_hh) + self.bs_hh)
        hs_update = (np.float32(1.) - zs_t) * hs_tm1 + zs_t * hs_tilde
         
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_update
        return hs_t, hs_tilde, rs_t, zs_t

    def build_encoder(self, h, x, xmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            if prev_state:
                hs_0 = prev_state
            else:
                hs_0 = T.alloc(np.float32(0), batch_size, self.sdim)

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)       

        if self.dialogue_step_type == "gated":
            f_hier = self.gated_dialogue_step
            o_hier_info = [hs_0, None, None, None]
        else:
            f_hier = self.plain_dialogue_step
            o_hier_info = [hs_0]
        
        # All hierarchical sentence
        # The hs sequence is based on the original mask
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[h, xmask],\
                               outputs_info=o_hier_info)
        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        return hs 

    def __init__(self, state, rng, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params()

class DialogDummyEncoder(EncoderDecoderBase):  
    # This dialogue encoder behaves like a DialogEncoder with the identity function.
    # At the end of each utterance, the input from the utterance encoder(s) is transferred
    # to its hidden state, which can then be transfered to the decoder.

    def init_params(self):
        """ Context weights """
        if self.deep_direct_connection:
            self.Ws_dummy_deep_input = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.inp_dim, self.inp_dim), name='Ws_dummy_deep_input'+self.name))
            self.bs_dummy_deep_input = add_to_params(self.params, theano.shared(value=np.zeros((self.inp_dim,), dtype='float32'), name='bs_dummy_deep_input'+self.name))


    def plain_dialogue_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        transformed_h_t = h_t
        if self.deep_direct_connection:
            transformed_h_t = self.dialogue_rec_activation(T.dot(h_t, self.Ws_dummy_deep_input) + self.bs_dummy_deep_input)

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * transformed_h_t 
        return hs_t

    def build_encoder(self, h, x, xmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            if prev_state:
                hs_0 = prev_state
            else:
                hs_0 = T.alloc(np.float32(0), batch_size, self.inp_dim) 

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        f_hier = self.plain_dialogue_step
        o_hier_info = [hs_0]
        
        # All hierarchical sentence
        # The hs sequence is based on the original mask
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[h, xmask],\
                               outputs_info=o_hier_info)
        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        return hs 

    def __init__(self, state, rng, parent, inp_dim, name=''):
        self.inp_dim = inp_dim
        self.name = name
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.init_params()



class UtteranceDecoder(EncoderDecoderBase):
    NCE = 0
    EVALUATION = 1
    SAMPLING = 2
    BEAM_SEARCH = 3

    def __init__(self, state, rng, parent, dialog_encoder, word_embedding_param):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        # Take as input the encoder instance for the embeddings..
        # To modify in the future
        assert(word_embedding_param != None)
        self.word_embedding_param = word_embedding_param
        self.dialog_encoder = dialog_encoder
        self.trng = MRG_RandomStreams(self.seed)
        self.init_params()

    def init_params(self): 
        if self.direct_connection_between_encoders_and_decoder:
            # When there is a direct connection between encoder and decoder, 
            # the input has dimensionality sdim + qdim if forward encoder, and
            # sdim + 2 x qdim for bidirectional encoder
            if self.bidirectional_utterance_encoder:
                self.input_dim = self.sdim + self.qdim*2
            else:
                self.input_dim = self.sdim + self.qdim
        else:
            # When there is no connection between encoder and decoder, 
            # the input has dimensionality sdim
            self.input_dim = self.sdim

        """ Decoder weights """
        self.bd_out = add_to_params(self.params, theano.shared(value=np.zeros((self.idim,), dtype='float32'), name='bd_out'))
        self.Wd_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='Wd_emb'))

        self.Wd_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='Wd_hh'))
        self.bd_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_hh'))
        self.Wd_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in')) 

        # We only include the initial hidden state if the utterance decoder is NOT reset 
        # and if its NOT a collapsed model (i.e. collapsed to standard RNN). 
        # In the collapsed model, we always initialize hidden state to zero.
        if (not self.collaps_to_standard_rnn) and (not self.never_reset_decoder):
            self.Wd_s_0 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim), name='Wd_s_0'))
            self.bd_s_0 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_s_0'))

        if self.sent_step_type == "gated":
            self.Wd_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in_r'))
            self.Wd_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in_z'))
            self.Wd_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='Wd_hh_r'))
            self.Wd_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='Wd_hh_z'))
            self.bd_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_r'))
            self.bd_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_z'))
        
            if self.decoder_bias_type == 'all':
                self.Wd_s_q = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim), name='Wd_s_q'))
                self.Wd_s_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim), name='Wd_s_z'))
                self.Wd_s_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim), name='Wd_s_r')) 

        if self.decoder_bias_type == 'selective':
            self.bd_sel = add_to_params(self.params, theano.shared(value=np.zeros((self.input_dim,), dtype='float32'), name='bd_sel'))
            self.Wd_s_q = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim), name='Wd_s_q'))
            # s -> g_r
            self.Wd_sel_s = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.input_dim, self.input_dim), \
                                                        name='Wd_sel_s'))
            # x_{n-1} -> g_r
            self.Wd_sel_e = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.rankdim, self.input_dim), \
                                                        name='Wd_sel_e'))
            # h_{n-1} -> g_r
            self.Wd_sel_h = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.qdim, self.input_dim), \
                                                        name='Wd_sel_h'))
         
        ######################   
        # Output layer weights
        ######################
        if self.maxout_out:
            if int(self.qdim) != 2*int(self.rankdim):
                raise ValueError('Error with maxout configuration in UtteranceDecoder!'
                                 + 'For maxout to work we need qdim = 2x rankdim')

        out_target_dim = self.qdim
        if not self.maxout_out:
            out_target_dim = self.rankdim

        self.Wd_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, out_target_dim), name='Wd_out'))
         
        # Set up deep output
        if self.deep_out:
            self.Wd_e_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, out_target_dim), name='Wd_e_out'))
            self.bd_e_out = add_to_params(self.params, theano.shared(value=np.zeros((out_target_dim,), dtype='float32'), name='bd_e_out'))
             
            if self.decoder_bias_type != 'first': 
                self.Wd_s_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, out_target_dim), name='Wd_s_out'))
   
    def build_output_layer(self, hs, xd, hd):
        pre_activ = T.dot(hd, self.Wd_out)
        
        if self.deep_out:
            pre_activ += T.dot(xd, self.Wd_e_out) + self.bd_e_out
            
            if self.decoder_bias_type != 'first':
                pre_activ += T.dot(hs, self.Wd_s_out)
                # ^ if bias all, bias the deep output
         
        if self.maxout_out:
            pre_activ = Maxout(2)(pre_activ)
         
        return pre_activ

    def build_next_probs_predictor(self, inp, x, prev_state):
        """ 
        Return output probabilities given prev_words x, hierarchical pass hs, and previous hd
        hs should always be the same (and should not be updated).
        """
        return self.build_decoder(inp, x, mode=UtteranceDecoder.BEAM_SEARCH, prev_state=prev_state)

    def approx_embedder(self, x):
        # Here we use the same embeddings learnt in the encoder.. !!!
        return self.word_embedding_param[x]
     
    def output_softmax(self, pre_activ):
        # returns a (timestep, bs, idim) matrix (huge)
        return SoftMax(T.dot(pre_activ, self.Wd_emb.T) + self.bd_out)
    
    def output_nce(self, pre_activ, y, y_hat):
        # returns a (timestep, bs, pos + neg) matrix (very small)
        target_embedding = self.Wd_emb[y]
        # ^ target embedding is (timestep x bs, rankdim)
        noise_embedding = self.Wd_emb[y_hat]
        # ^ noise embedding is (10, timestep x bs, rankdim)
        
        # pre_activ is (timestep x bs x rankdim)
        pos_scores = (target_embedding * pre_activ).sum(2)
        neg_scores = (noise_embedding * pre_activ).sum(3)
 
        pos_scores += self.bd_out[y]
        neg_scores += self.bd_out[y_hat]
         
        pos_noise = self.parent.t_noise_probs[y] * 10
        neg_noise = self.parent.t_noise_probs[y_hat] * 10
        
        pos_scores = - T.log(T.nnet.sigmoid(pos_scores - T.log(pos_noise)))
        neg_scores = - T.log(1 - T.nnet.sigmoid(neg_scores - T.log(neg_noise))).sum(0)
        return pos_scores + neg_scores

    def build_decoder(self, decoder_inp, x, xmask=None, y=None, y_neg=None, mode=EVALUATION, prev_state=None, step_num=None):

        # If model collapses to standard RNN, then reset all input to decoder
        if self.collaps_to_standard_rnn:
            decoder_inp = decoder_inp * 0

        # Check parameter consistency
        if mode == UtteranceDecoder.EVALUATION or mode == UtteranceDecoder.NCE:
            assert y
        else:
            assert not y
            assert prev_state
         
        # if mode == EVALUATION
        #   xd = (timesteps, batch_size, qdim)
        #
        # if mode != EVALUATION
        #   xd = (n_samples, dim)
        xd = self.approx_embedder(x)
        if not xmask:
            xmask = T.neq(x, self.eos_sym)
        
        # we must zero out the </s> embedding
        # i.e. the embedding x_{-1} is the 0 vector
        # as well as hd_{-1} which will be reseted in the scan functions
        if xd.ndim != 3:
            assert mode != UtteranceDecoder.EVALUATION
            xd = (xd.dimshuffle((1, 0)) * xmask).dimshuffle((1, 0))
        else:
            assert mode == UtteranceDecoder.EVALUATION or mode == UtteranceDecoder.NCE
            xd = (xd.dimshuffle((2,0,1)) * xmask).dimshuffle((1,2,0))
        
        # Run the decoder
        #if mode == UtteranceDecoder.EVALUATION or mode == UtteranceDecoder.NCE:
        #    hd_init = T.alloc(np.float32(0), x.shape[1], self.qdim)
        #else:
        #    hd_init = prev_state

        # Run the decoder
        if prev_state:
            hd_init = prev_state
        else:
            hd_init = T.alloc(np.float32(0), x.shape[1], self.qdim)

        if self.sent_step_type == "gated":
            f_dec = self.gated_step
            o_dec_info = [hd_init, None, None, None]
            if self.decoder_bias_type == "selective":
                o_dec_info += [None, None]
        else:
            f_dec = self.plain_step
            o_dec_info = [hd_init]
            if self.decoder_bias_type == "selective":
                o_dec_info += [None, None] 
         
        # If the mode of the decoder is EVALUATION
        # then we evaluate by default all the sentence
        # xd - i.e. xd.ndim == 3, xd = (timesteps, batch_size, qdim)
        if mode == UtteranceDecoder.EVALUATION or mode == UtteranceDecoder.NCE: 
            _res, _ = theano.scan(f_dec,
                              sequences=[xd, xmask, decoder_inp],\
                              outputs_info=o_dec_info)
        # else we evaluate only one step of the recurrence using the
        # previous hidden states and the previous computed hierarchical 
        # states.
        else:
            _res = f_dec(xd, xmask, decoder_inp, prev_state)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hd = _res[0]
        else:
            hd = _res
        
        # if we are using selective bias, we should update our decoder_inp
        # to the step-selective decoder_inp
        step_decoder_inp = decoder_inp
        if self.decoder_bias_type == "selective":
            step_decoder_inp = _res[1]
        pre_activ = self.build_output_layer(step_decoder_inp, xd, hd)

        # EVALUATION  : Return target_probs + all the predicted ranks
        # target_probs.ndim == 3
        if mode == UtteranceDecoder.EVALUATION:
            outputs = self.output_softmax(pre_activ)
            target_probs = GrabProbs(outputs, y)
            return target_probs, hd, _res, outputs 
        elif mode == UtteranceDecoder.NCE:
            return self.output_nce(pre_activ, y, y_neg), hd
        # BEAM_SEARCH : Return output (the softmax layer) + the new hidden states
        elif mode == UtteranceDecoder.BEAM_SEARCH:
            return self.output_softmax(pre_activ), hd
        # SAMPLING    : Return a vector of n_sample from the output layer 
        #                 + log probabilities + the new hidden states
        elif mode == UtteranceDecoder.SAMPLING:
            outputs = self.output_softmax(pre_activ)
            if outputs.ndim == 1:
                outputs = outputs.dimshuffle('x', 0) 
            sample = self.trng.multinomial(pvals=outputs, dtype='int64').argmax(axis=-1)
            if outputs.ndim == 1:
                sample = sample[0] 
            log_prob = -T.log(T.diag(outputs.T[sample])) 
            return sample, log_prob, hd

    def gated_step(self, xd_t, m_t, decoder_inp_t, hd_tm1): 
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        # If model collapses to standard RNN, or the never_reset_decoder flag is on,
        # then never reset decoder. Otherwise, reset the decoder at every utterance turn.
        if (not self.collaps_to_standard_rnn) and (not self.never_reset_decoder):
            hd_tm1 = (m_t) * hd_tm1 + (1 - m_t) * T.tanh(T.dot(decoder_inp_t, self.Wd_s_0) + self.bd_s_0)

        # ^ iff x_{t - 1} = </s> (m_t = 0) then x_{t - 1} = 0
        # and hd_{t - 1} = tanh(W_s_0 decoder_inp_t + bd_s_0) else hd_{t - 1} is left unchanged (m_t = 1)
  
        # In the 'selective' decoder bias type each hidden state of the decoder
        # RNN receives the decoder_inp_t modified by the selective bias -> decoder_inpr_t 
        if self.decoder_bias_type == 'selective':
            rd_sel_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_sel_e) + T.dot(hd_tm1, self.Wd_sel_h) + T.dot(decoder_inp_t, self.Wd_sel_s) + self.bd_sel)
            decoder_inpr_t = rd_sel_t * decoder_inp_t
             
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + self.bd_z)
            hd_tilde = self.sent_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + T.dot(decoder_inpr_t, self.Wd_s_q) \
                                        + self.bd_hh)

            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde 
            output = (hd_t, decoder_inpr_t, rd_sel_t, rd_t, zd_t, hd_tilde)
        
        # In the 'all' decoder bias type each hidden state of the decoder
        # RNN receives the decoder_inp_t vector as bias without modification
        elif self.decoder_bias_type == 'all':
        
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + T.dot(decoder_inp_t, self.Wd_s_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + T.dot(decoder_inp_t, self.Wd_s_z) + self.bd_z)
            hd_tilde = self.sent_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + T.dot(decoder_inp_t, self.Wd_s_q) \
                                        + self.bd_hh)
            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde 
            output = (hd_t, rd_t, zd_t, hd_tilde)
                 
        else:
            # Do not bias all the decoder (force to store very useful information in the first state)
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + self.bd_z)
            hd_tilde = self.sent_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + self.bd_hh) 
            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde
            output = (hd_t, rd_t, zd_t, hd_tilde)
        return output
    
    def plain_step(self, xd_t, m_t, decoder_inp_t, hd_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        # If model collapses to standard RNN, or the never_reset_decoder flag is on,
        # then never reset decoder. Otherwise, reset the decoder at every utterance turn.
        if (not self.collaps_to_standard_rnn) and (not self.never_reset_decoder):
            # We already assume that xd are zeroed out
            hd_tm1 = (m_t) * hd_tm1 + (1-m_t) * T.tanh(T.dot(decoder_inp_t, self.Wd_s_0) + self.bd_s_0)

        # ^ iff x_{t - 1} = </s> (m_t = 0) then x_{t-1} = 0
        # and hd_{t - 1} = 0 else hd_{t - 1} is left unchanged (m_t = 1)

        if self.decoder_bias_type == 'first':
            # Do not bias all the decoder (force to store very useful information in the first state) 
            hd_t = self.sent_rec_activation( T.dot(xd_t, self.Wd_in) \
                                             + T.dot(hd_tm1, self.Wd_hh) \
                                             + self.bd_hh )
            output = (hd_t,)
        elif self.decoder_bias_type == 'all':
            hd_t = self.sent_rec_activation( T.dot(xd_t, self.Wd_in) \
                                             + T.dot(hd_tm1, self.Wd_hh) \
                                             + T.dot(decoder_inp_t, self.Wd_s_q) \
                                             + self.bd_hh )
            output = (hd_t,)
        elif self.decoder_bias_type == 'selective':
            rd_sel_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_sel_e) + T.dot(hd_tm1, self.Wd_sel_h) + T.dot(decoder_inp_t, self.Wd_sel_s) + self.bd_sel)
            decoder_inpr_t = rd_sel_t * decoder_inp_t
             
            hd_tilde = self.sent_rec_activation( T.dot(xd_t, self.Wd_in) \
                                        + T.dot(hd_tm1, self.Wd_hh) \
                                        + T.dot(decoder_inpr_t, self.Wd_s_q) \
                                        + self.bd_hh )
            output = (hd_t, decoder_inpr_t, rd_sel_t)

        return output

class DialogEncoderDecoder(Model):
    def indices_to_words(self, seq, exclude_end_sym=True):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        def convert():
            for word_index in seq:
                if word_index > len(self.idx_to_str):
                    raise ValueError('Word index is too large for the model vocabulary!')
                if not exclude_end_sym or (word_index != self.eos_sym):
                    yield self.idx_to_str[word_index]
        return list(convert())

    def words_to_indices(self, seq):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        return [self.str_to_idx.get(word, self.unk_sym) for word in seq]

    def compute_updates(self, training_cost, params):
        updates = []
         
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))

        # Clip stuff
        c = numpy.float32(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))
        
        grads = OrderedDict(clip_grads)

        if self.initialize_from_pretrained_word_embeddings and self.fix_pretrained_word_embeddings:
            # Keep pretrained word embeddings fixed
            logger.debug("Will use mask to fix pretrained word embeddings")
            grads[self.W_emb] = grads[self.W_emb] * self.W_emb_pretrained_mask
        else:
            logger.debug("Will train all word embeddings")

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)  
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads)
        else:
            raise Exception("Updater not understood!") 

        return updates
  
    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function")
                
            self.train_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                         self.x_max_length, self.x_cost_mask, 
                                                         self.x_semantic_targets, self.x_reset_mask],
                                            outputs=self.training_cost,
                                            updates=self.updates + self.state_updates, 
                                            on_unused_input='warn', 
                                            name="train_fn")

        return self.train_fn
    
    def build_nce_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building NCE train function")

            self.nce_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                  self.y_neg, self.x_max_length, 
                                                  self.x_cost_mask, self.x_semantic_targets,
                                                  self.x_reset_mask],
                                            outputs=self.training_cost,
                                            updates=self.updates + self.state_updates, 
                                            on_unused_input='warn', 
                                            name="train_fn") 

        return self.nce_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")
            self.eval_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_cost_mask, self.x_semantic_targets, self.x_reset_mask], 
                                            outputs=[self.softmax_cost_acc, self.softmax_cost], 
                                            updates=self.state_updates,
                                            on_unused_input='warn', name="eval_fn")
        return self.eval_fn

    def build_get_states_function(self):
        if not hasattr(self, 'get_states_fn'):
            # Compile functions
            logger.debug("Building selective function")
            
            outputs = [self.h, self.hs, self.hd] + [x for x in self.utterance_decoder_states]
            self.get_states_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_semantic_targets, self.x_reset_mask],
                                            outputs=outputs, updates=self.state_updates, on_unused_input='warn',
                                            name="get_states_fn")
        return self.get_states_fn

    # Currently does not supported truncated computations...
    def build_next_probs_function(self):
        if not hasattr(self, 'next_probs_fn'):
            outputs, hd = self.utterance_decoder.build_next_probs_predictor(self.beam_hs, self.beam_source, prev_state=self.beam_hd)
            self.next_probs_fn = theano.function(inputs=[self.beam_hs, self.beam_source, self.beam_hd],
                outputs=[outputs, hd],
                name="next_probs_fn")
        return self.next_probs_fn

    # Currently does not supported truncated computations...
    def build_encoder_function(self):
        if not hasattr(self, 'encoder_fn'):

            if self.bidirectional_utterance_encoder:
                res_forward = self.utterance_encoder_forward.build_encoder(self.x_data)
                res_backward = self.utterance_encoder_backward.build_encoder(self.x_data_reversed)

                # Each encoder gives a single output vector
                h = T.concatenate([res_forward, res_backward], axis=2)

                hs = self.dialog_encoder.build_encoder(h, self.x_data)

                if self.direct_connection_between_encoders_and_decoder:
                    hs_dummy = self.dialog_dummy_encoder.build_encoder(h, self.x_data)
                    hs_and_hs_dummy = T.concatenate([hs, h], axis=2)

                    self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_semantic_targets],
                        outputs=[h, hs_and_hs_dummy], on_unused_input='warn', name="encoder_fn")
                else:
                    self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_semantic_targets],
                        outputs=[h, hs], on_unused_input='warn', name="encoder_fn")
            else:
                h = self.utterance_encoder.build_encoder(self.x_data)

                hs = self.dialog_encoder.build_encoder(h, self.x_data)

                if self.direct_connection_between_encoders_and_decoder:
                    hs_dummy = self.dialog_dummy_encoder.build_encoder(h, self.x_data)
                    hs_and_hs_dummy = T.concatenate([hs, hs_dummy], axis=2)

                    self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_semantic_targets],
                        outputs=[h, hs_and_hs_dummy], on_unused_input='warn', name="encoder_fn")
                else:
                    # Define Theano function to compute encodings
                    self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_semantic_targets],
                        outputs=[h, hs], on_unused_input='warn', name="encoder_fn")

        return self.encoder_fn

    def __init__(self, state):
        Model.__init__(self)    

        # Compatibility towards older models
        if 'bootstrap_from_semantic_information' in state:
            assert state['bootstrap_from_semantic_information'] == False # We don't support semantic info right now...
              

        if not 'bidirectional_utterance_encoder' in state:
            state['bidirectional_utterance_encoder'] = False

        if 'encode_with_l2_pooling' in state:
            assert state['encode_with_l2_pooling'] == False # We don't support L2 pooling right now...

        if not 'direct_connection_between_encoders_and_decoder' in state:
            state['direct_connection_between_encoders_and_decoder'] = False

        if not 'deep_direct_connection' in state:
            state['deep_direct_connection'] = False

        if state['direct_connection_between_encoders_and_decoder']:
            assert(state['deep_direct_connection'] == False)

        if not 'collaps_to_standard_rnn' in state:
            state['collaps_to_standard_rnn'] = False

        if not 'never_reset_decoder' in state:
            state['never_reset_decoder'] = False

        if not 'deep_dialogue_input' in state:
            state['deep_dialogue_input'] = True

        self.state = state
        self.global_params = []

        self.__dict__.update(state)
        self.rng = numpy.random.RandomState(state['seed']) 

        # Load dictionary
        raw_dict = cPickle.load(open(self.dictionary, 'r'))
        # Probabilities for each term in the corpus
        self.noise_probs = [x[2] for x in sorted(raw_dict, key=operator.itemgetter(1))]
        self.noise_probs = numpy.array(self.noise_probs, dtype='float64')
        self.noise_probs /= numpy.sum(self.noise_probs)
        self.noise_probs = self.noise_probs ** 0.75
        self.noise_probs /= numpy.sum(self.noise_probs)
        
        self.t_noise_probs = theano.shared(self.noise_probs.astype('float32'), 't_noise_probs')
        # Dictionaries to convert str to idx and vice-versa
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])

        # Extract document (dialogue) frequency for each word
        self.word_freq = dict([(tok_id, freq) for _, tok_id, freq, _ in raw_dict])
        self.document_freq = dict([(tok_id, df) for _, tok_id, _, df in raw_dict])

        #if '</s>' not in self.str_to_idx \
        #   or '</d>' not in self.str_to_idx:
        #   raise Exception("Error, malformed dictionary!")

        if '</s>' not in self.str_to_idx:
           raise Exception("Error, malformed dictionary!")
         
        # Number of words in the dictionary 
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim
        logger.debug("idim: " + str(self.idim))

        logger.debug("Initializing Theano variables")
        self.y_neg = T.itensor3('y_neg')
        self.x_data = T.imatrix('x_data')
        self.x_data_reversed = T.imatrix('x_data_reversed')
        self.x_cost_mask = T.matrix('cost_mask')
        self.x_reset_mask = T.vector('reset_mask')
        self.x_max_length = T.iscalar('x_max_length')

        self.x_semantic_targets = T.imatrix('x_semantic')
        
        
        # The training data is defined as all symbols except the last, and
        # the target data is defined as all symbols except the first.
        training_x = self.x_data[:(self.x_max_length-1)]
        training_x_reversed = self.x_data_reversed[:(self.x_max_length-1)]
        training_y = self.x_data[1:self.x_max_length]

        # Here we find the end-of-sentence tokens in the minibatch.
        training_hs_mask = T.neq(training_x, self.eos_sym)
        training_x_cost_mask = self.x_cost_mask[1:self.x_max_length].flatten()
        
        # Backward compatibility
        if 'decoder_bias_type' in self.state:
            logger.debug("Decoder bias type {}".format(self.decoder_bias_type))


        # Build word embeddings, which are shared throughout the model
        if self.initialize_from_pretrained_word_embeddings == True:
            # Load pretrained word embeddings from pickled file
            logger.debug("Loading pretrained word embeddings")
            pretrained_embeddings = cPickle.load(open(self.pretrained_word_embeddings_file, 'r'))

            # Check all dimensions match from the pretrained embeddings
            assert(self.idim == pretrained_embeddings[0].shape[0])
            assert(self.rankdim == pretrained_embeddings[0].shape[1])
            assert(self.idim == pretrained_embeddings[1].shape[0])
            assert(self.rankdim == pretrained_embeddings[1].shape[1])

            self.W_emb_pretrained_mask = theano.shared(pretrained_embeddings[1].astype(numpy.float32), name='W_emb_mask')
            self.W_emb = add_to_params(self.global_params, theano.shared(value=pretrained_embeddings[0].astype(numpy.float32), name='W_emb'))
        else:
            # Initialize word embeddings randomly
            self.W_emb = add_to_params(self.global_params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))

        # Variables to store encoder and decoder states
        if self.bidirectional_utterance_encoder:
            # Previous states variables
            self.ph_fwd = theano.shared(value=numpy.zeros((self.bs, self.qdim), dtype='float32'), name='ph_fwd')
            self.ph_bck = theano.shared(value=numpy.zeros((self.bs, self.qdim), dtype='float32'), name='ph_bck')
            self.phs = theano.shared(value=numpy.zeros((self.bs, self.sdim), dtype='float32'), name='phs')
            self.phd = theano.shared(value=numpy.zeros((self.bs, self.qdim), dtype='float32'), name='phd')

            if self.direct_connection_between_encoders_and_decoder:
                self.phs_dummy = theano.shared(value=numpy.zeros((self.bs, self.qdim*2), dtype='float32'), name='phs_dummy')

        else:
            # Previous states variables
            self.ph = theano.shared(value=numpy.zeros((self.bs, self.qdim), dtype='float32'), name='ph')
            self.phs = theano.shared(value=numpy.zeros((self.bs, self.sdim), dtype='float32'), name='phs')
            self.phd = theano.shared(value=numpy.zeros((self.bs, self.qdim), dtype='float32'), name='phd')

            if self.direct_connection_between_encoders_and_decoder:
                self.phs_dummy = theano.shared(value=numpy.zeros((self.bs, self.qdim), dtype='float32'), name='phs_dummy')

        # Build utterance encoders
        if self.bidirectional_utterance_encoder:
            logger.debug("Initializing forward utterance encoder")
            self.utterance_encoder_forward = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')
            logger.debug("Build forward utterance encoder")
            res_forward = self.utterance_encoder_forward.build_encoder(training_x, xmask=training_hs_mask, prev_state=self.ph_fwd)

            logger.debug("Initializing backward utterance encoder")
            self.utterance_encoder_backward = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'bck')
            logger.debug("Build backward utterance encoder")
            res_backward = self.utterance_encoder_backward.build_encoder(training_x_reversed, xmask=training_hs_mask, prev_state=self.ph_bck)

            # The encoder h embedding is a concatenation of final states of the forward and backward encoder RNNs
            self.h = T.concatenate([res_forward, res_backward], axis=2)
        else:
            logger.debug("Initializing utterance encoder")
            self.utterance_encoder = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')

            logger.debug("Build utterance encoder")
            # The encoder h embedding is the final hidden state of the forward encoder RNN
            self.h = self.utterance_encoder.build_encoder(training_x, xmask=training_hs_mask, prev_state=self.ph)

        logger.debug("Initializing dialog encoder")
        self.dialog_encoder = DialogEncoder(self.state, self.rng, self, '')

        logger.debug("Build dialog encoder")
        self.hs = self.dialog_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask, prev_state=self.phs)

        # We initialize the decoder, and fix its word embeddings to that of the encoder(s)
        logger.debug("Initializing decoder")
        self.utterance_decoder = UtteranceDecoder(self.state, self.rng, self, self.dialog_encoder, self.W_emb)

        if self.direct_connection_between_encoders_and_decoder:
            logger.debug("Initializing dialog dummy encoder")
            if self.bidirectional_utterance_encoder:
                self.dialog_dummy_encoder = DialogDummyEncoder(self.state, self.rng, self, self.qdim*2)
            else:
                self.dialog_dummy_encoder = DialogDummyEncoder(self.state, self.rng, self, self.qdim)

            logger.debug("Build dialog dummy encoder")
            self.hs_dummy = self.dialog_dummy_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask, prev_state=self.phs_dummy)

            logger.debug("Build decoder (NCE) with direct connection from encoder(s)")
            self.hs_and_hs_dummy = T.concatenate([self.hs, self.hs_dummy], axis=2)

            contrastive_cost, self.hd_nce = self.utterance_decoder.build_decoder(self.hs_and_hs_dummy, training_x, y_neg=self.y_neg, y=training_y, xmask=training_hs_mask, mode=UtteranceDecoder.NCE, prev_state=self.phd)

            logger.debug("Build decoder (EVAL) with direct connection from encoder(s)")
            target_probs, self.hd, self.utterance_decoder_states, target_probs_full_matrix = self.utterance_decoder.build_decoder(self.hs_and_hs_dummy, training_x, xmask=training_hs_mask, y=training_y, mode=UtteranceDecoder.EVALUATION, prev_state=self.phd)

        else:
            logger.debug("Build decoder (NCE)")
            contrastive_cost, self.hd_nce = self.utterance_decoder.build_decoder(self.hs, training_x, y_neg=self.y_neg, y=training_y, xmask=training_hs_mask, mode=UtteranceDecoder.NCE, prev_state=self.phd)

            logger.debug("Build decoder (EVAL)")
            target_probs, self.hd, self.utterance_decoder_states, target_probs_full_matrix = self.utterance_decoder.build_decoder(self.hs, training_x, xmask=training_hs_mask, y=training_y, mode=UtteranceDecoder.EVALUATION, prev_state=self.phd)

        # Prediction cost and rank cost
        self.contrastive_cost = T.sum(contrastive_cost.flatten() * training_x_cost_mask)
        self.softmax_cost = -T.log(target_probs) * training_x_cost_mask
        self.softmax_cost_acc = T.sum(self.softmax_cost)

        # Prediction accuracy
        self.training_misclassification = T.neq(T.argmax(target_probs_full_matrix, axis=2), training_y).flatten() * training_x_cost_mask

        self.training_misclassification_acc = T.sum(self.training_misclassification)

        # Compute training cost, which equals standard cross-entropy error
        self.training_cost = self.softmax_cost_acc
        if self.use_nce:
            self.training_cost = self.contrastive_cost

        # Init params
        if self.collaps_to_standard_rnn:
                self.params = self.global_params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_decoder.params))
        else:
            if self.bidirectional_utterance_encoder:
                self.params = self.global_params + self.utterance_encoder_forward.params + self.utterance_encoder_backward.params + self.dialog_encoder.params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_encoder_forward.params) + len(self.utterance_encoder_backward.params) + len(self.dialog_encoder.params) + len(self.utterance_decoder.params))
            else:
                self.params = self.global_params + self.utterance_encoder.params + self.dialog_encoder.params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_encoder.params) + len(self.dialog_encoder.params) + len(self.utterance_decoder.params))

        self.updates = self.compute_updates(self.training_cost / training_x.shape[1], self.params)

        # Truncate gradients properly by brining forward previous states
        self.state_updates = []
        if self.bidirectional_utterance_encoder:
            x_reset = self.x_reset_mask.dimshuffle(0, 'x')
            self.state_updates.append((self.ph_fwd, x_reset * res_forward[-1]))
            self.state_updates.append((self.ph_bck, x_reset * res_backward[-1]))
            self.state_updates.append((self.phs, x_reset * self.hs[-1]))
            self.state_updates.append((self.phd, x_reset * self.hd[-1]))
        else:
            x_reset = self.x_reset_mask.dimshuffle(0, 'x')
            self.state_updates.append((self.ph, x_reset * self.h[-1]))
            self.state_updates.append((self.phs, x_reset * self.hs[-1]))
            self.state_updates.append((self.phd, x_reset * self.hd[-1]))

        if self.direct_connection_between_encoders_and_decoder:
            self.state_updates.append((self.phs_dummy, x_reset * self.hs_dummy[-1]))


        # Beam-search variables
        self.beam_source = T.lvector("beam_source")
        self.beam_hs = T.matrix("beam_hs")
        self.beam_step_num = T.lscalar("beam_step_num")
        self.beam_hd = T.matrix("beam_hd")

        # DEBUG CODE USED TO COMPARE TO NO-TRUNCATED VERSION
        #self.get_hs_func = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_cost_mask, self.x_semantic_targets, self.x_reset_mask], outputs=self.hd, updates=self.state_updates, on_unused_input='warn', name="get_hs_fn")




