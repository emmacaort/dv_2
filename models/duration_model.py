import tensorflow as tf


class DurationModel(object):

    def __init__(self,
                 input_vocab_size,
                 num_speakers,
                 num_buckets,
                 model_parameters
                 ):
        self.input_vocab_size = input_vocab_size
        self.num_speakers = num_speakers
        self.num_buckets = num_buckets
        self.model_parameters = model_parameters

    def __buil_logits(self,
                      phonemes, phonemes_seq_len,
                      speaker_ids, dropout_prob, reuse
                      ):
        with tf.variable_scope("speaker_embedding", reuse=reuse):
            speaker_embedding = tf.get_variable(
                'speaker_embedding',
                shape=(self.num_speakers, self.model_parameters["speaker_embedding_size"]),
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.05)
            ) # shape=(n_speaker,speaker_embed_size) e.g: (10,16) means 10 speakers and 16 embeddings
            
            speaker_embedding_output = tf.nn.embedding_lookup(speaker_embedding, speaker_ids)
            # shape = (batch_size,speaker_embed_size) e.g: (2,16) means 2 phonemes and 16 embeddings

        with tf.variable_scope("dense_layers", reuse=reuse):
            phonemes_embedding = tf.get_variable(
                'phonemes_embedding',
                shape=(self.input_vocab_size, self.model_parameters["phonemes_embedding_size"]),
                dtype=tf.float32
            ) # shape = (vocab_size, phoneme_embed_size) e.g: (50,56) means 50 kinds of phonemes and 56 embeddings
            
            phonemes_output = tf.nn.embedding_lookup(phonemes_embedding, phonemes)
            # shape = (sentences, phonemes,  phoneme_embed_size) e.g: (2,200,56) means 2 exampels, 
            # 200 phonemes and 56 embeddings (maybe there is an embedding for sil)
            
            speaker_embedding_projection = tf.tile(tf.expand_dims(tf.layers.dense(
                speaker_embedding_output, self.model_parameters["phonemes_embedding_size"],
                tf.nn.softsign
            ), 1), [1, tf.shape(phonemes)[1], 1])  
            # from (sentence,speaker_embed_size) to (sentence,phonemes_n,phoneme_embed_size)
            # from (2,16) to (2,56) to (2,200,56)

            output = tf.concat([speaker_embedding_projection, phonemes_output], 2)
            # concatenate at the unit-size axis, (2,200,56) to (2,200,112)

            for i in xrange(self.model_parameters["num_dense_layers"]):
                with tf.variable_scope("dense_layer_" + str(i)):
                    output = tf.layers.dense(
                        output, self.model_parameters["dense_layers_units"]
                    ) # (2,200,112) to (2,200,16)

        with tf.variable_scope("bidirectional_layers", reuse=reuse):
            cells_fw = [
                tf.nn.rnn_cell.GRUCell(
                    self.model_parameters["num_bidirectional_units"]
                ) for i in xrange(self.model_parameters["num_bidirectional_layers"])
            ]
            cells_bw = [
                tf.nn.rnn_cell.GRUCell(
                    self.model_parameters["num_bidirectional_units"]
                ) for i in xrange(self.model_parameters["num_bidirectional_layers"])
            ]

            speaker_embedding_projection = tf.layers.dense(
                speaker_embedding_output, self.model_parameters["num_bidirectional_units"]
            )

            bidirectional, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw, cells_bw, output,
                sequence_length=phonemes_seq_len, dtype=tf.float32,
                initial_states_fw=[speaker_embedding_projection] * self.model_parameters["num_bidirectional_layers"],
                initial_states_bw=[speaker_embedding_projection] * self.model_parameters["num_bidirectional_layers"]
            )
            # (2,200,16) to (2,200,32). 32=16*2 where 16 is the fw/bw unit size
            dropout = tf.layers.dropout(bidirectional, dropout_prob)
            logits = tf.layers.dense(dropout, self.num_buckets)            

        return logits

    def build_train_operations(self,
                 phonemes, phonemes_seq_len,
                 speaker_ids,
                 durations,
                 train_parameters, reuse=None
                 ):
        logits = self.__buil_logits(phonemes, phonemes_seq_len,
                                    speaker_ids, train_parameters["dropout_prob"],
                                    reuse)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, durations, phonemes_seq_len
        )
        loss = tf.reduce_mean(-log_likelihood)
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(
            0, name="global_step", trainable=False
        )
        learning_rate = tf.train.exponential_decay(
            train_parameters["lr"], global_step,
            train_parameters["decay_steps"],
            train_parameters["decay_rate"]
        )
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients, variables = zip(*opt.compute_gradients(loss))
        train_op = opt.apply_gradients(zip(gradients, variables), global_step=global_step)

        summary = tf.summary.merge_all()

        return train_op, loss, global_step, summary, logits, transition_params
    """
    def val_operations(self,phonemes, phonemes_seq_len,
                 speaker_ids,
                 durations,
                 train_parameters, reuse=None):
        
        logits = self.__buil_logits(phonemes, phonemes_seq_len,
                                    speaker_ids, 0.0,
                                    reuse)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, durations, phonemes_seq_len)
        loss = tf.reduce_mean(-log_likelihood)
        return loss
     """
    def viterbi_predict(self,
                 phonemes, phonemes_seq_len,
                 speaker_ids, trans_params, reuse=None
                 ):
        logits = self.__buil_logits(phonemes, phonemes_seq_len,
                                    speaker_ids, 0.0,
                                    reuse)
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
        logits, trans_params, phonemes_seq_len)
        return viterbi_sequence
