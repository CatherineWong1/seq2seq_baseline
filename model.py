# -*- coding:utf-8 -*-
import time

import numpy as np
import tensorflow as tf

import config

class ChatBotModel:
    def __init__(self, forward_only, batch_size):
        """
        :param forward_only: 如果设置了，将不构造模型中的反向传递
        :param batch_size:
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size

    def _create_placeholders(self):
        # fees for inputs
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]
        # weights
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)]

        # targets are decoder inputs shifted by one(ignore <GO> model)
        self.targets = self.decoder_inputs[1:]

    def _inference(self):
        print('Create inference')
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            '''
            sampled_softmax_loss：计算并返回采样数据的softmax loss
            通常会低估了完全的softmax loss
            '''
            # tf.transpose：对于二维向量是转置，对于其他维，则是交换张量维度即[2,1,0]变为[0,1,2]
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w),
                                               biases=b,
                                               inputs=logits,
                                               labels=labels,
                                               num_sampled=config.NUM_SAMPLES,
                                               num_classes=config.DEC_VOCAB)
        self.softmax_loss_function = sampled_loss

        single_cell = tf.nn.rnn_cell.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell for _ in range(config.NUM_LAYERS)])

    def _create_loss(self):
        print('Creating loss... \nIt might take a couple of minutes depending on how many buckets you have.')
        start = time.time()

        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            setattr(tf.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda  self, _:self)
            '''
            embedding_attention_seq2seq：
            1. 将encoder_inputs变成shape:num_encoder_symbols*input_size的embedding
            2. 用RNN将encoder_inputs编码为state向量
            3. 保持每一个step的outputs，方便后续使用attention
            4. 将decoder_inputs变成:num_decoder_symbols*input_size的embedding
            5. 运行attention decoder，初始化最后一个encoder的状态以及decoder_inputs
            返回一个元组(outpus,state)
            '''
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.cell,
                num_encoder_symbols=config.ENC_VOCAB,
                num_decoder_symbols=config.DEC_VOCAB,
                embedding_size=config.HIDDEN_SIZE,
                output_projection=self.output_projection,
                feed_previous=do_decode
            )

        if self.fw_only:
            '''
            model_with_buckets：创建支持bucketing的seq2seq
            bucketing的目的：和padding一样，均是为了处理不同长度句子的情况。
            经常采用的是PAD符合来填充长度，但是对于较短的句子，添加较多PAD没有意义，且十分低效
            因此设置一定数量的buckets
            BUCKETS = [(19, 19), (28, 29), (33, 33), (40, 43), (50, 53), (60, 63)]
            如果input和output均小于19，则使用(19,19)进行填充，如果input的长度为10，output的长度为21，则使用(28,29)进行填充
            :return (outputs,losses),outputs是each bucket的output
            '''
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, self.targets,
                self.decoder_masks, config.BUCKETS, lambda x, y: _seq2seq_f(x, y, True),
                softmax_loss_function=self.softmax_loss_function)

            if self.output_projection:
                for bucket in range(len(config.BUCKETS)):
                    # wx+b
                    self.outputs[bucket] = [tf.matmul(output,
                                                      self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,self.decoder_inputs, self.targets,
                self.decoder_masks,config.BUCKETS,lambda  x,y: _seq2seq_f(x,y, False),
                softmax_loss_function=self.softmax_loss_function)

        print('Time:', time.time()-start)

    def _create_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):
                    # gradients:构造loss在trainables的导数,返回的是 sum（dy/dx）
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                              trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                         global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()

    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()





