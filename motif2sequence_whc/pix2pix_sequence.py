import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import collections
import random
import time
import math

BATCH_SIZE = 32
EPS = 1e-12
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


class DataLoader():
    def __init__(self):
        self.file_dir = './'
        self.file_name = 'part_dataset.csv'
        self.batch_size = BATCH_SIZE

    # 读取数据集，输出所需的模型输入序列及模型输出序列
    def csv_load(self, file_dir, file_name):
        path = os.path.join(file_dir, file_name)
        file = pd.read_table(path, sep=',')
        sequence = file['sequence']
        outline_sequence = file['sequence_outline']
        return sequence, outline_sequence

    def seq2onehot(self, sequence):
        # one hot Encoder definition
        onehot_encoder = OneHotEncoder(sparse=False)
        store_onehot = np.zeros((len(sequence), len(sequence[0]), 5))
        for index, item in sequence.items():
            # 添加一个Z，因为在原始序列中不含'Z'，无法产生5列的one-hot编码；
            seq_array = np.array(list(item + 'ZAGCT'))
            # integer encode the sequence
            label_encoder = LabelEncoder()
            integer_encoded_seq = label_encoder.fit_transform(seq_array)

            # reshape because that's what OneHotEncoder likes
            integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
            onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)[0:-5]
            store_onehot[index, :, :] = onehot_encoded_seq
        return store_onehot, onehot_encoder, label_encoder

    def make_generator(self, data_matrix, batch_size, random_shuffle=True):
        while True:
            if random_shuffle:
                np.random.shuffle(data_matrix)
            for i in range(0, len(data_matrix) - batch_size + 1, batch_size):
                yield data_matrix[i:i + batch_size]


class GANModel:
    def __init__(self):
        self.kernel_size = 4
        self.strides = 2
        self.ngf = 64 # number of generator filters
        self.ndf = 64 # number of discriminator filters
        self.gan_weight = 1.0
        self.l1_weight = 100.0
        self.lr = 0.0002
        self.beta1 = 0.5
        self.batch_size = BATCH_SIZE

    def gen_conv(self, inputs, channels, kernel_size, padding_method):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv1d(inputs, channels, kernel_size=kernel_size, strides=self.strides,
                                padding=padding_method,
                                kernel_initializer=initializer)

    def gen_deconv(self, inputs, channels, kernel_size, padding_method = 'SAME'):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        filter = tf.Variable(tf.random_normal([kernel_size, channels, int(inputs.shape[2])]))  #[kernel_size, output_channels, input_channels]

        # output_shape = [batch_size, output_shape(width), output_channels]
        if padding_method == 'VALID':
            return tf.contrib.nn.conv1d_transpose(inputs, filter=filter,
                                                  output_shape=[self.batch_size, int(inputs.shape[1]) * 2 + kernel_size - 1, channels],
                                                  stride=self.strides, padding=padding_method)
        else:
            return tf.contrib.nn.conv1d_transpose(inputs, filter=filter,
                                                  output_shape=[self.batch_size, int(inputs.shape[1]) * 2, channels],
                                                  stride=self.strides, padding=padding_method)

    def discrim_conv(self, batch_input, out_channels, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv1d(padded_input, out_channels, kernel_size = self.kernel_size, strides=self.strides,
                                padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def lrelu(self, x, a):
        with tf.name_scope("lrelu"):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: a*x/2 - a*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(self, inputs):
        return tf.layers.batch_normalization(inputs, axis=2, epsilon=1e-5, momentum=0.1, training=True,
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

    def create_generator(self, generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 118, in_channels] => [batch, 59, ngf]
        with tf.variable_scope("encoder_1"):
            output = self.gen_conv(generator_inputs, self.ngf, kernel_size=4, padding_method='same')
            layers.append(output)

        layer_specs = [
            self.ngf * 2,  # encoder_2: [batch, 59, ngf * 2] => [batch, 30, ngf * 4]
            self.ngf * 4,   # encoder_3: [batch, 30, ngf * 4] => [batch, 15, ngf * 8]
            self.ngf * 8,  # encoder_4: [batch, 15, ngf * 8] => [batch, 8, ngf * 8]
            self.ngf * 8,  # encoder_5: [batch, 8, ngf * 8] => [batch, 4, ngf * 8]
            self.ngf * 8,  # encoder_6: [batch, 4, ngf * 8] => [batch, 2, ngf * 8]
        ]

        for decoder_layer, out_channels in enumerate(layer_specs):
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = self.lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                if decoder_layer == 0 or decoder_layer == 3:
                    convolved = self.gen_conv(rectified, out_channels, kernel_size=4, padding_method='valid')
                else:
                    convolved = self.gen_conv(rectified, out_channels, kernel_size=4, padding_method='same')
                output = self.batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (self.ngf * 8, 0.5),  # decoder_6: [batch, 1,  ngf * 8] => [batch, 2,  ngf * 8 * 2]   0
            (self.ngf * 8, 0.5),  # decoder_5: [batch, 2,  ngf * 8 * 2] => [batch, 7, ngf * 8 * 2]  1*
            (self.ngf * 8, 0.5),  # decoder_4: [batch, 7,  ngf * 8 * 2] => [batch, 14, ngf * 8 * 2]  2
            (self.ngf * 8, 0.0),  # decoder_3: [batch, 14,  ngf * 8 * 2] => [batch, 28, ngf * 8 * 2]  3
            (self.ngf * 4, 0.0),  # decoder_2: [batch, 28, ngf * 8 * 2] => [batch, 59, ngf 0* 4 * 2]   4*
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=2)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]

                if decoder_layer == 1 or decoder_layer == 4:
                    output = self.gen_deconv(rectified, out_channels, kernel_size = 4, padding_method = 'VALID')
                else:
                    output = self.gen_deconv(rectified, out_channels, kernel_size = 4)

                output = self.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 58, ngf * 4 * 2] => [batch, 118, ngf * 2 * 2]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=2)
            rectified = tf.nn.relu(input)
            output = self.gen_deconv(rectified, generator_outputs_channels, kernel_size = 4, padding_method = 'SAME')
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]

    def create_model(self, inputs, targets):
        def create_discriminator(discrim_inputs, discrim_targets):
            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=2)

            # layer_1: [batch, 118, in_channels * 2] => [batch, 59, ndf]
            with tf.variable_scope("layer_1"):
                convolved = self.discrim_conv(input, self.ndf, stride=2)
                rectified = self.lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 59, ndf] => [batch, 29, ndf * 2]
            # layer_3: [batch, 29, ndf * 2] => [batch, 14, ndf * 4]
            # layer_4: [batch, 14, ndf * 4] => [batch, 7, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = self.ndf * min(2 ** (i + 1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = self.discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = self.batchnorm(convolved)
                    rectified = self.lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = self.discrim_conv(rectified, out_channels=1, stride=1)

                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = self.create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(inputs, outputs)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * self.gan_weight + gen_loss_L1 * self.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )


if __name__ == '__main__':
    file_dir = './'
    file_name = 'part_dataset.csv'
    batch_size = BATCH_SIZE
    ran  = True # Random or not
    output_dir = './output'
    seq_len = 118
    trace_freq = 0
    summary_freq = 100
    max_epochs = 100
    progress_freq = 50
    generate_samples = 500
    save_freq = 5000

    if ran == True:
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = 0

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the data
    Loader = DataLoader()
    sequence, outline_sequence = Loader.csv_load(file_dir, file_name)

    seq_onehot, onehot_encoder, label_encoder = Loader.seq2onehot(sequence)
    outline_onehot, nouse, nouse2 = Loader.seq2onehot(outline_sequence)
    gen_seq = Loader.make_generator(seq_onehot, batch_size)
    gen_outline = Loader.make_generator(outline_onehot, batch_size)
    steps_per_epoch = int(len(seq_onehot) / batch_size)


    GAN = GANModel()
    # inputs and targets are [batch_size, height, width, channels]
    inputs = tf.placeholder(tf.float32, shape=[batch_size, seq_len, 5])
    targets = tf.placeholder(tf.float32, shape=[batch_size, seq_len, 5])
    model = GAN.create_model(inputs, targets)

    # summaries

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = output_dir if (trace_freq > 0 or summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        max_steps = 2**32
        if max_epochs is not None:
            max_steps = steps_per_epoch * max_epochs

        if True:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(summary_freq):
                    fetches["summary"] = sv.summary_op

                seq = next(gen_seq)
                outline = next(gen_outline)

                results = sess.run(fetches, options=options, run_metadata=run_metadata, feed_dict={inputs:seq, targets:outline})


                if should(summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
                    train_step = (results["global_step"] - 1) % steps_per_epoch + 1
                    rate = (step + 1) * batch_size / (time.time() - start)
                    remaining = (max_steps - step) * batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)

                if should(generate_samples):
                    print("Generate samples")
                    samples = []
                    for i in range(steps_per_epoch):
                        tmp_inputs = outline_onehot[i*batch_size: (i+1)*batch_size]
                        result = sess.run(model.outputs, feed_dict={inputs: tmp_inputs})
                        for item in result:
                            label = onehot_encoder.inverse_transform(item)
                            seq = label_encoder.inverse_transform(label.ravel())
                            seq = list(seq)
                            str = ''
                            for sub_item in seq:
                                str = str + sub_item
                            samples.append(str)
                    with open('samples_%d_%d.txt' % (train_epoch, train_step), 'w') as f:
                        for item in samples:
                            f.write(item + '\n')



                if sv.should_stop():
                    break

