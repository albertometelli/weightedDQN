import numpy as np
import tensorflow as tf

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
)

class SimpleNet:
    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self._session = tf.Session(config=config)

        if load_path is not None:
            self._load(load_path)
        else:
            self._build(convnet_pars)

        if self._name == 'train':
            self._train_saver = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name))
        elif self._name == 'target':
            w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name)

            with tf.variable_scope(self._scope_name):
                self._target_w = list()
                self._w = list()
                with tf.variable_scope('weights_placeholder'):
                    for i in range(len(w)):
                        self._target_w.append(tf.placeholder(w[i].dtype,
                                                             shape=w[i].shape))
                        self._w.append(w[i].assign(self._target_w[i]))

    def predict(self, s):
        out = np.array([self._session.run([self._q, self._sigma], feed_dict={self._x: s})])

        return out

    def fit(self, s, a, q_and_sigma, prob_exploration):
        summaries, _, loss = self._session.run(
            [self._merged, self._train_step, self.loss],
            feed_dict={self._x: s,
                       self._action: a.ravel().astype(np.uint8),
                       self._target_q: q_and_sigma[0, :],
                       self._target_sigma: q_and_sigma[1, :],
                       self._prob_exploration: prob_exploration}
        )

        if hasattr(self, '_train_writer'):
            self._train_writer.add_summary(summaries, self._train_count)

        self._train_count += 1

    def set_weights(self, weights):
        w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=self._scope_name)
        assert len(w) == len(weights)

        for i in range(len(w)):
            self._session.run(self._w[i],
                              feed_dict={self._target_w[i]: weights[i]})

    def get_weights(self, only_trainable=False):
        if not only_trainable:
            w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name)
        else:
            w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self._scope_name)

        return self._session.run(w)

    def save(self):
        self._train_saver.save(
            self._session,
            self._folder_name + '/' + self._scope_name[:-1] + '/' +
            self._scope_name[:-1]
        )

    def _load(self, path):
        self._scope_name = 'train/'
        restorer = tf.train.import_meta_graph(
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1] +
            '.meta')
        restorer.restore(
            self._session,
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1]
        )
        self._restore_collection()

    def _build(self, convnet_pars):

        with tf.variable_scope(None, default_name=self._name):
            self._scope_name = tf.get_default_graph().get_name_scope() + '/'

            with tf.variable_scope('State'):
                self._x = tf.placeholder(tf.float32,
                                         shape=[None] + list(
                                             convnet_pars['input_shape']),
                                         name='input')

            with tf.variable_scope('Action'):
                self._action = tf.placeholder('uint8', [None], name='action')

                action_one_hot = tf.one_hot(self._action,
                                            convnet_pars['output_shape'][0],
                                            name='action_one_hot')

            if convnet_pars['n_states'] is not None:
                x = tf.one_hot(tf.cast(self._x[..., 0], tf.int32),
                               convnet_pars['n_states'])
            else:
                x = self._x[...]

            self.sigma_weight = convnet_pars['sigma_weight']
            self.q_min = convnet_pars['q_min']
            self.q_max = convnet_pars['q_max']
            mean = (self.q_min + self.q_max) / 2.
            logsigma = np.log((self.q_max - self.q_min) / np.sqrt(12))

            with tf.variable_scope('Q_Net'):
                self._features_q_1 = tf.layers.dense(
                    x, convnet_pars['n_features'],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='features_q_1'
                )
                self._features_q_2 = tf.layers.dense(
                    self._features_q_1, convnet_pars['n_features'],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='features_q_2'
                )
                self._q = tf.layers.dense(
                    self._features_q_2,
                    convnet_pars['output_shape'][0],
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.constant_initializer(mean),
                    name='q'
                )
                self._q_acted = tf.reduce_sum(self._q * action_one_hot,
                                  axis=1,
                                  name='q_acted')

            with tf.variable_scope('Sigma_Net'):
                self._features_sigma__1 = tf.layers.dense(
                    x, convnet_pars['n_features'],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='features_sigma_1'
                )
                self._features_sigma_2 = tf.layers.dense(
                    self._features_sigma_1, convnet_pars['n_features'],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='features_sigma_2'
                )
                self._log_sigma = tf.layers.dense(
                    self._features_sigma_2,
                    convnet_pars['output_shape'][0],
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.constant_initializer(logsigma),
                    name='log_sigma'
                )

                self._sigma = tf.exp(self._log_sigma, name='sigma')
                self._sigma_acted = tf.reduce_sum(self._sigma * action_one_hot,
                                                  axis=1,
                                                  name='sigma_acted')


            self._target_q = tf.placeholder(
                'float32',
                [None],
                name='target_q'
            )
            self._target_sigma = tf.placeholder(
                'float32',
                [None],
                name='target_sigma'
            )
            self.out = [self._q, self._sigma]
            loss = 0.
            if convnet_pars["loss"] == "huber_loss":
                self.loss_fuction = tf.losses.huber_loss
            else:
                self.loss_fuction = tf.losses.mean_squared_error


            loss = huber_loss((self._q_acted - self._target_q) ** 2 + \
                                     tf.scalar_mul(
                                         self.sigma_weight,
                                         (self._sigma_acted - self._target_sigma) ** 2))
            self._prob_exploration = tf.placeholder('float32', (),
                                                    name='prob_exploration')

            tf.summary.scalar(convnet_pars["loss"], tf.reduce_mean(loss))
            tf.summary.scalar('average_q', tf.reduce_mean(self._q))
            tf.summary.scalar('average_sigma', tf.reduce_mean(self._sigma))
            #tf.summary.scalar('prob_exploration', self._prob_exploration)
            #tf.summary.histogram('qs', self._q)
            #tf.summary.histogram('qs', self._sigma)
            self._merged = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES,
                                  scope=self._scope_name)
            )

            optimizer = convnet_pars['optimizer']

            if optimizer['name'] == 'rmspropcentered':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'],
                                                centered=True)
                sigma_opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr_sigma'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'],
                                                centered=True)
            elif optimizer['name'] == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'])
                sigma_opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr_sigma'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'])
            elif optimizer['name'] == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=optimizer['lr'])
                sigma_opt = tf.train.AdamOptimizer(learning_rate=optimizer['lr_sigma'])
            elif optimizer['name'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate=optimizer['lr'])
                sigma_opt = tf.train.AdadeltaOptimizer(learning_rate=optimizer['lr_sigma'])
            else:
                raise ValueError('Unavailable optimizer selected.')

            self._train_step = opt.minimize(loss=loss)
            self.loss =loss
            initializer = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self._scope_name))

        self._session.run(initializer)

        if self._folder_name is not None:
            self._train_writer = tf.summary.FileWriter(
                self._folder_name + '/' + self._scope_name[:-1],
                graph=tf.get_default_graph()
            )

        self._train_count = 0

        self._add_collection()

    @property
    def n_features(self):
        return self._features.shape[-1]

    def _add_collection(self):
        tf.add_to_collection(self._scope_name + '_x', self._x)
        tf.add_to_collection(self._scope_name + '_action', self._action)
        tf.add_to_collection(self._scope_name + '_features_q_1', self._features_q_1)
        tf.add_to_collection(self._scope_name + '_features_q_2', self._features_q_2)
        tf.add_to_collection(self._scope_name + '_features_sigma_1', self._features_sigma_1)
        tf.add_to_collection(self._scope_name + '_features_sigma_2', self._features_sigma_2)
        tf.add_to_collection(self._scope_name + '_q', self._q)
        tf.add_to_collection(self._scope_name + '_log_sigma', self._log_sigma)
        tf.add_to_collection(self._scope_name + '_sigma', self._sigma)
        tf.add_to_collection(self._scope_name + '_target_q', self._target_q)
        tf.add_to_collection(self._scope_name + '_target_sigma', self._target_sigma)
        tf.add_to_collection(self._scope_name + '_q_acted', self._q_acted)
        tf.add_to_collection(self._scope_name + '_sigma_acted', self._sigma_acted)
        tf.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.add_to_collection(self._scope_name + '_train_step', self._train_step)

    def _restore_collection(self):
        self._x = tf.get_collection(self._scope_name + '_x')[0]
        self._action = tf.get_collection(self._scope_name + '_action')[0]
        self._features_q_1 = tf.get_collection(self._scope_name + '_features_q_1')[0]
        self._features_q_2 = tf.get_collection(self._scope_name + '_features_q_2')[0]
        self._features_sigma_1 = tf.get_collection(self._scope_name + '_features_sigma_1')[0]
        self._features_sigma_2 = tf.get_collection(self._scope_name + '_features_sigma_2')[0]
        self._q = tf.get_collection(self._scope_name + '_q')[0]
        self._log_sigma = tf.get_collection(self._scope_name + '_log_sigma')[0]
        self._sigma = tf.get_collection(self._scope_name + '_sigma')[0]
        self._target_q = tf.get_collection(self._scope_name + '_target_q')[0]
        self._target_sigma = tf.get_collection(self._scope_name + '_target_sigma')[0]
        self._q_acted = tf.get_collection(self._scope_name + '_q_acted')[0]
        self._sigma_acted = tf.get_collection(self._scope_name + '_sigma_acted')[0]
        self._merged = tf.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.get_collection(self._scope_name + '_train_step')[0]