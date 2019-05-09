import numpy as np
import tensorflow as tf


class SimpleNet:
    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self._session = tf.Session(config=config)

        if load_path is not None:
            self._load(load_path, convnet_pars)
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

    def predict(self, s, idx=None):
        if idx is not None:
            return self._session.run(self._q[idx], feed_dict={self._x: s})
        else:
            #s = np.array([s])
            return np.array(
                [self._session.run(self._q, feed_dict={self._x: s})])

    def fit(self, s, a, q, mask):
        summaries, _ = self._session.run(
            [self._merged, self._train_step],
            feed_dict={self._x: s,
                       self._action: a.ravel().astype(np.uint8),
                       self._target_q: q,
                       self._mask: mask}
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

    def _load(self, path, convnet_pars):
        self._scope_name = 'train/'
        restorer = tf.train.import_meta_graph(
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1] +
            '.meta')
        restorer.restore(
            self._session,
            path + '/' + self._scope_name[:-1] + '/' + self._scope_name[:-1]
        )
        self._restore_collection(convnet_pars)

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

            with tf.variable_scope('Mask'):
                self._mask = tf.placeholder(
                    tf.float32, shape=[None, convnet_pars['n_approximators']])

            if convnet_pars['n_states'] is not None:
                x = tf.one_hot(tf.cast(self._x[..., 0], tf.int32),
                               convnet_pars['n_states'])
            else:
                x = self._x[...]

            self._features = list()
            self._features2 = list()
            self._q = list()
            self._q_acted = list()
            self.n_approximators = convnet_pars['n_approximators']


            for i in range(self.n_approximators):
                with tf.variable_scope('head_' + str(i)):
                    self._features.append(tf.layers.dense(
                        x, convnet_pars['n_features'],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='features_' + str(i)
                    ))
                    self._features2.append(tf.layers.dense(
                        self._features[i], convnet_pars['n_features'],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='features2_' + str(i)
                    ))
                    self._q.append(tf.layers.dense(
                        self._features2[i],
                        convnet_pars['output_shape'][0],
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='q_' + str(i)
                    ))
                    self._q_acted.append(
                        tf.reduce_sum(self._q[i] * action_one_hot,
                                      axis=1,
                                      name='q_acted_' + str(i))
                    )

            self._target_q = tf.placeholder(
                'float32',
                [None, convnet_pars['n_approximators']],
                name='target_q'
            )
            loss = 0.
            for i in range(convnet_pars['n_approximators']):
                loss += tf.losses.huber_loss(
                    self._mask[:, i] * self._target_q[:, i],
                    self._mask[:, i] * self._q_acted[i]
                )
            tf.summary.scalar('huber_loss', loss)
            tf.summary.scalar('average_q', tf.reduce_mean(self._q))
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
            elif optimizer['name'] == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate=optimizer['lr'],
                                                decay=optimizer['decay'],
                                                epsilon=optimizer['epsilon'])
            elif optimizer['name'] == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=optimizer['lr'])
            elif optimizer['name'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate=optimizer['lr'])
            else:
                raise ValueError('Unavailable optimizer selected.')

            self._train_step = opt.minimize(loss=loss)

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

        for i in range(len(self._features)):
            tf.add_to_collection(self._scope_name + '_features_' + str(i),
                                 self._features[i])
            tf.add_to_collection(self._scope_name + '_features2_' + str(i),
                                 self._features2[i])
            tf.add_to_collection(self._scope_name + '_q_' + str(i), self._q[i])
            tf.add_to_collection(self._scope_name + '_q_acted_' + str(i),
                                 self._q_acted[i])
        tf.add_to_collection(self._scope_name + '_target_q', self._target_q)
        tf.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.add_to_collection(self._scope_name + '_train_step', self._train_step)
        tf.add_to_collection(self._scope_name + '_mask', self._mask)

    def _restore_collection(self, convnet_pars):
        self._x = tf.get_collection(self._scope_name + '_x')[0]
        self._action = tf.get_collection(self._scope_name + '_action')[0]

        features = list()
        features2 = list()
        q = list()
        q_acted = list()
        for i in range(convnet_pars['n_approximators']):
            features.append(tf.get_collection(
                self._scope_name + '_features_' + str(i))[0])
            features2.append(tf.get_collection(
                self._scope_name + '_features2_' + str(i))[0])
            q.append(tf.get_collection(self._scope_name + '_q_' + str(i))[0])
            q_acted.append(tf.get_collection(
                self._scope_name + '_q_acted_' + str(i))[0])

        self._features = features
        self._features2 = features2
        self._q = q
        self._q_acted = q_acted
        self._target_q = tf.get_collection(self._scope_name + '_target_q')[0]
        self._merged = tf.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.get_collection(
            self._scope_name + '_train_step')[0]

        ##needs to be saved
        self._mask = tf.placeholder(
            tf.float32, shape=[None, convnet_pars['n_approximators']])
        # self._mask = tf.get_collection( self._scope_name + '_mask')[0]


        self._train_count = 0