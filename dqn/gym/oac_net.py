import numpy as np
import tensorflow as tf


class SimpleNet:
    @staticmethod
    def triple_loss(particles, targets, k, margin):
        loss = 0
        for i in range(k):
            d_p = tf.reduce_mean(tf.square(particles[:, i] - targets[:, i]), 0)
            if i == 0:
                d_n = tf.reduce_mean(tf.square(particles[:, i] - targets[:, i+1]), 0)
            elif i == k-1:
                d_n = tf.reduce_mean(tf.square(particles[:, i] - targets[:, i - 1]), 0)
            else:
                d_n = 0.5 * tf.reduce_mean(tf.square(particles[:, i] - targets[:, i - 1]), 0) + \
                      0.5 * tf.reduce_mean(tf.square(particles[:, i] - targets[:, i - 1]), 0)

            l = tf.maximum(d_p, margin + d_p - d_n)
            loss += l
        return loss / k

    def __init__(self, name=None, folder_name=None, load_path=None,
                 **convnet_pars):
        self._name = name
        self._folder_name = folder_name

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.convnet_pars = convnet_pars
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

    def predict(self, s, a, idx=None):
        if idx is not None:
            return self._session.run(self._q[idx], feed_dict={self._x: s, self._action: a})
        else:
            #s = np.array([s])
            return np.array(
                [self._session.run(self._q, feed_dict={self._x: s, self._action: a})])

    def fit(self, s, a, q, prob_exploration, margin):
        #s = np.transpose(s, [0, 2, 3, 1])
        #s = np.array([s])
        summaries, _ = self._session.run(
            [self._merged, self._train_step],
            feed_dict={self._x: s,
                       self._action: a,
                       self._target_q: q,
                       self._prob_exploration: prob_exploration,
                       self._margin: margin}
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
        self._folder_name = path
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

            if convnet_pars['n_states'] is not None:
                x = tf.one_hot(tf.cast(self._x[..., 0], tf.int32),
                               convnet_pars['n_states'])
            else:
                x = self._x[...]

            x = tf.stack([x, self._action], axis=0)
            self.n_approximators = convnet_pars['n_approximators']
            self.q_min = convnet_pars['q_min']
            self.q_max = convnet_pars['q_max']
            self.init_type = convnet_pars['init_type']

            if self.init_type == 'boot':
                kernel_initializer = lambda _: tf.glorot_uniform_initializer()
                bias_initializer = lambda _: tf.zeros_initializer()
            else:
                initial_values = np.linspace(self.q_min, self.q_max, self.n_approximators)
                kernel_initializer = tf.glorot_uniform_initializer()
                bias_initializer = tf.constant_initializer(initial_values)

            with tf.variable_scope('q'):
                if convnet_pars["net_type"] == 'features':
                    self._features = tf.layers.dense(
                        x, 24,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='features'
                    )
                    self._features2 = tf.layers.dense(
                        self._features, 48,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='features2'
                    )
                    self._q = tf.layers.dense(
                        self._features2,
                        self.n_approximators,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        name='q'
                    )
                else:
                    self._q = tf.layers.dense(
                        x,
                        self.n_approximators,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        name='q'
                    )

            self._target_q = tf.placeholder(
                'float32',
                [None, convnet_pars['n_approximators']],
                name='target_q'
            )
            self._margin = tf.placeholder('float32', (),
                                                    name='margin')
            self._q_sorted = tf.contrib.framework.sort(self._q, axis=1)
            self._target_q_sorted = tf.contrib.framework.sort(self._target_q, axis=1)
            loss = 0.
            if convnet_pars["loss"] == "huber_loss":
                self.loss_fuction = tf.losses.huber_loss
            else:
                self.loss_fuction = tf.losses.mean_squared_error
            k = convnet_pars['n_approximators']
            if convnet_pars["loss"] == "triple_loss":
                loss = SimpleNet.triple_loss(self._q_sorted, self._target_q_sorted, k , self._margin)
            else:
                for i in range(convnet_pars['n_approximators']):
                    loss += self.loss_fuction(
                        self._target_q_sorted[:, i],
                        self._q_sorted[:, i]
                    )
            self._prob_exploration = tf.placeholder('float32', (),
                                                    name='prob_exploration')
            tf.summary.scalar(convnet_pars["loss"], loss)
            tf.summary.scalar('average_q', tf.reduce_mean(self._q))
            # tf.summary.scalar('average_std', tf.reduce_mean(tf.sqrt(tf.nn.moments(self._q, axes=[0])[1])))
            tf.summary.scalar('prob_exploration', self._prob_exploration)
            tf.summary.scalar('std_acted', tf.reduce_mean(tf.nn.moments(self._q, axes=1)[1]))
            tf.summary.histogram('qs', tf.reduce_mean(self._q,axis=0))

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

        if self.convnet_pars['net_type'] == 'features':
            tf.add_to_collection(self._scope_name + '_features',
                                 self._features)
            tf.add_to_collection(self._scope_name + '_features2',
                                 self._features2)
        tf.add_to_collection(self._scope_name + '_q', self._q)
        tf.add_to_collection(self._scope_name + '_target_q', self._target_q)
        tf.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.add_to_collection(self._scope_name + '_train_step', self._train_step)

    def _restore_collection(self, convnet_pars):
        self._x = tf.get_collection(self._scope_name + '_x')[0]
        self._action = tf.get_collection(self._scope_name + '_action')[0]

        if self.convnet_pars['net_type'] == 'features':
            self._features= tf.get_collection(
                self._scope_name + '_features')[0]
            self.features2 = tf.get_collection(
                self._scope_name + '_features2')[0]
        self._q = tf.get_collection(self._scope_name + '_q')[0]
        self._target_q = tf.get_collection(self._scope_name + '_target_q')[0]
        self._merged = tf.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.get_collection(
            self._scope_name + '_train_step')[0]
        self._train_count = 0