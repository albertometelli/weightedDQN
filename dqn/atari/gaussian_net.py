import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U

class GaussianNet:
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

    def predict(self, s):
        s = np.transpose(s, [0, 2, 3, 1])

        return np.array([self._session.run([self._q, self._sigma], feed_dict={self._x: s})])

    def fit(self, s, a, q_and_sigma):
        s = np.transpose(s, [0, 2, 3, 1])
        summaries, _, loss = self._session.run(
            [self._merged, self._train_step, self.loss],
            feed_dict={self._x: s,
                       self._action: a.ravel().astype(np.uint8),
                       self._target_q: q_and_sigma[0],
                       self._target_sigma: q_and_sigma[1]}
        )
        print(loss)
        input()
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


            with tf.variable_scope('Convolutions'):
                hidden_1 = tf.layers.conv2d(
                    self._x / 255., 32, 8, 4, activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='hidden_1'
                )
                hidden_2 = tf.layers.conv2d(
                    hidden_1, 64, 4, 2, activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='hidden_2'
                )
                hidden_3 = tf.layers.conv2d(
                    hidden_2, 64, 3, 1, activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='hidden_3'
                )
                flatten = tf.reshape(hidden_3, [-1, 7 * 7 * 64], name='flatten')

                def scale_gradient():
                    with flatten.graph.gradient_override_map(
                            {'Identity': 'scaled_gradient_' + self._name}):
                        return tf.identity(flatten, name='identity')

                @tf.RegisterGradient('scaled_gradient_' + self._name)
                def scaled_gradient(op, grad):
                    return grad / float(2)

                identity = scale_gradient()


            self.sigma_weight = convnet_pars['sigma_weight']
            #self.n_approximators = convnet_pars['n_approximators']
            self.q_min = convnet_pars['q_min']
            self.q_max = convnet_pars['q_max']
            #self.init_type = convnet_pars['init_type']

            '''if self.init_type == 'boot':
                kernel_initializer = lambda _: tf.glorot_uniform_initializer()
                bias_initializer = lambda _: tf.zeros_initializer()
            else:
                initial_values = np.linspace(self.q_min, self.q_max, self.n_approximators)
                kernel_initializer = lambda _: tf.glorot_uniform_initializer()
                bias_initializer = lambda i: tf.constant_initializer(initial_values[i])'''
            mean = (self.q_min +self.q_max) /2.
            logsigma = np.log((self.q_max - self.q_min) / np.sqrt(12))

            with tf.variable_scope('q_value'):
                self.q_features = tf.layers.dense(
                    identity, 512, activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='q_features'
                )
                self._q = tf.layers.dense(
                    self.q_features,
                    convnet_pars['output_shape'][0],
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.constant_initializer(mean),
                    name='q'
                )
                self._q_acted = tf.reduce_sum(self._q * action_one_hot,
                                  axis=1,
                                  name='q_acted')
                with tf.variable_scope("sigma"):
                    self.sigma_features = tf.layers.dense(
                        identity, 512, activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        name='sigma_features'
                    )
                    self._log_sigma = tf.layers.dense(
                        self.sigma_features,
                        convnet_pars['output_shape'][0],
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.constant_initializer(0),
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

            zeros = tf.fill(tf.shape(self._action), 0.)

            loss = self.loss_fuction(zeros,(self._q_acted - self._target_q) ** 2 + \
                                     tf.math.scalar_mul(
                                         self.sigma_weight,
                                         (self._sigma_acted - self._target_sigma) ** 2))

            tf.summary.scalar(convnet_pars["loss"], loss)
            tf.summary.scalar('average_q', tf.reduce_mean(self._q))
            tf.summary.histogram('qs', self._q)
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

            '''q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope_name)
            sigma_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope_name + "/sigma")
            q_vars=[]
            for i, var in enumerate(q_func_vars):
                if var not in sigma_vars:
                    q_vars.append(var)
            q_gradients = opt.compute_gradients(loss=loss, var_list=q_vars)
            sigma_gradients = sigma_opt.compute_gradients(loss=loss, var_list=q_vars)
            self._train_step = tf.group(*[opt.apply_gradients(q_gradients), 
                                          sigma_opt.apply_gradients(sigma_gradients)])'''
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

    def _add_collection(self):
        tf.add_to_collection(self._scope_name + '_x', self._x)
        tf.add_to_collection(self._scope_name + '_action', self._action)

        tf.add_to_collection(self._scope_name + 'q_features',
                                 self.q_features)
        tf.add_to_collection(self._scope_name + 'q',self._q)
        tf.add_to_collection(self._scope_name + 'q_acted',
                                 self._q_acted)
        tf.add_to_collection(self._scope_name + 'sigma_features',
                             self.sigma_features)
        tf.add_to_collection(self._scope_name + 'log_sigma', self._log_sigma)
        tf.add_to_collection(self._scope_name + 'sigma', self._sigma)
        tf.add_to_collection(self._scope_name + 'sigma_acted',
                             self._q_acted)
        tf.add_to_collection(self._scope_name + '_target_q', self._target_q)
        tf.add_to_collection(self._scope_name + '_target_sigma', self._target_sigma)
        tf.add_to_collection(self._scope_name + '_merged', self._merged)
        tf.add_to_collection(self._scope_name + '_train_step', self._train_step)

    def _restore_collection(self, convnet_pars):
        self._x = tf.get_collection(self._scope_name + '_x')[0]
        self._action = tf.get_collection(self._scope_name + '_action')[0]

        self.q_features = tf.get_collection(self._scope_name + 'q_features')[0]
        self._q = tf.get_collection(self._scope_name + 'q')[0]
        self._q_acted = tf.get_collection(self._scope_name + 'q_acted')[0]
        self.sigma_features = tf.get_collection(self._scope_name + 'sigma_features')[0]
        self._log_sigma = tf.get_collection(self._scope_name + 'log_sigma')[0]
        self._sigma = tf.get_collection(self._scope_name + 'sigma')[0]
        self._sigma_acted = tf.get_collection(self._scope_name + 'sigma_acted')[0]

        self._target_q = tf.get_collection(self._scope_name + '_target_q')[0]
        self._target_sigma = tf.get_collection(self._scope_name + '_target_sigma')[0]
        self._merged = tf.get_collection(self._scope_name + '_merged')[0]
        self._train_step = tf.get_collection(
            self._scope_name + '_train_step')[0]

        self._train_count = 0
