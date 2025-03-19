import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from tqdm import tqdm
from DDPMUtil import DDPMUtil
from RecEvaluator import diffusion_test
from U_RecordHolder import RecordHolder


class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        a = tf.math.log(10000.0)
        b = (self.half_dim - 1)
        self.emb = a / b
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


def TimeMLP(units, activation_fn=tf.keras.activations.swish):
    def apply(inputs):
        temb = tf.keras.layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = tf.keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply


class DMCIR:
    def __init__(self, emb_dim, cond_dim, cond_emb_dim, T, num_experts=5, gamma=0.8, lr=2e-4, reg_w=0.0001,
                 lb=0, clip_range=[-1.0, 1.0],
                 noise_mode="linear", start_beta=0.0001, end_beta=0.02, noise_scale=1.0,
                 loss_type="MSE"):
        self.emb_dim = emb_dim
        self.cond_dim = cond_dim
        self.cond_emb_dim = cond_emb_dim
        self.T = T
        self.num_experts = num_experts
        self.predictor = None
        self.tr_predictor = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss_type = loss_type
        self.loss = tf.keras.losses.MeanAbsoluteError()

        self.reg_w = reg_w
        self.clip_min = clip_range[0]
        self.clip_max = clip_range[1]
        self.gamma = gamma
        self.lb = lb # 0: eps Lower Bound, 1: x0 Lower Bound

        self.start_beta = start_beta
        self.end_beta = end_beta
        self.noise_mode = noise_mode
        self.ddpm_util = DDPMUtil(T, self.clip_min, self.clip_max)

        self.param_shape = self.ddpm_util.param_shape((-1, emb_dim))
        self.ddpm_util.gen_noise_schedule(start_beta=self.start_beta, end_beta=self.end_beta,
                                          noise_scale=1.0, mode=self.noise_mode)

    def build_predictor(self, hlayer_config=[100, 100]):
        num_experts = self.num_experts
        l2_regularizer = tf.keras.regularizers.l2(self.reg_w)
        outlayer_nodes = self.emb_dim

        data_input = tf.keras.Input(shape=[self.emb_dim, ], name="data_input", dtype=tf.float32)
        time_input = tf.keras.Input(shape=[], dtype=tf.int64, name="time_input")
        condition_input = tf.keras.Input(shape=[self.cond_dim, ], name="condition_input", dtype=tf.float32)
        dropout_indicator = tf.keras.Input(shape=[1, ], name='dropout_indicator', dtype=tf.float32)

        time_emb = TimeEmbedding(dim=4.0)(time_input)
        time_emb = TimeMLP(units=4, activation_fn=tf.keras.activations.swish)(time_emb)

        cond_f = condition_input

        cond_emb = tf.keras.layers.Dense(self.cond_emb_dim, activation=None,
                                         kernel_initializer=kernel_init(1.0),
                                         kernel_regularizer=l2_regularizer)(cond_f)

        do_coef = 1.0 - dropout_indicator
        cond_emb_after_do = do_coef * cond_emb
        if num_experts > 0:
            item_content_gate = tf.keras.layers.Dense(self.num_experts, activation="tanh",
                                                      kernel_initializer=kernel_init(1.0),
                                                      kernel_regularizer=l2_regularizer)(cond_emb_after_do)
            item_content_expert_list = []
            for i in range(num_experts):
                h = tf.keras.layers.Concatenate(axis=1)([cond_emb_after_do])
                l_out = tf.keras.layers.Dense(outlayer_nodes, activation=None,
                                              kernel_initializer=kernel_init(1.0),
                                              kernel_regularizer=l2_regularizer)(h)

                item_content_expert_list.append(tf.reshape(l_out, [-1, 1, self.emb_dim]))

            item_content_expert_concat = tf.concat(item_content_expert_list, 1)
            item_content_expert_concat = tf.linalg.matmul(
                tf.reshape(item_content_gate, [-1, 1, num_experts]), item_content_expert_concat)
            # size: batch_size X self.output_rank
            x_predicted_1 = tf.reshape(tf.nn.tanh(item_content_expert_concat), [-1, self.emb_dim])

            h = tf.keras.layers.Concatenate(axis=1)([data_input, time_emb, x_predicted_1, cond_emb_after_do])

            for nodes in hlayer_config:
                h = tf.keras.layers.Dense(nodes, activation="swish",
                                          kernel_initializer=kernel_init(1.0),
                                          kernel_regularizer=l2_regularizer)(h)

            x_predicted_2 = tf.keras.layers.Dense(outlayer_nodes, activation=None,
                                                  kernel_initializer=kernel_init(1.0),
                                                  kernel_regularizer=l2_regularizer)(h)

            tr_predictor = tf.keras.Model(inputs=[data_input, time_input, condition_input, dropout_indicator],
                                          outputs=[x_predicted_1, x_predicted_2])
            predictor = tf.keras.Model(inputs=[data_input, time_input, condition_input, dropout_indicator],
                                       outputs=[x_predicted_2])
        else:
            h = tf.keras.layers.Concatenate(axis=1)([data_input, time_emb, cond_emb_after_do])

            for nodes in hlayer_config:
                h = tf.keras.layers.Dense(nodes, activation="swish",
                                          kernel_initializer=kernel_init(1.0),
                                          kernel_regularizer=l2_regularizer)(h)

            x_predicted_2 = tf.keras.layers.Dense(outlayer_nodes, activation=None,
                                                  kernel_initializer=kernel_init(1.0),
                                                  kernel_regularizer=l2_regularizer)(h)

            tr_predictor = tf.keras.Model(inputs=[data_input, time_input, condition_input, dropout_indicator],
                                          outputs=[x_predicted_2, x_predicted_2])
            predictor = tf.keras.Model(inputs=[data_input, time_input, condition_input, dropout_indicator],
                                       outputs=[x_predicted_2])

        self.tr_predictor = tr_predictor
        self.predictor = predictor

    def get_model_full_path(self, model_dir, model_file_name):
        complete_file_path = os.path.join(model_dir)
        complete_file_path = os.path.join(complete_file_path, model_file_name)
        return complete_file_path

    def save_weights(self, save_path):
        self.predictor.save_weights(save_path)

    def load_weights(self, load_path):
        self.predictor.load_weights(load_path)

    def get_noised_data(self, x_0, t):
        eps = np.random.normal(0, 1, size=[self.emb_dim])
        eps = eps.astype(np.float32)
        x_t = self.ddpm_util.q_sample(x_0, t, eps, self.param_shape)
        return x_t

    def get_noised_data_w_eps(self, x_0, t, eps):
        x_t = self.ddpm_util.q_sample(x_0, t, eps, self.param_shape)
        return x_t

    def fit(self, train_vecs, train_conds, val_data, test_data, data_batch_size, epochs,
                    model_path=None, log_path=None, result_path = None, eval_start=1,
                    eval_period=20, e_stop_limit=3, top_k_list=[10, 20, 50, 100]):
        if log_path is not None:
            f_log = open(log_path, 'w')
        else:
            f_log = None

        e_stop_cnt = 0
        best_record = RecordHolder(top_k_list=top_k_list, test_top_k_list=top_k_list)
        for epoch in range(epochs):
            random_idx = np.random.permutation(train_vecs.shape[0])
            n_targets = len(random_idx)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]
            num_batchs = len(data_batch)
            gen = data_batch
            gen = tqdm(gen)

            prediction_loss_epoch = 0.0
            reg_loss_epoch = 0.0
            weighted_loss_epoch = 0.0
            dropout = 1.0 - self.gamma

            for itr_cnter, (start, stop) in enumerate(gen):
                batch_idx = random_idx[start:stop]
                bsz = len(batch_idx)
                batch_data = train_vecs[batch_idx]
                batch_conditions = train_conds[batch_idx]

                if dropout > 0.0:
                    n_to_drop = int(np.floor(dropout * bsz))  # number of u-i pairs to be dropped
                    zero_index = np.random.choice(np.arange(bsz), n_to_drop, replace=False)
                else:
                    zero_index = []
                time_steps = np.random.randint(low=1, high=self.T+1, size=[bsz])
                eps = np.random.normal(0, 1, size=[bsz, self.emb_dim])
                eps = eps.astype(np.float32)
                x_t = self.get_noised_data_w_eps(batch_data, time_steps, eps)

                dropout_indicator = np.zeros(bsz)
                if len(zero_index) > 0:
                    dropout_indicator[zero_index] = 1.0

                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    train_inputs = [x_t, time_steps, batch_conditions, dropout_indicator]
                    predictions1, predictions2 = self.tr_predictor(train_inputs, training=True)

                    reg_loss_batch = tf.math.add_n(self.predictor.losses)
                    if self.lb == 0:
                        prediction_loss_batch1 = self.loss(eps, predictions1)
                        prediction_loss_batch2 = self.loss(eps, predictions2)
                    else:
                        prediction_loss_batch1 = self.loss(batch_data, predictions1)
                        prediction_loss_batch2 = self.loss(batch_data, predictions2)

                    weighted_loss_batch = prediction_loss_batch1 + prediction_loss_batch2 + reg_loss_batch
                gradients = tape.gradient(weighted_loss_batch,
                                          self.predictor.trainable_weights)

                self.optimizer.apply_gradients(zip(gradients, self.predictor.trainable_weights))

                prediction_loss_epoch += (prediction_loss_batch1 + prediction_loss_batch2)
                reg_loss_epoch += reg_loss_batch
                weighted_loss_epoch += (prediction_loss_batch1 + prediction_loss_batch2) + reg_loss_batch

            prediction_loss_epoch = prediction_loss_epoch / num_batchs
            reg_loss_epoch = reg_loss_epoch / num_batchs
            weighted_loss_epoch = weighted_loss_epoch / num_batchs
            if self.lb == 0:
                print("loss sum in epoch %d/%d - all: %.8f, eps_loss: %.8f, reg: %.8f"
                      % (epoch + 1, epochs, weighted_loss_epoch, prediction_loss_epoch, reg_loss_epoch))
            else:
                print("loss sum in epoch %d/%d - all: %.8f, x0_loss: %.8f, reg: %.8f"
                      % (epoch + 1, epochs, weighted_loss_epoch, prediction_loss_epoch, reg_loss_epoch))

            if val_data is not None and (epoch + 1) >= eval_start \
                    and eval_period > 0 and (epoch + 1) % eval_period == 0:
                precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
                    = diffusion_test(val_data, self, top_k_list,
                                     num_samples=1, result_label="Eval")
                updated = best_record.update_best_records_based_xth_recall(epoch+1, precision_list, recall_list,
                                                                           ndcg_list,
                                                                           num_hits_list, num_gts_list)
                if updated:
                    self.save_weights(model_path)
                    e_stop_cnt = 0
                else:
                    e_stop_cnt += 1
                    if e_stop_cnt >= e_stop_limit:
                        print("No improvement. Early stop triggered.")
                        print("No improvement. Early stop triggered.", file=f_log)
                        break

                if updated and test_data is not None:
                    test_precision_list, test_recall_list, test_ndcg_list, test_num_hits_list, test_num_gts_list \
                        = diffusion_test(test_data, self, top_k_list,
                                         num_samples=1, result_label="Test")
                    best_record.update_test_records_when_best_records_got(test_precision_list, test_recall_list,
                                                                          test_ndcg_list,
                                                                          test_num_hits_list, test_num_gts_list)
                best_record.print_best_record(fd=f_log)

        if f_log is not None:
            f_log.close()

        if result_path is not None:
            f_result = open(result_path, 'w')
            best_record.print_best_record(fd=f_result)
            f_result.close()

    def generate(self, num_vecs=1, conditions=None, clip=True):
        np.random.seed(int(datetime.now().timestamp()))
        x_t = np.random.normal(0, 1, size=[num_vecs, self.emb_dim])
        x_t = x_t.astype(np.float32)
        x_t = np.clip(x_t, self.clip_min, self.clip_max)
        dropout_indicator = np.zeros(num_vecs, dtype=np.float32)

        if conditions is None:
            print("No conditions are given")

        for t in range(self.T, 0, -1):
            if t > 1:
                noise = np.random.normal(0, 1, size=[num_vecs, self.emb_dim])
            else:
                noise = np.zeros(shape=(num_vecs, self.emb_dim))
            noise = noise.astype(np.float32)
            time_steps = np.repeat(t, num_vecs)
            gen_inputs = [x_t, time_steps, conditions, dropout_indicator]
            predictions = self.predictor(gen_inputs, training=False)
            predictions = np.array(predictions)
            if self.lb == 0:
                x_t = self.ddpm_util.p_sample_eps_lb(x_t, t, predictions, noise, self.param_shape, clip)
            else:
                x_t = self.ddpm_util.p_sample_x0_lb(x_t, t, predictions, noise, self.param_shape, clip)
        return x_t

    def gen_itemvecs(self, item_content_vec, num_samples=1):
        n_items = np.shape(item_content_vec)[0]

        v_sample_list = []
        for i in range(num_samples):
            v_pref_gen_s = self.generate(n_items, item_content_vec)
            v_sample_list.append(v_pref_gen_s)

        d_stacked = np.dstack(v_sample_list)
        v_pref_gen = np.mean(d_stacked, axis=2)

        return v_pref_gen
