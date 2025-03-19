import tensorflow as tf
import numpy as np
from U_EvalMetricCalculator import EvalMetricCalculator
from U_RecordHolder import RecordHolder
from tqdm import tqdm
import os.path
from U_CustomLayers import CustomSchedule
from U_CustomLayers import DenseBatchLayer
from U_CustomLayers import FCLayer


class DMCIRPlus:
    def __init__(self, cf_vec_rank, item_content_vec_rank,
                 learning_rate, lr_decay, lr_decay_step, cf_vec_dropout,
                 layer_weight_reg, hlayers=[200], fold=0, trial=0, seed=0):
        self.cf_vec_rank = cf_vec_rank
        self.item_content_vec_rank = item_content_vec_rank
        self.fold = fold
        self.trial = trial
        self.seed = seed
        self.hlayers = hlayers

        # optimizer
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step
        self.lr_schedule = CustomSchedule(initial_lr=learning_rate, decay=lr_decay, decay_step=lr_decay_step)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.9)

        self.predictor = None
        self.user_predictor = None
        self.item_predictor = None

        self.cf_vec_dropout = cf_vec_dropout
        self.layer_weight_reg = layer_weight_reg

    def print_configurations(self):
        print("/**********************[ALL Runtime Configurations]********************/")
        print("CV-fold: %d" % self.fold)
        print("trial: %d" % self.trial)
        print("cf_vec_rank: %d" % self.cf_vec_rank)
        print("item_content_vec_rank: %d" % self.item_content_vec_rank)
        print("cf_vec_dropout: %.2f" % self.cf_vec_dropout)
        print("layer_weight_reg: %f" % self.layer_weight_reg)
        print("learn rate: %f" % self.lr)
        print("learn rate decay: %f" % self.lr_decay)
        print("learn rate decay step: %d" % self.lr_decay_step)
        print("/**********************************************************************/")

    def build_model(self):
        self.model_name = "2ndStageRec"

        u_in = tf.keras.Input(shape=[self.cf_vec_rank, ], name="in_u_cf", dtype=tf.float32)
        v_in = tf.keras.Input(shape=[self.cf_vec_rank, ], name="in_v_cf", dtype=tf.float32)
        v_in_gen = tf.keras.Input(shape=[self.cf_vec_rank, ], name="in_v_cf_gen", dtype=tf.float32)
        dropout_indicator = tf.keras.Input(shape=(1,), name='dropout_indicator', dtype=tf.dtypes.float32)

        ############## Item Cold Start
        v_content_in = tf.keras.Input(shape=[self.item_content_vec_rank, ], name="in_v_content", dtype=tf.float32)

        v_content_emb = v_content_in

        cf_vec_filter = 1.0 - dropout_indicator
        item_selected_cf = (v_in * cf_vec_filter + v_in_gen * dropout_indicator)
        v_mid_layer = item_selected_cf
        u_mid_layer = u_in

        v_mid_layer = tf.concat([v_mid_layer, v_content_emb], axis=1)

        for ihid, hid in enumerate(self.hlayers):  # self.cf_vec_rank
            u_mid_layer = DenseBatchLayer(units=hid, is_training=True,
                                          do_norm=True, regularizer_weight=self.layer_weight_reg)(u_mid_layer)
            v_mid_layer = DenseBatchLayer(units=hid, is_training=True,
                                          do_norm=True, regularizer_weight=self.layer_weight_reg)(v_mid_layer)

        u_embedding_1 = tf.concat([u_mid_layer, u_in], axis=1)
        v_embedding_1 = tf.concat([v_mid_layer, item_selected_cf, v_content_emb], axis=1)

        u_embedding = FCLayer(units=self.cf_vec_rank, use_activation=False,
                              regularizer_weight=self.layer_weight_reg)(u_embedding_1)

        v_embedding = FCLayer(units=self.cf_vec_rank, use_activation=False,
                              regularizer_weight=self.layer_weight_reg)(v_embedding_1)

        prediction = tf.math.multiply(u_embedding, v_embedding)
        prediction = tf.math.reduce_sum(prediction, axis=1,
                                        keepdims=True)  # output of the model, the predicted scores

        model_inputs = [u_in, v_in, v_in_gen, dropout_indicator, v_content_in]
        model_outputs = [prediction]
        self.predictor = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

        user_predictor_inputs = [u_in]
        item_predictor_inputs = [v_in, v_in_gen, dropout_indicator, v_content_in]

        self.user_predictor = tf.keras.Model(inputs=user_predictor_inputs, outputs=u_embedding)
        self.item_predictor = tf.keras.Model(inputs=item_predictor_inputs, outputs=v_embedding)

    def save_weights(self, save_path):
        self.predictor.save_weights(save_path)

    def load_weights(self, load_path):
        self.predictor.load_weights(load_path)

    @staticmethod
    def negative_sampling(pos_user_array, pos_item_array, neg, item_warm):
        neg = int(neg)
        user_pos = pos_user_array.reshape((-1))
        user_neg = np.tile(pos_user_array, neg).reshape((-1))
        item_pos = pos_item_array.reshape((-1))
        item_neg = np.random.choice(item_warm, size=(neg * pos_user_array.shape[0]), replace=True).reshape((-1))
        target_pos = np.ones_like(item_pos)
        target_neg = np.zeros_like(item_neg)
        return np.concatenate((user_pos, user_neg)), np.concatenate((item_pos, item_neg)), \
               np.concatenate((target_pos, target_neg))

    def fit(self, user_tr_list, item_tr_list,
            u_pref,
            v_pref, v_pref_gen, v_content, item_warm,
            num_negative_samples=5, data_batch_size=1024, dropout=0.0, epochs=1, estop_limit=3,
            tuning_data=None, test_data=None,
            model_prefix=None, log_path=None, result_path=None,
            eval_top_k_list=[20], eval_top_k_list_test=[10, 20, 50, 100]):

        e_stop_cnt = 0

        if log_path is not None:
            f_log = open(log_path, 'w')
        else:
            f_log = None

        self.print_configurations()
        print("[Evaluation before train]")
        best_record = RecordHolder(top_k_list=eval_top_k_list, test_top_k_list=eval_top_k_list_test)

        if tuning_data is not None:
            precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
                = self.test(tuning_data, eval_top_k_list, "Eval")

        for epoch in range(epochs):
            self.lr_schedule.set_epoch(epoch + 1)
            user_array, item_array, target_array = DMCIRPlus.negative_sampling(user_tr_list, item_tr_list,
                                                                                  num_negative_samples, item_warm)

            random_idx = np.random.permutation(user_array.shape[0])
            n_targets = len(random_idx)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]

            # variables for loss monitoring -- start
            loss_epoch = 0.
            rating_loss_epoch = 0.
            regularization_loss_epoch = 0.
            regularization_loss_sum = 0.
            # variables for loss monitoring -- end

            gen = data_batch
            gen = tqdm(gen)

            for itr_cnter, (start, stop) in enumerate(gen):
                batch_idx = random_idx[start:stop]
                batch_users = user_array[batch_idx]
                batch_items = item_array[batch_idx]
                batch_targets = target_array[batch_idx]

                # dropout
                if dropout != 0:
                    n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                    zero_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                else:
                    zero_index = np.array([])

                user_cf_batch = u_pref[batch_users, :]
                item_cf_batch = v_pref[batch_items, :]
                item_cf_gen_batch = v_pref_gen[batch_items, :]
                item_content_batch = v_content[batch_items, :]

                target_batch_tf = tf.convert_to_tensor(batch_targets)
                num_targets = tf.shape(target_batch_tf)[0]
                target_batch_tf = tf.reshape(target_batch_tf, shape=[num_targets, 1])
                target_batch_tf = tf.cast(x=target_batch_tf, dtype=tf.float32)

                dropout_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                if len(zero_index) > 0:
                    dropout_indicator[zero_index] = 1.0

                train_inputs = [user_cf_batch, item_cf_batch, item_cf_gen_batch, dropout_indicator]
                train_inputs.append(item_content_batch)

                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    rating_predictions = self.predictor(train_inputs, training=True)

                    regularization_loss = tf.math.add_n(self.predictor.losses)

                    rating_prediction_loss_o = tf.math.reduce_mean(
                        tf.math.squared_difference(target_batch_tf, rating_predictions))

                    weighted_loss = rating_prediction_loss_o + regularization_loss

                gradients = tape.gradient(weighted_loss,
                                          self.predictor.trainable_weights)

                self.optimizer.apply_gradients(zip(gradients, self.predictor.trainable_weights))

                rating_loss_epoch += rating_prediction_loss_o
                regularization_loss_sum += regularization_loss  # because this loss is related to layer weights
                regularization_loss_epoch = regularization_loss_sum / itr_cnter
                weighted_loss_epoch = rating_prediction_loss_o
                loss_epoch += weighted_loss_epoch

            print("loss sum in epoch %d/%d - all: %.3f, rating: %.3f, reg: %.3f"
                  % (epoch+1, epochs, loss_epoch, rating_loss_epoch,
                     regularization_loss_epoch))

            if tuning_data is not None:
                precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
                    = self.test(tuning_data, eval_top_k_list, "Eval")
                updated = best_record.update_best_records_based_xth_recall(epoch+1, precision_list, recall_list,
                                                                           ndcg_list,
                                                                           num_hits_list, num_gts_list)

                if updated:
                    self.save_weights(model_prefix)
                    e_stop_cnt = 0
                else:
                    e_stop_cnt += 1
                    if e_stop_cnt >= estop_limit:
                        print("No improvement. Early stop triggered.")
                        break

                if updated and test_data is not None:
                    test_precision_list, test_recall_list, test_ndcg_list, test_num_hits_list, test_num_gts_list \
                        = self.test(test_data, eval_top_k_list_test, "Test")
                    best_record.update_test_records_when_best_records_got(test_precision_list, test_recall_list,
                                                                          test_ndcg_list,
                                                                          test_num_hits_list, test_num_gts_list)
                best_record.print_best_record(fd=f_log)

        if f_log is not None:
            f_log.close()

        if result_path is not None:
            f_result = open(result_path, 'w')
            best_record.print_best_record_simple(fd=f_result)
            f_result.close()

    def predict_4_item_cs(self, u_pref_tf, v_pref_gen_tf, v_content_tf, top_k):
        num_items = tf.shape(v_pref_gen_tf)[0]

        item_cf_vec_dropout = np.ones((num_items, 1))
        item_cf_vec_dropout_tf = tf.convert_to_tensor(item_cf_vec_dropout)
        v_pref_tf = np.zeros(np.shape(v_pref_gen_tf))
        item_vecs = self.item_predictor([v_pref_tf, v_pref_gen_tf, item_cf_vec_dropout_tf, v_content_tf], training=False)
        user_vecs = self.user_predictor([u_pref_tf], training=False)

        predicted_ratings = tf.linalg.matmul(user_vecs, item_vecs, transpose_b=True)
        top_k_val_mat, top_k_index_mat = tf.math.top_k(input=predicted_ratings, k=top_k, sorted=True)

        top_k_index_tensor = tf.convert_to_tensor(value=top_k_index_mat, dtype=tf.int32)
        top_k_val_tensor = tf.convert_to_tensor(value=top_k_val_mat, dtype=tf.float32)
        return top_k_index_tensor, top_k_val_tensor

    def test(self, test_data, top_k_list, result_label="Test"):
        metric = EvalMetricCalculator(top_k_list=top_k_list)
        max_top_k = 1
        for k in top_k_list:
            if max_top_k < k:
                max_top_k = k

        v_pref_gen_eval = test_data.V_pref_gen_test
        v_pref_gen_eval_tf = tf.convert_to_tensor(v_pref_gen_eval)

        v_content_eval = test_data.V_content_test
        v_content_eval_tf = tf.convert_to_tensor(v_content_eval)

        for i, (eval_start, eval_finish) in enumerate(test_data.eval_batch):
            u_pref_eval = test_data.U_pref_test[eval_start:eval_finish]
            u_pref_eval_tf = tf.convert_to_tensor(u_pref_eval)
            top_k_index_mat, top_k_val_mat \
                = self.predict_4_item_cs(u_pref_tf=u_pref_eval_tf,
                                         v_pref_gen_tf=v_pref_gen_eval_tf,
                                         v_content_tf=v_content_eval_tf,
                                         top_k=max_top_k)

            metric.collect_predictions(top_k_index_mat=top_k_index_mat)

        precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
            = metric.get_metric_distributed(test_data=test_data)
        EvalMetricCalculator.print_test_result(top_k_list, precision_list, recall_list, ndcg_list,
                                               num_hits_list, num_gts_list, result_label)

        return precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list

