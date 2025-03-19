import numpy as np
import tensorflow as tf
from U_EvalMetricCalculator import EvalMetricCalculator


def predict(u_vecs, i_vecs, top_k=20):
    predicted_ratings = tf.linalg.matmul(u_vecs, i_vecs, transpose_b=True)
    top_k_val_mat, top_k_index_mat = tf.math.top_k(input=predicted_ratings, k=top_k, sorted=True)
    top_k_index_tensor = tf.convert_to_tensor(value=top_k_index_mat, dtype=tf.int32)
    top_k_val_tensor = tf.convert_to_tensor(value=top_k_val_mat, dtype=tf.float32)
    return top_k_index_tensor, top_k_val_tensor


def diffusion_test_with_genItemVecs(test_data, gen_ivecs, top_k_list, result_label="Test", fd=None):
    metric = EvalMetricCalculator(top_k_list=top_k_list)
    max_top_k = 1
    for k in top_k_list:
        if max_top_k < k:
            max_top_k = k
    v_pref_eval = gen_ivecs
    for i, (eval_start, eval_finish) in enumerate(test_data.eval_batch):
        u_pref_eval = test_data.U_pref_test[eval_start:eval_finish]

        top_k_index_mat, top_k_val_mat \
            = predict(u_pref_eval, v_pref_eval, top_k=max_top_k)

        metric.collect_predictions(top_k_index_mat=top_k_index_mat)

    precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list \
        = metric.get_metric_distributed(test_data=test_data)
    EvalMetricCalculator.print_test_result(top_k_list, precision_list, recall_list, ndcg_list,
                                           num_hits_list, num_gts_list, result_label, fd)

    return precision_list, recall_list, ndcg_list, num_hits_list, num_gts_list


def diffusion_test(test_data, ddpm, top_k_list, result_label="Test",
                   num_samples=1):

    v_content_eval = test_data.V_content_test
    v_pref_eval = ddpm.gen_itemvecs(item_content_vec=v_content_eval,
                                    num_samples=num_samples)

    return diffusion_test_with_genItemVecs(test_data=test_data, gen_ivecs=v_pref_eval,
                                           top_k_list=top_k_list, result_label=result_label)


def diffusion_test_2nd(test_data, ddpm, top_k_list, result_label="Test",
                       num_samples=1):

    v_content_eval = test_data.V_content_test
    v_pref_gen = test_data.V_pref_gen_test
    v_pref_eval = ddpm.gen_itemvecs_2nd(item_content_vec=v_content_eval, item_cf_gen=v_pref_gen,
                                    num_samples=num_samples)

    return diffusion_test_with_genItemVecs(test_data=test_data, gen_ivecs=v_pref_eval,
                                           top_k_list=top_k_list, result_label=result_label)

