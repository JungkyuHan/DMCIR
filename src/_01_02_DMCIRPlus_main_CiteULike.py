import U_utils as utils
import numpy as np
import tensorflow as tf
import U_load_data_CiteULike as load_data_CiteULike
from _00_DMCIR_cfg import DMCIRCfg, GetFileAllinOne_DMCIRPlus, GetItemVecFileName_DMCIR

import argparse
from datetime import datetime
from DMCIRPlus import DMCIRPlus
import os


def main(cfg):
    seed = args.rand_seed
    if seed == 0 or seed > 1000:
        seed = seed
    else:
        seed = int(datetime.now().timestamp())
    print("seed: %d" % seed)
    tf.keras.utils.set_random_seed(seed)
    idx_gpu_used = args.gpu_idx
    list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in list_gpu:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(list_gpu[idx_gpu_used], 'GPU')

    dat = load_data_CiteULike.load_data(cfg.data_path)
    user_cf_vec = dat['user_cf_vec'] # normalized bpr user vec
    item_cf_vec = dat['item_cf_vec'] # normalized bpr item vec
    item_content_vec = dat['item_content_vec']
    test_eval = dat['test_eval']
    validation_eval = dat['validation_eval']

    user_list = dat['user_list']
    item_list = dat['item_list']
    item_warm = np.unique(item_list)

    timer = utils.timer(name='main').tic()
    # prep eval
    timer.tic()
    item_cf_vec_gen = np.load(cfg.i_gen_file_path)
    test_eval.init_tf(user_cf_vec, item_cf_vec, item_content_vec, None,
                      cfg.eval_batch_size, cold_item=True, item_factors_gen=item_cf_vec_gen)  # init data for evaluation
    validation_eval.init_tf(user_cf_vec, item_cf_vec, item_content_vec, None,
                            cfg.eval_batch_size, cold_item=True, item_factors_gen=item_cf_vec_gen)  # init data for evaluation

    timer.toc('initialized eval data').tic()

    cf_vec_rank = user_cf_vec.shape[1]
    item_content_vec_rank = item_content_vec.shape[1]

    dmcir_plus = DMCIRPlus(cf_vec_rank=cf_vec_rank,
                       item_content_vec_rank=item_content_vec_rank,
                       learning_rate=cfg.r_lr, lr_decay=cfg.r_lr_decay, lr_decay_step=cfg.r_decay_lr_every,
                       cf_vec_dropout=cfg.r_cf_vec_dropout, layer_weight_reg=cfg.r_param_L2_w,
                       hlayers=cfg.r_hlayers,
                       fold=cfg.fold, trial=cfg.r_trial, seed=seed)
    dmcir_plus.build_model()
    dmcir_plus.fit(user_tr_list=user_list, item_tr_list=item_list,
               u_pref=user_cf_vec,
               v_pref=item_cf_vec, v_pref_gen=item_cf_vec_gen, v_content=item_content_vec, item_warm=item_warm,
               num_negative_samples=cfg.r_num_neg_samples, data_batch_size=cfg.r_data_batch_size,
               dropout=cfg.r_cf_vec_dropout,
               epochs=cfg.r_num_epochs, estop_limit=cfg.r_estop_limit,
               tuning_data=validation_eval, test_data=test_eval,
               model_prefix=cfg.r_model_prefix, log_path=cfg.r_log_path, result_path=cfg.r_result_path,
               eval_top_k_list=cfg.eval_k_list, eval_top_k_list_test=cfg.eval_k_list)
    print("Train finished. Bye~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMCIRPlus_CiteULike",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_idx', type=int, default=0, help='Using GPU Idx')
    parser.add_argument('--rand_seed', type=int, default=100, help='use random seed')
    parser.add_argument('--dn_type', type=str, default="linear", help='linear/cosine')
    parser.add_argument('--dn_end', type=float, default=0.05, help='linear noise end')
    parser.add_argument('--d_t', type=int, default=400, help='diffusion step')
    parser.add_argument('--d_ex', type=int, default=3, help='num experts')
    parser.add_argument('--d_hlayers', type=int, nargs='+', default=[200, 200], help='diff hlayers')
    parser.add_argument('--fold', type=int, default=0, help='CV fold')
    parser.add_argument('--d_trial', type=int, default=0, help='trial')
    parser.add_argument('--r_hlayers', type=int, nargs='+', default=[200], help='rec hlayers')
    parser.add_argument('--r_cfdo', type=float, default=0.5, help='rec hlayers')
    parser.add_argument('--r_trial', type=int, default=0, help='trial')
    parser.add_argument('--r_epochs', type=int, default=100, help='rec epochs')

    args = parser.parse_args()
    args, _ = parser.parse_known_args()
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))

    cfg = DMCIRCfg()
    cfg.SetDefault_CiteULike()

    cfg.d_noise_mode = args.dn_type
    cfg.d_end_beta = args.dn_end
    cfg.d_max_t = args.d_t
    cfg.d_moe_num_experts = args.d_ex
    cfg.d_predictor_hlayers = args.d_hlayers
    cfg.fold = args.fold
    cfg.d_trial = args.d_trial

    cfg.r_hlayers = args.r_hlayers
    cfg.r_cf_vec_dropout = args.r_cfdo
    cfg.r_trial = args.r_trial
    cfg.r_num_epochs = args.r_epochs

    model_file_name, log_file_name, result_file_name, i_gen_file_name = GetFileAllinOne_DMCIRPlus(dataset="CL",
                                               noise_type=cfg.d_noise_mode, d_end_beta=cfg.d_end_beta,
                                               d_max_t=cfg.d_max_t, d_moe_num_experts=cfg.d_moe_num_experts,
                                               d_hlayer_num=len(cfg.d_predictor_hlayers),
                                               fold=cfg.fold, trial=cfg.d_trial,
                                               r_hlayer_num=len(cfg.r_hlayers), do=cfg.r_cf_vec_dropout,
                                               secRec_trial=cfg.r_trial)
    if not os.path.exists(cfg.r_model_dir):
        os.makedirs(cfg.r_model_dir)
    if not os.path.exists(cfg.r_log_dir):
        os.makedirs(cfg.r_log_dir)
    if not os.path.exists(cfg.r_result_dir):
        os.makedirs(cfg.r_result_dir)

    cfg.r_model_prefix = cfg.r_model_dir + model_file_name
    cfg.r_log_path = cfg.r_log_dir + log_file_name
    cfg.r_result_path = cfg.r_result_dir + result_file_name
    cfg.i_gen_file_path = cfg.i_gen_file_dir + i_gen_file_name

    main(cfg)
