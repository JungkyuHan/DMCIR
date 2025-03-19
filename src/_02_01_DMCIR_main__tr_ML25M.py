import U_utils as utils
import numpy as np
import tensorflow as tf
import U_load_data_ML25M as load_data_ML25M
from DMCIR import DMCIR
from _00_DMCIR_cfg import DMCIRCfg, GetFileAllinOne_DMCIR
from RecEvaluator import diffusion_test_with_genItemVecs
import argparse
from datetime import datetime
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

    dat = load_data_ML25M.load_data(cfg.data_path, fold=cfg.fold)
    user_cf_vec = dat['user_cf_vec']  # normalized bpr user vec
    item_cf_vec = dat['item_cf_vec']  # normalized bpr item vec

    # test
    item_content_vec = dat['item_content_vec']
    test_eval = dat['test_eval']
    validation_eval = dat['validation_eval']

    item_list = dat['item_list']
    item_warm = np.unique(item_list)
    timer = utils.timer(name='main').tic()
    # prep eval
    timer.tic()

    test_eval.init_tf(user_cf_vec, item_cf_vec, item_content_vec, None,
                       cfg.eval_batch_size, cold_item=True)  # init data for evaluation
    validation_eval.init_tf(user_cf_vec, item_cf_vec, item_content_vec, None,
                             cfg.eval_batch_size, cold_item=True)  # init data for evaluation

    timer.toc('initialized eval data').tic()

    ############################
    tr_item_cf_vec = item_cf_vec[item_warm, :]
    tr_item_content_vec = item_content_vec[item_warm, :]

    dmcir = DMCIR(emb_dim=cfg.d_emb_dim, cond_dim=cfg.d_cond_dim,
                                cond_emb_dim=cfg.d_cond_emb_dim, T=cfg.d_max_t,
                                num_experts=cfg.d_moe_num_experts,
                                gamma=cfg.d_gamma, lr=cfg.d_lr, reg_w=cfg.d_reg_w,
                                lb=cfg.d_lb_type, clip_range=cfg.d_clip,
                                start_beta=cfg.d_start_beta, end_beta=cfg.d_end_beta)
    dmcir.build_predictor(hlayer_config=cfg.d_predictor_hlayers)

    dmcir.fit(train_vecs=tr_item_cf_vec, train_conds=tr_item_content_vec,
                     val_data=validation_eval, test_data=test_eval,
                     data_batch_size=cfg.d_batch_size, epochs=cfg.d_epochs,
                     eval_start=cfg.d_eval_start, eval_period=cfg.d_eval_period, e_stop_limit=cfg.d_estop_limit,
                     model_path=cfg.d_model_prefix, log_path=cfg.d_log_path, result_path=cfg.d_result_path)

    if cfg.i_gen is True:
        print("Generates Item vecs")
        result_fd = open(cfg.d_result_path, "w")
        dmcir.load_weights(load_path=cfg.d_model_prefix)
        generated_ivecs = dmcir.gen_itemvecs(item_content_vec, num_samples=cfg.i_num_samples)
        np.save(cfg.i_gen_file_path, generated_ivecs)

        ####### Validation check
        validation_ivecs = generated_ivecs[validation_eval.test_item_ids, :]
        diffusion_test_with_genItemVecs(validation_eval, validation_ivecs, top_k_list=cfg.eval_k_list,
                                        result_label="Eval", fd=result_fd)
        test_ivecs = generated_ivecs[test_eval.test_item_ids, :]
        diffusion_test_with_genItemVecs(test_eval, test_ivecs, top_k_list=cfg.eval_k_list, result_label="Test",
                                        fd=result_fd)
        result_fd.close()

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMCIR_Tr_ML25M",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_idx', type=int, default=0, help='Using GPU Idx')
    parser.add_argument('--rand_seed', type=int, default=100, help='use random seed')
    parser.add_argument('--dn_type', type=str, default="linear", help='linear/cosine')
    parser.add_argument('--dn_end', type=float, default=0.05, help='linear noise end')
    parser.add_argument('--d_t', type=int, default=300, help='diffusion step')
    parser.add_argument('--d_ex', type=int, default=5, help='num experts')
    parser.add_argument('--d_hlayers', type=int, nargs='+', default=[128, 128], help='diff hlayers')
    parser.add_argument('--fold', type=int, default=0, help='CV fold')
    parser.add_argument('--d_trial', type=int, default=0, help='trial')
    parser.add_argument('--d_epochs', type=int, default=100, help='trial')

    args = parser.parse_args()
    args, _ = parser.parse_known_args()
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))

    print(args.d_hlayers)

    cfg = DMCIRCfg()
    cfg.SetDefault_ML25M(fold=args.fold)

    cfg.d_noise_mode = args.dn_type
    cfg.d_end_beta = args.dn_end
    cfg.d_max_t = args.d_t
    cfg.d_moe_num_experts = args.d_ex
    cfg.d_predictor_hlayers = args.d_hlayers
    cfg.fold = args.fold
    cfg.d_trial = args.d_trial
    cfg.d_epochs = args.d_epochs

    model_file_name, log_file_name, result_file_name, i_gen_file_name = GetFileAllinOne_DMCIR(dataset="ML",
                                               noise_type=cfg.d_noise_mode, d_end_beta=cfg.d_end_beta,
                                               d_max_t=cfg.d_max_t, d_moe_num_experts=cfg.d_moe_num_experts,
                                               d_hlayer_num=len(cfg.d_predictor_hlayers),
                                               fold=cfg.fold, trial=cfg.d_trial)

    if not os.path.exists(cfg.d_model_dir):
        os.makedirs(cfg.d_model_dir)
    if not os.path.exists(cfg.d_log_dir):
        os.makedirs(cfg.d_log_dir)
    if not os.path.exists(cfg.d_result_dir):
        os.makedirs(cfg.d_result_dir)
    if not os.path.exists(cfg.i_gen_file_dir):
        os.makedirs(cfg.i_gen_file_dir)

    cfg.d_model_prefix = cfg.d_model_dir + model_file_name
    cfg.d_log_path = cfg.d_log_dir + log_file_name
    cfg.d_result_path = cfg.d_result_dir + result_file_name
    cfg.i_gen_file_path = cfg.i_gen_file_dir + i_gen_file_name

    main(cfg)
