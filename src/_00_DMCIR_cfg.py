import numpy as np


def GetFilePrefix_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial):
    file_prefix = "DMCIR_%s_%s_ebeta%.3f_T%d_experts%d_h%d_fd%d_t%d" \
                  % (dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    return file_prefix


def GetModelFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial):
    file_prefix = GetFilePrefix_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    file_name = file_prefix + "_model"
    return file_name


def GetLogFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial):
    file_prefix = GetFilePrefix_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    file_name = file_prefix + "_log.txt"
    return file_name


def GetResultFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial):
    file_prefix = GetFilePrefix_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    file_name = file_prefix + "_result.txt"
    return file_name


def GetItemVecFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial):
    file_prefix = GetFilePrefix_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    file_name = file_prefix + "_itemvec.npy"
    return file_name


def GetFileAllinOne_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial):
    model_prefix = GetModelFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    log_file = GetLogFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    result_file = GetResultFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    item_vec_file = GetItemVecFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial)
    return model_prefix, log_file, result_file, item_vec_file


def GetFilePrefix_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                         r_hlayer_num, do, secRec_trial):
    file_prefix = "DMCIRPlus_%s_%s_ebeta%.3f_T%d_experts%d_h%d_f%d_t%d_rh%d_do%.2f_st%d" \
                  % (dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                     r_hlayer_num, do, secRec_trial)
    return file_prefix


def GetModelFileName_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                          r_hlayer_num, do, secRec_trial):
    file_prefix = GetFilePrefix_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                                        r_hlayer_num, do, secRec_trial)
    file_name = file_prefix + "_model"
    return file_name


def GetLogFileName_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                          r_hlayer_num, do, secRec_trial):
    file_prefix = GetFilePrefix_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                                        r_hlayer_num, do, secRec_trial)
    file_name = file_prefix + "_log.txt"
    return file_name


def GetResultFileName_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                             r_hlayer_num, do, secRec_trial):
    file_prefix = GetFilePrefix_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                                        r_hlayer_num, do, secRec_trial)
    file_name = file_prefix + "_result.txt"
    return file_name


def GetFileAllinOne_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                           r_hlayer_num, do, secRec_trial):
    model_file = GetModelFileName_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                                       r_hlayer_num, do, secRec_trial)
    log_file = GetLogFileName_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                                     r_hlayer_num, do, secRec_trial)
    result_file = GetResultFileName_DMCIRPlus(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts, d_hlayer_num, fold, trial,
                                           r_hlayer_num, do, secRec_trial)
    item_vec_file = GetItemVecFileName_DMCIR(dataset, noise_type, d_end_beta, d_max_t, d_moe_num_experts,
                                               d_hlayer_num, fold, trial)
    return model_file, log_file, result_file, item_vec_file


class DMCIRCfg:
    def __init__(self):
        ################### Cfg for evaluation data - start
        self.data_path = ""
        self.eval_batch_size = 10000  # the batch size for accuracy evaluation
        self.fold = 0
        self.eval_k_list = [10, 20, 50]

        ################### Cfg for evaluation data - end

        ##################### Cfg for diffusions - start
        ##### General
        self.d_trial = 0
        self.d_lb_type = 1   # 1: x0 lb, 0: noise lb
        self.d_predictor_hlayers = [200, 200]  # predictor hlayer config,
        self.d_batch_size = 400
        self.d_epochs = 160
        self.d_lr = 0.001         # learning rate
        self.d_reg_w = 0.00001
        self.d_emb_dim = 200       # input dimension
        self.d_cond_dim = 300      # condition dimension
        self.d_cond_emb_dim = 200  # condition embedding dimension
        self.d_clip = np.array([-5.0, 5.0])
        self.d_moe_num_experts = 5
        ##### Input dropout: 0.0 for default

        ##### Classifier-free
        self.d_gamma = 1.0  ## Classifier-free guidance param

        ##### Noise schedule
        self.d_max_t = 400
        self.d_noise_mode = "linear"
        self.d_start_beta = 0.0001
        self.d_end_beta = 0.02

        ##### Model save
        self.d_model_dir = ""
        self.d_log_dir = ""
        self.d_result_dir = ""
        self.d_model_prefix = ""
        self.d_log_path = ""
        self.d_result_path = ""
        self.d_eval_start = 0
        self.d_eval_period = 20
        self.d_estop_limit = 3
        #### item vec gen
        self.i_gen = True
        self.i_gen_file_dir = "../model/"
        self.i_gen_file_path = ""
        self.i_num_samples = 1
        self.i_gen_use_average = False
        ######################## Cfg for diffusions - end

        ##################### Cfg for 2nd stage Recommender - start
        #### General
        self.r_trial = 0
        self.r_hlayers = [200, 200]
        self.r_num_epochs = 10
        self.r_estop_limit = 3
        self.r_num_neg_samples = 5
        self.r_cf_vec_dropout = 0.5
        self.r_data_batch_size = 1024
        self.r_model_dir = ""
        self.r_log_dir = ""
        self.r_result_dir = ""
        self.r_model_prefix = ""
        self.r_log_path = ""
        self.r_result_path = ""

        #### Optimizer cfg
        self.r_lr = 0.0001
        self.r_decay_lr_every = 10  # learning rate decay for each epoch?
        self.r_lr_decay = 0.8
        self.r_param_L2_w = 0.0001
        ##################### Cfg for 2nd stage Recommender - end

    def SetDefault_CiteULike(self):
        self.fold = 0
        self.data_path = "../data/CiteULike/"
        self.d_model_dir = "../model/CiteULike/Diff/"
        self.d_log_dir = "../model/CiteULike/Diff/Log/"
        self.d_result_dir = "../model/CiteULike/Diff/Result/"
        self.i_gen_file_dir = "../model/CiteULike/"
        self.r_model_dir = "../model/CiteULike/2ndRec/"
        self.r_log_dir = "../model/CiteULike/2ndRec/Log/"
        self.r_result_dir = "../model/CiteULike/2ndRec/Result/"


        self.d_lr = 0.001  # learning rate
        self.d_epochs = 100
        self.d_predictor_hlayers = [200, 200]  # predictor hlayer config
        self.d_clip = np.array([-1.0, 1.0])
        self.d_emb_dim = 200       # input dimension
        self.d_cond_dim = 300      # condition dimension
        self.d_cond_emb_dim = 200  # condition embedding dimension
        self.d_eval_period = 5
        self.d_estop_limit = 5
        self.d_moe_num_experts = 5
        self.d_gamma = 1.0  ## Classifier-free guidance param

        self.d_max_t = 400         # 400
        self.d_noise_mode = "linear"
        self.d_start_beta = 0.0001
        self.d_end_beta = 0.03     # 0.02

        self.d_model_prefix = ""
        self.d_log_path = ""
        self.d_result_path = ""
        self.i_gen_file_path = ""
        self.i_gen_use_average = False

        self.r_trial = 0
        self.r_hlayers = [200, 200]
        self.r_num_epochs = 100
        self.r_estop_limit = 30
        self.r_num_neg_samples = 5
        self.r_cf_vec_dropout = 0.5
        self.r_data_batch_size = 1024
        self.r_log_path = ""
        self.r_result_path = ""
        #### Optimizer cfg
        self.r_lr = 0.005
        self.r_decay_lr_every = 10  # learning rate decay for each epoch?
        self.r_lr_decay = 0.8
        self.r_param_L2_w = 0.0001

        return

    def SetDefault_ML25M(self, fold=0):
        self.fold = fold
        self.data_path = "../data/ML25M/"
        self.d_model_dir = "../model/ML25M/Diff/"
        self.d_log_dir = "../model/ML25M/Diff/Log/"
        self.d_result_dir = "../model/ML25M/Diff/Result/"
        self.i_gen_file_dir = "../model/ML25M/"
        self.r_model_dir = "../model/ML25M/2ndRec/"
        self.r_log_dir = "../model/ML25M/2ndRec/Log/"
        self.r_result_dir = "../model/ML25M/2ndRec/Result/"

        self.eval_batch_size = 100000  # the batch size for accuracy evaluation

        self.d_lb_type = 1  # 1: x0 lb, 0: noise lb
        self.d_predictor_hlayers = [128, 128]  # predictor hlayer config
        self.d_clip = np.array([-1.0, 1.0])
        self.d_epochs = 100
        self.d_emb_dim = 128       # input dimension
        self.d_cond_dim = 400      # condition dimension
        self.d_cond_emb_dim = 128  # condition embedding dimension
        self.d_eval_start = 0
        self.d_eval_period = 20
        self.d_moe_num_experts = 5
        ##### Noise schedule
        self.d_max_t = 400         # 400
        self.d_start_beta = 0.0001
        self.d_end_beta = 0.03  ### 0.02

        self.i_gen_file_path = ""
        self.i_num_samples = 1
        self.i_gen_use_average = True

        self.r_num_epochs = 10
        self.r_estop_limit = 2
        self.r_num_neg_samples = 5
        self.r_cf_vec_dropout = 0.5
        self.r_data_batch_size = 1024 * 10
        #### Optimizer cfg
        self.r_lr = 0.005
        self.r_decay_lr_every = 10  # learning rate decay for each epoch?
        self.r_lr_decay = 0.8
        self.r_param_L2_w = 0.0001  # default : 0.0001

        return
