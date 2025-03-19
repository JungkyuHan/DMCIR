import U_utils as utils
import U_data as data
import numpy as np
import pandas as pd


def load_data(data_path, fold):
    timer = utils.timer(name='main').tic()
    item_content_file = data_path + '/mv-tag-emb.npy'
    train_file = data_path + "/BPR_cv/BPR_tr_{fold}.tsv".format(fold=fold)
    test_file = data_path + "/BPR_cv/cold_movies_rating_test_{fold}.tsv".format(fold=fold)
    validation_file = data_path + "/BPR_cv/cold_movies_rating_vali_{fold}.tsv".format(fold=fold)
    user_cf_vec_file = data_path + "/BPR_cv/BPR_uvec_{fold}.npy".format(fold=fold)
    item_cf_vec_file = data_path + "/BPR_cv/BPR_ivec_{fold}.npy".format(fold=fold)

    info_file = data_path + "/stats.tsv"
    info = pd.read_csv(info_file, dtype=np.int32, delimiter='\t')
    num_users = info['users'][0]
    num_items = info['movies'][0]

    timer.toc('loaded num_users:%d, num_items:%d' % (num_users, num_items))

    dat = {}
    # load preference data
    timer.tic()
    user_cf_vec = np.load(user_cf_vec_file)
    item_cf_vec = np.load(item_cf_vec_file)
    dat['num_users'] = num_users
    dat['num_items'] = num_items
    dat['user_cf_vec'] = user_cf_vec
    dat['item_cf_vec'] = item_cf_vec

    timer.toc('loaded U:%s,V:%s' % (str(user_cf_vec.shape), str(item_cf_vec.shape)))

    # load content data
    timer.tic()
    item_content_vec = np.load(item_content_file)

    dat['item_content_vec'] = item_content_vec
    timer.toc('loaded item feature sparse matrix: %s' % (str(item_content_vec.shape)))

    # load split
    timer.tic()
    uid_col = "uid"
    iid_col = "mid"
    dtype = {uid_col: np.int32, iid_col: np.int32}
    train = pd.read_csv(train_file, sep='\t', usecols=[uid_col, iid_col], dtype=dtype)
    dat['user_list'] = train[uid_col].values
    dat['item_list'] = train[iid_col].values
    dat['user_indices'] = np.unique(train[uid_col].values)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    dat['test_eval'] = data.load_eval_data_ML(test_file=test_file, uid_col=uid_col, iid_col=iid_col)
    dat['validation_eval'] = data.load_eval_data_ML(test_file=validation_file, uid_col=uid_col, iid_col=iid_col)
    return dat

