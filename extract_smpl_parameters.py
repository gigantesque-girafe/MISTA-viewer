import os
import pickle as pkl
import numpy as np

from utils.paths import body_models_path

if __name__ == '__main__':
    male_path = body_models_path('smpl', 'male', 'model.pkl')
    female_path = body_models_path('smpl', 'female', 'model.pkl')
    neutral_path = body_models_path('smpl', 'neutral', 'model.pkl')

    data_m = pkl.load(open(male_path, 'rb'), encoding='latin1')
    data_f = pkl.load(open(female_path, 'rb'), encoding='latin1')
    data_n = pkl.load(open(neutral_path, 'rb'), encoding='latin1')

    misc_dir = body_models_path('misc')
    if not os.path.exists(misc_dir):
        os.makedirs(misc_dir)

    np.savez(os.path.join(misc_dir, 'faces.npz'), faces=data_m['f'].astype(np.int64))
    np.savez(os.path.join(misc_dir, 'J_regressors.npz'), male=data_m['J_regressor'].toarray(), female=data_f['J_regressor'].toarray(), neutral=data_n['J_regressor'].toarray())
    np.savez(os.path.join(misc_dir, 'posedirs_all.npz'), male=data_m['posedirs'], female=data_f['posedirs'], neutral=data_n['posedirs'])
    np.savez(os.path.join(misc_dir, 'shapedirs_all.npz'), male=data_m['shapedirs'], female=data_f['shapedirs'], neutral=data_n['shapedirs'])
    np.savez(os.path.join(misc_dir, 'skinning_weights_all.npz'), male=data_m['weights'], female=data_f['weights'], neutral=data_n['weights'])
    np.savez(os.path.join(misc_dir, 'v_templates.npz'), male=data_m['v_template'], female=data_f['v_template'], neutral=data_n['v_template'])
    np.save(os.path.join(misc_dir, 'kintree_table.npy'), data_m['kintree_table'].astype(np.int32))
