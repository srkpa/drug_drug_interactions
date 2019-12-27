import logging
import os


# If not exists creates the specified folder
def prepare_data_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def setup_running_directories(setting_dir, model_dir, result_dir):
    if not os.path.exists(setting_dir):
        os.makedirs(setting_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


def save_experiment_settings(setting_dir, exp_prefix):
    setting_npy_path = os.path.join(setting_dir, exp_prefix + '.npy')
    logging.info('Setting of the experiment is saved to %s', setting_npy_path)
	# np.save(setting_npy_path, opt)


def combine(d1, d2):
	for (k, v) in d2.items():
		if k not in d1:
			d1[k] = v
		else:
			d1[k].extend(v)
	return d1
