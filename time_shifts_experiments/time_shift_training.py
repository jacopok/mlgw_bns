import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Sans Serif']})
rc('text', usetex=True)

from sklearn.gaussian_process import GaussianProcessRegressor

training_timeshifts = np.load('time_shifts_experiments/data/timeshifts_5k.npy')
training_param_array = np.load('time_shifts_experiments/data/params_5k.npy')

print(training_param_array)
print(training_timeshifts)

param_names = ['q', 'l1', 'l2', 'c1', 'c2']

def scale(data):
    scale_values = np.array([1, 5000, 5000, 1, 1])
    shift_values = np.array([0, 0, 0, 0.5, 0.5])
    return (data+shift_values[np.newaxis, :])/scale_values[np.newaxis, :]

def create_regressor(training_params, training_timeshifts):
    
    regressor = GaussianProcessRegressor()
    
    regressor.fit(scale(training_params), training_timeshifts)
    
    def predict(params):
        return regressor.predict(scale(params))
    
    return predict

def scatterplot():
    
    N_train = len(training_timeshifts) // 10 * 9
    
    predictor = create_regressor(training_param_array[:N_train], training_timeshifts[:N_train])
    
    prediction = predictor(training_param_array[N_train:])
    
    fig, axs = plt.subplots(1, 5, sharey=True)
    
    for i in range(5):
        axs[i].scatter(training_param_array[N_train:,i], (training_timeshifts[N_train:] - prediction) * 1e3, s=5)
        
        axs[i].set_xlabel(param_names[i])
    axs[0].set_ylabel('Residual timeshifts [ms]')
    plt.show()

def histogram():

    N_train = len(training_timeshifts) // 10 * 9
    
    predictor = create_regressor(training_param_array[:N_train], training_timeshifts[:N_train])
    
    prediction = predictor(training_param_array[N_train:])

    residuals = training_timeshifts[N_train:] - prediction
    
    plt.hist(np.log10(abs(residuals)), bins=50)
    plt.xlabel('$\log_{10} |\\Delta t - \\Delta t _{\\mathrm{pred}}|$')
    plt.title(f'Training: {N_train}, Testing: {len(residuals)}')
    plt.show()

if __name__ == '__main__':
    histogram()