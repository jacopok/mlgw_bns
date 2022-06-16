from mlgw_bns.model import *
import numpy as np
import matplotlib.pyplot as plt
from mlgw_bns.data_management import Residuals

f = np.linspace(20, 2048, num=1<<20)

mmodel = ModesModel(modes=[Mode(2,2)])
mmodel.load()
# mmodel.generate(100, 10000, 10000)
# mmodel.set_hyper_and_train_nn()

model = Model.default()

for lam in np.linspace(10, 1000, num=20):
    p = ParametersWithExtrinsic(1.2, lam, lam, 0, 0, 40, 0, 2.8)
    hpm, hcm = mmodel.predict(f, p)
    hpn, hcn = model.predict(f, p)

    res = Residuals(
        np.log(abs(hcm) / abs(hcn))[np.newaxis, :],
        (np.unwrap(np.angle(hcm)) - np.unwrap(np.angle(hcn)))[np.newaxis, :]
    )
    
    res.flatten_phase(f)

    plt.semilogx(f, res.amplitude_residuals[0])    
    # plt.plot(f, res.phase_residuals[0])
plt.show()

# plt.plot(f, mmodel.models[Mode(2, 2)].predict_amplitude_phase(f, p)[1])
plt.show()
