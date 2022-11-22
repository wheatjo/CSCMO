import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt

n_samples = 300

# X = np.concatenate((np.random.normal((-2, -2), size=(n_samples, 2)),
#     np.random.normal((2, 2), size=(n_samples, 2)) ))

X = np.random.random((300, 2))
fcm = FCM(n_clusters=5)
fcm.fit(X)

fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)

model = np.empty((1,2), dtype=object)
model[0, 1] = fcm
model[0, 0] = fcm
r = model[0, 1].predict(X)
print(r)