import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
time_start = datetime.now()
time_start_formatted = time_start.strftime("%Y-%m-%d %H_%M_%S")

a=10
x = np.random.exponential(scale=1/a, size=1000)
plt.figure()
plt.hist(x)
plt.savefig('/home/lixiangchen/MyWorkspace/InferenceOffloadingDRL/Results/' + time_start_formatted)

