
import numpy as np
from scipy.stats import chi2
d = 23
p = 0.95  # 置信度
threshold = np.sqrt(chi2.ppf(p, d))
print(threshold)