import numpy as np


# Fourier feature: https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf
def state_feature(n_dim):
  # Construct features for the critic
  dim_state = 4
  feature = []
  for i in range((n_dim + 1) ** dim_state):
    index = i
    temp_c = []
    for j in range(dim_state):
      temp_c.append(index % (n_dim + 1))
      index //= n_dim + 1
    feature.append(temp_c)
  feature = np.asarray(feature)
  return feature

if __name__=='__main__':
  n_dim = 1
  state = np.array([0.1, 0.2, 0.3, 0.4])
  c_critic = state_feature(n_dim)
  print(c_critic.size)
  feature_cur = np.cos(np.pi * c_critic.dot(state))
  print(len(feature_cur))