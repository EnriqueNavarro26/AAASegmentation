import pandas as pd
import matplotlib.pyplot as plt

# Create dataframe

k = 5

file_name = "training_FOLD" + str(k) + ".log"
df = pd.read_csv(file_name, sep=',')

plt.rcParams['axes.facecolor'] = (1.0, 1.0, 1.0)
plt.plot(df['epoch'], df['loss'], label='loss', linestyle='-', color='g')
plt.plot(df['epoch'], df['val_loss'], label='val_loss', linestyle='-')

# plt.plot(df['epoch'], df['jacard_coef_loss']*(-1), label='jaccard', linestyle='-', color='g')
# plt.plot(df['epoch'], df['val_jacard_coef_loss']*(-1), label='val_jaccard', linestyle='-')


plt.title(f'LOSS KFOLD {k}')
plt.xlabel("Epoch")
plt.legend()
plt.show()