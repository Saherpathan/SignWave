import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')


# Load your model and data
model_dict = pickle.load(open('C:\\Users\Saher\PycharmProjects\Kiosk_Sign\model.p', 'rb'))
model = model_dict['model']

data_dict = pickle.load(open('C:\\Users\Saher\PycharmProjects\Kiosk_Sign\data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Get predictions
y_predict = model.predict(x_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_predict)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()
