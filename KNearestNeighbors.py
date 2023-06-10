# Step 1: Start

# Step 2: Load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 3: Import 'diabetes.csv' file
# Step 4: Read csv file
diabetes = pd.read_csv('diabetes.csv')

# Step 5: Print columns of diabetes file
print(diabetes.columns)

# Step 6: Create training and test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], 
                                                    diabetes['Outcome'], 
                                                    stratify=diabetes['Outcome'], 
                                                    random_state=66)

# Step 7: Try N_neighbors from 1 to 10
k_values = range(1, 11)
training_accuracy = []
test_accuracy = []

# Step 8: Build the model
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Step 9: Record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    
    # Step 10: Record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))


# Step 11: Plot training accuracy and test accuracy
plt.plot(k_values, training_accuracy, label='Training Accuracy')
plt.plot(k_values, test_accuracy, label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 12: Print training accuracy and test accuracy
for k, train_acc, test_acc in zip(k_values, training_accuracy, test_accuracy):
    print(f'K = {k}: Training Accuracy = {train_acc:.2f}, Test Accuracy = {test_acc:.2f}')

# Step 13: Stop
