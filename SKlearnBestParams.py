import sklearn
import seaborn as sns
import random
import os
import numpy as np
import joblib
import torch
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
from PIL import Image


def load_images_from_folder(folder, valid_extensions=("png", "jpg", "jpeg")):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path)
                img = img.resize((100, 100))  # Resize images to a consistent size
                img_gray = img.convert("RGB")  # Convert to RGB
                images.append(np.array(img_gray).flatten())
                labels.append(
                    folder.split("/")[-1]
                )  # Assuming your folder structure is consistent
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images, labels


apple = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\apple"
avocado = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\avocado"
banana = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\banana"
cherry = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\cherry"
kiwi = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\kiwi"
mango = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\mango"
orange = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\orange"
pineapple = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\pinenapple"
stawberries = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\stawberries"
watermelon = r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\MY_data\test\watermelon"

apple_image, apple_labels = load_images_from_folder(apple)
avocado_image, avocado_labels = load_images_from_folder(avocado)
banana_image, banana_labels = load_images_from_folder(banana)
cherry_image, cherry_labels = load_images_from_folder(cherry)
kiwi_image, kiwi_labels = load_images_from_folder(kiwi)
mango_image, mango_labels = load_images_from_folder(mango)
orange_image, orange_labels = load_images_from_folder(orange)
pineapple_image, pineapple_labels = load_images_from_folder(pineapple)
stawberries_image, stawberries_labels = load_images_from_folder(stawberries)
watermelon_image, watermelon_labels = load_images_from_folder(watermelon)

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.1, 0.01, 0.001],
    "kernel": ["linear", "rbf"],
}

X = (
    apple_image
    + avocado_image
    + banana_image
    + cherry_image
    + kiwi_image
    + mango_image
    + orange_image
    + pineapple_image
    + stawberries_image
    + watermelon_image
)
y = (
    apple_labels
    + avocado_labels
    + banana_labels
    + cherry_labels
    + kiwi_labels
    + mango_labels
    + orange_labels
    + pineapple_labels
    + stawberries_labels
    + watermelon_labels
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("-----------Start Training-----------")
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=10, verbose=3)

fit = grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_svm_model = grid_search.best_estimator_
print("Best hyperparameters:", best_params)

y_pred = best_svm_model.predict(X_test)
joblib.dump(
    best_svm_model,
    r"C:\Users\me095\OneDrive\Desktop\one file\WorkOnly\All Code\Python\Spain\Best_params_model_ClassificationDogFeeling.joblib",
)
accuracy_score = accuracy_score(y_pred=y_pred, y_true=y_test)
try:
    conf_matrix = confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels={
            0: "apple",
            1: "avocado",
            2: "banana",
            3: "cherry",
            4: "kiwi",
            5: "mango",
            6: "orange",
            7: "pineapple",
            8: "stawberries",
            9: "watermelon",
        },
    )
except:
    pass

sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=[
        "apple",
        "avocado",
        "banana",
        "cherry",
        "kiwi",
        "mango",
        "orange",
        "pineapple",
        "stawberries",
        "watermelon",
    ],
    yticklabels=[
        "apple",
        "avocado",
        "banana",
        "cherry",
        "kiwi",
        "mango",
        "orange",
        "pineapple",
        "stawberries",
        "watermelon",
    ],
)

print(f"Accuracy Score : {accuracy_score}")
print(f"Confusion Metrix : \n{conf_matrix}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

index = random.randint(0, len(X_test) - 1)
predicted_label = os.path.basename(y_pred[index])
true_label = os.path.basename(y_test[index])

plt.figure()
plt.imshow(np.array(X_test[index]).reshape(100, 100, 3))
plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
plt.axis("off")
plt.show()
