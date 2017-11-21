import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics

# load dataset
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.title('Training: %i' % label)

n_samples = len(digits.images)
# reshape digits.images to n_samples lines and auto calculate column lines
data = digits.images.reshape((n_samples, -1))

# create SVM classifier
classifier = svm.SVC(gamma=0.001)

# the first half is used to train
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# and then predict the second half of the data
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
#
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.get_cmap('gray'))
#     plt.title('Prediction: %i' % prediction)

got = 0
for index in range(len(predicted)):
    if expected[index] != predicted[index]:
        plt.subplot(2, 4, got + 5)
        got += 1
        plt.axis('off')
        plt.imshow(digits.images[n_samples // 2:][index], cmap=plt.get_cmap('gray'))
        plt.title('%i -> %i' % (expected[index], predicted[index]))
        if got >= 4:
            break

plt.show()


