import matplotlib.pyplot as plt

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()

# Set seed for reproducibility
np.random.seed(0)


def generate_data(m):
    # Generate input and target data
    factor = 5
    X = np.random.rand(m, 3) * factor
    Y = f(X)

    return X, Y


# We will use the following function as a running example in the demo scripts
def f(X):
    # For a training example - y = x1 + 2 * x2^2 + 3 * x3^.5, the fancy indexing preserves 2D shape
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y


def generate_moons_data(num_samples, num_classes):

    num_samples_per_class = num_samples // num_classes

    # Generate random angles for each moon class
    random_angles = np.random.rand(num_samples_per_class, num_classes) * np.pi

    # Define radius for each moon class
    radius_step = 20
    radii = np.arange(1, num_classes + 1) * radius_step

    # Generate coordinates for each moon
    x = np.cos(random_angles) * radii
    y = np.sin(random_angles) * radii
    
    # Add noise
    noise = 0.1
    x += np.random.normal(scale = noise, size = x.shape)
    y += np.random.normal(scale = noise, size = y.shape)

    # Combine coordinates
    X = np.vstack([x.reshape(1, -1), y.reshape(1, -1)]).T
    # Assign labels to moons
    Y = np.tile(np.arange(num_classes), num_samples_per_class)
    if num_classes > 2:
        # Convert labels to one-hot encoding if multi-class classification
        Y = to_one_hot(Y, num_classes = num_classes)
    elif num_classes == 2:
        Y = Y.reshape(1, -1).T

    return X, Y


def to_one_hot(labels, num_classes):
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_samples, num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1.0
    
    return one_hot_labels


def plot_moons_data(X, Y):
    # Get number of classes
    num_classes = 2 if Y.shape[1] == 1 else Y.shape[1]

    if num_classes == 2:
        Y = to_one_hot(Y, num_classes = num_classes)

    # Plot the data
    plt.figure(figsize = (10, 6))
    for i in range(num_classes):
        plt.scatter(X[Y[:, i] == 1][:, 0], X[Y[:, i] == 1][:, 1], label = f'Class {i}')

    plt.title('Moons Dataset')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()