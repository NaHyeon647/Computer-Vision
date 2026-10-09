import sys
import csv
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# You may import a non-deep-learning classifier here, for example:
from sklearn.neighbors import KNeighborsClassifier

def read_train_data_csv(filename):
    img_paths = []
    labels = []
    with open(filename, 'r', newline='', encoding='utf-8') as train_csv:
        reader = csv.reader(train_csv)
        next(reader, None)  # Skip the CSV header.
        for row in reader:
            img_paths.append(row[0])
            labels.append(row[1])
    return img_paths, labels


def read_test_data_csv(filename):
    img_paths = []
    with open(filename, 'r', newline='', encoding='utf-8') as test_csv:
        reader = csv.reader(test_csv)
        next(reader, None)  # Skip the CSV header.
        for row in reader:
            img_paths.append(row[0])
    return img_paths


def load_images(img_paths):
    imgs = []
    for path in img_paths:
        img = io.imread(path, as_gray=True)
        img = np.squeeze(np.asarray(img, dtype=float))

        # Some GIF readers retain a frame dimension. Use the first frame.
        if img.ndim == 3:
            img = img[0]
        if img.ndim != 2:
            raise ValueError(f"Expected a 2D image after conversion to grayscale: {path}")

        imgs.append(img)
    return imgs


def get_fft(img):
    z = np.fft.fft2(img)  # Calculate the FFT of the image.
    q = np.fft.fftshift(z)  # Shift the zero-frequency component to the centre.
    mag = np.absolute(q)  # Magnitude spectrum.
    phase = np.angle(q)  # Phase spectrum.
    return mag, phase


def box_feature(mag, start_x=0, start_y=0, end_x=400, end_y=640):
    """Return the total spectral energy in a rectangular Fourier region."""
    region = mag[start_x:end_x, start_y:end_y]
    return np.sum(region ** 2)


def get_fft_dataset(ims):
    mags = []
    phases = []
    for im in ims:
        mag, phase = get_fft(im)
        mags.append(mag)
        phases.append(phase)
    return mags, phases


def get_features(mags, phases):
    """Return [directional entropy, angle between the two strongest directions]."""
    features = np.zeros((len(mags), 2), dtype=float)
    num_bins = 12  # 180 degrees / 15 degrees

    for i, mag in enumerate(mags):
        h, w = mag.shape

        # Actual frequency coordinates, accounting for image dimensions.
        fy = np.fft.fftshift(np.fft.fftfreq(h))
        fx = np.fft.fftshift(np.fft.fftfreq(w))
        v, u = np.meshgrid(fy, fx, indexing="ij")

        radius = np.hypot(u, v)
        angle = np.mod(np.degrees(np.arctan2(v, u)), 180.0)
        energy = mag ** 2

        # Exclude the dominant DC / very-low-frequency region.
        valid = radius >= 0.03

        # Sum energy in each of the 12 directional bins.
        bin_ids = np.floor(angle[valid] / 15.0).astype(int)
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)
        bins = np.bincount(
            bin_ids, weights=energy[valid], minlength=num_bins
        ).astype(float)

        total = bins.sum()
        if total <= 0:
            continue

        p = bins / total

        # Normalized entropy (0 to 1).
        nonzero = p > 0
        entropy = -np.sum(p[nonzero] * np.log(p[nonzero])) / np.log(num_bins)

        # Smooth neighboring bins to reduce boundary effects.
        smoothed = (
            0.25 * np.roll(p, 1)
            + 0.50 * p
            + 0.25 * np.roll(p, -1)
        )

        first = int(np.argmax(smoothed))

        # Find an independent second peak at least 30 degrees away.
        distances = np.abs(np.arange(num_bins) - first)
        distances = np.minimum(distances, num_bins - distances)
        candidates = np.where(distances >= 2)[0]

        second = int(candidates[np.argmax(smoothed[candidates])])
        if smoothed[second] < 0.25 * smoothed[first]:
            angle_feature = 0.0  # No sufficiently strong second direction.
        else:
            separation = min(abs(first - second) * 15.0,
                             180.0 - abs(first - second) * 15.0)
            angle_feature = separation / 90.0

        features[i] = [entropy, angle_feature]

    return features

def plot_fft(mag, phase):
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(np.abs(mag) + 1), cmap='gray')
    plt.title('Magnitude')
    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='gray')
    plt.title('Phase')
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != '--plot'):
        print(f"Usage: python {sys.argv[0]} <test.csv> [--plot]")
        sys.exit(1)

    test_csv = sys.argv[1]

    train_img_paths, train_labels = read_train_data_csv('train.csv')
    test_img_paths = read_test_data_csv(test_csv)

    train_imgs = load_images(train_img_paths)
    test_imgs = load_images(test_img_paths)

    mags, phases = get_fft_dataset(train_imgs)
    test_mags, test_phases = get_fft_dataset(test_imgs)

    # Plotting is optional so that it does not interrupt automated marking.
    if len(sys.argv) == 3:
        plot_fft(mags[0], phases[0])

    train_features = get_features(mags, phases)
    test_features = get_features(test_mags, test_phases)

    # Train one KNN on the two Fourier-derived features.
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    predictions_file = test_csv[:-4] + '_predictions.csv' if test_csv.lower().endswith('.csv') else test_csv + '_predictions.csv'
    with open(predictions_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image_path', 'prediction'])
        for i, pred in enumerate(predictions):
            writer.writerow([test_img_paths[i], pred])


if __name__ == "__main__":
    main()
