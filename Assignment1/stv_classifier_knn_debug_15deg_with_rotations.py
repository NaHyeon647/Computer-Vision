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


def debug_direction_details(mag):
    """Reproduce get_features()'s exact 15-degree direction analysis."""
    h, w = mag.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    v, u = np.meshgrid(fy, fx, indexing="ij")
    radius = np.hypot(u, v)
    angle = np.mod(np.degrees(np.arctan2(v, u)), 180.0)
    valid = radius >= 0.03
    ids = np.clip(np.floor(angle[valid] / 15.0).astype(int), 0, 11)
    bins = np.bincount(ids, weights=(mag[valid] ** 2), minlength=12).astype(float)
    total = bins.sum()
    p = bins / total if total > 0 else np.zeros(12)
    smoothed = .25 * np.roll(p, 1) + .50 * p + .25 * np.roll(p, -1)
    first = int(np.argmax(smoothed))
    dist = np.abs(np.arange(12) - first)
    dist = np.minimum(dist, 12 - dist)
    candidates = np.where(dist >= 2)[0]
    second = int(candidates[np.argmax(smoothed[candidates])])
    ratio = smoothed[second] / smoothed[first] if smoothed[first] > 0 else 0
    raw_diff = abs(first - second) * 15
    acute = min(raw_diff, 180 - raw_diff) if ratio >= .25 else 0.
    return p, smoothed, first, second, ratio, acute


def save_debug_plot(path, img, mag, p, smoothed, first, second):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Test image")
    axs[0].axis("off")
    axs[1].imshow(np.log1p(mag), cmap="magma")
    axs[1].set_title("Log Fourier magnitude")
    axs[1].axis("off")
    degrees = np.arange(12) * 15
    axs[2].bar(degrees, p * 100, width=13, alpha=.50, label="Raw 15-degree energy")
    axs[2].plot(degrees, smoothed * 100, "o-", label="Smoothed")
    axs[2].axvline(first * 15, color="green", ls="--", label="Peak 1")
    axs[2].axvline(second * 15, color="red", ls="--", label="Peak 2")
    axs[2].set_xticks(degrees)
    axs[2].set_xlabel("Direction bin start (degrees)")
    axs[2].set_ylabel("Energy (%)")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

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
    if len(sys.argv) < 2 or any(arg not in ('--plot', '--debug') for arg in sys.argv[2:]):
        print(f"Usage: python {sys.argv[0]} <test.csv> [--plot] [--debug]")
        sys.exit(1)
    debug = '--debug' in sys.argv[2:]

    test_csv = sys.argv[1]

    train_img_paths, train_labels = read_train_data_csv('train.csv')
    test_img_paths = read_test_data_csv(test_csv)

    # Include the six rotated GIFs in the test predictions automatically.
    # Paths are assumed to be in the same test_data folder as the other GIFs.
    from pathlib import Path
    rotated_names = {
        'V_2_rotate', 'S_3_rotate', 'V_5_rotate',
        'T_5_rotate', 'V_4_rotate', 'S_2_rotate'
    }
    test_folder = Path('test_data')
    existing = {Path(p).as_posix().casefold() for p in test_img_paths}
    added = []
    if test_folder.is_dir():
        for candidate in sorted(test_folder.iterdir()):
            if (candidate.is_file()
                    and candidate.suffix.casefold() == '.gif'
                    and candidate.stem.casefold() in
                    {name.casefold() for name in rotated_names}):
                path = candidate.as_posix()
                if path.casefold() not in existing:
                    test_img_paths.append(path)
                    existing.add(path.casefold())
                    added.append(path)
    missing = rotated_names - {Path(p).stem for p in added} - {
        Path(p).stem for p in test_img_paths if Path(p).stem in rotated_names
    }
    if added:
        print(f"Added {len(added)} rotated test images:")
        for path in added:
            print(f"  {path}")
    if missing:
        print("Rotated GIFs not found in test_data (skipped):",
              ', '.join(sorted(missing)))

    train_imgs = load_images(train_img_paths)
    test_imgs = load_images(test_img_paths)

    mags, phases = get_fft_dataset(train_imgs)
    test_mags, test_phases = get_fft_dataset(test_imgs)

    # Plotting is optional so that it does not interrupt automated marking.
    if '--plot' in sys.argv[2:]:
        plot_fft(mags[0], phases[0])

    train_features = get_features(mags, phases)
    test_features = get_features(test_mags, test_phases)

    # Train one KNN on the two Fourier-derived features.
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    if debug:
        from pathlib import Path
        debug_folder = Path('fft_debug_plots')
        debug_folder.mkdir(exist_ok=True)
        debug_csv = Path(test_csv).stem + '_debug.csv'
        distances, neighbor_indices = classifier.kneighbors(test_features, n_neighbors=3)

        with open(debug_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'test_image', 'prediction', 'entropy', 'angle_degrees',
                'peak1_bin_degrees', 'peak2_bin_degrees', 'peak2_peak1_ratio',
                'neighbor1_file', 'neighbor1_label', 'neighbor1_distance',
                'neighbor2_file', 'neighbor2_label', 'neighbor2_distance',
                'neighbor3_file', 'neighbor3_label', 'neighbor3_distance',
                'energy_bins_percent_0_to_165'
            ])
            print("\n=== FFT / KNN DEBUG ===")
            for i, (path, label) in enumerate(zip(test_img_paths, predictions)):
                p, smooth, first, second, ratio, acute = debug_direction_details(test_mags[i])
                near = [(train_img_paths[j], train_labels[j], float(d))
                        for j, d in zip(neighbor_indices[i], distances[i])]
                print(f"\n[{i+1}/{len(test_imgs)}] {path} -> predicted {label}")
                print(f"  Entropy={test_features[i,0]:.4f}, angle={test_features[i,1]*90:.1f} deg")
                print(f"  Peaks: {first*15} deg, {second*15} deg; "
                      f"2nd/1st={ratio:.3f}; selected angle={acute:.1f} deg")
                print("  Raw 15-degree bin energy (%):",
                      np.round(100*p, 1).tolist())
                for rank, (npth, nlab, dst) in enumerate(near, 1):
                    print(f"  Neighbor {rank}: {nlab}, distance={dst:.4f}, {npth}")
                writer.writerow([
                    path, label, round(float(test_features[i, 0]), 6),
                    round(float(test_features[i, 1] * 90), 3),
                    first * 15, second * 15, round(ratio, 5),
                    *[value for entry in near for value in entry],
                    ' '.join(f'{100*x:.2f}' for x in p)
                ])
                save_debug_plot(
                    debug_folder / f'test_{i+1:03d}.png',
                    test_imgs[i], test_mags[i], p, smooth, first, second
                )
        print(f"\nDebug CSV: {debug_csv}")
        print(f"Debug plots: {debug_folder.resolve()}")
        print("These logs show predictions, not correctness: supply true test labels separately.")

    predictions_file = test_csv[:-4] + '_predictions.csv' if test_csv.lower().endswith('.csv') else test_csv + '_predictions.csv'
    with open(predictions_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image_path', 'prediction'])
        for i, pred in enumerate(predictions):
            writer.writerow([test_img_paths[i], pred])


if __name__ == "__main__":
    main()
