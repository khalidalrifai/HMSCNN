import numpy as np
from scipy import fftpack
from scipy.signal import resample
from scipy.interpolate import interp1d


def apply_with_probability(prob=0.5):
    """Decorator to apply function with a given probability."""

    def decorator(func):
        def wrapper(self, seq):
            if np.random.rand() < prob:
                return func(self, seq)
            return seq

        return wrapper

    return decorator


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape:
    def __call__(self, seq):
        return seq.transpose()


class Retype:
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussianNoise:
    def __init__(self, sigma=0.01, prob=1.0):
        self.sigma = sigma
        self.prob = prob

    @apply_with_probability(prob=1.0)
    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class Scale:
    def __init__(self, sigma=0.01, prob=1.0):
        self.sigma = sigma
        self.prob = prob

    @apply_with_probability(prob=1.0)
    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq * scale_matrix


class Normalize:
    def __init__(self, type="0-1"):  # "0-1", "1-1", "mean-std"
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            return (seq - seq.min()) / (seq.max() - seq.min())
        elif self.type == "1-1":
            return 2 * (seq - seq.min()) / (seq.max() - seq.min()) - 1
        elif self.type == "mean-std":
            return (seq - seq.mean()) / seq.std()
        raise ValueError('This normalization is not included!')


# From here onwards, all classes will be assumed to have a probabilistic application due to their "Random" nature

class RandomTimeWarping:
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    @apply_with_probability()
    def __call__(self, seq):
        seq_len = seq.shape[1]
        warp_amt = np.random.uniform(-self.sigma, self.sigma)
        src_indices = np.arange(seq_len)
        dest_indices = np.arange(1 - warp_amt, seq_len + 1 - warp_amt)
        interpolator = interp1d(src_indices, seq, axis=1, fill_value="extrapolate")
        return interpolator(dest_indices)


class RandomJitter(AddGaussianNoise):
    """ Reusing AddGaussianNoise but with a different default sigma and probability."""

    def __init__(self, sigma=0.005):
        super().__init__(sigma=sigma, prob=0.5)


class RandomDropout:
    def __init__(self, dropout_prob=0.05):
        self.dropout_prob = dropout_prob

    @apply_with_probability()
    def __call__(self, seq):
        mask = np.random.choice([0, 1], size=seq.shape, p=[self.dropout_prob, 1 - self.dropout_prob])
        return seq * mask


class RandomStretch:
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    @apply_with_probability()
    def __call__(self, seq):
        seq_aug = np.zeros(seq.shape)
        len = seq.shape[1]
        length = int(len * (1 + (np.random.rand() - 0.5) * self.sigma))
        for i in range(seq.shape[0]):
            y = resample(seq[i, :], length)
            if length < len:
                if np.random.rand() < 0.5:
                    seq_aug[i, :length] = y
                else:
                    seq_aug[i, len - length:] = y
            else:
                if np.random.rand() < 0.5:
                    seq_aug[i, :] = y[:len]
                else:
                    seq_aug[i, :] = y[length - len:]
        return seq_aug


class RandomCrop:
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    @apply_with_probability()
    def __call__(self, seq):
        max_index = seq.shape[1] - self.crop_len
        random_index = np.random.randint(max_index)
        seq[:, random_index:random_index + self.crop_len] = 0
        return seq


class RandomFlip:
    @apply_with_probability()
    def __call__(self, seq):
        return -seq  # Multiply by -1


class RandomFill:
    def __init__(self, fill_len=20, fill_type="mean"):
        self.fill_len = fill_len
        self.fill_type = fill_type

    @apply_with_probability()
    def __call__(self, seq):
        max_index = seq.shape[1] - self.fill_len
        random_index = np.random.randint(max_index)
        if self.fill_type == "mean":
            fill_value = np.mean(seq)
        elif self.fill_type == "zero":
            fill_value = 0
        else:
            fill_value = np.median(seq)
        seq[:, random_index:random_index + self.fill_len] = fill_value
        return seq


class RandomShift:
    def __init__(self, shift_range=10):
        self.shift_range = shift_range

    @apply_with_probability()
    def __call__(self, seq):
        shift_value = np.random.randint(-self.shift_range, self.shift_range)
        return np.roll(seq, shift=shift_value)


class RandomFrequencyNoise:
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    @apply_with_probability()
    def __call__(self, seq):
        fourier = fftpack.fft(seq)
        power = np.abs(fourier)
        phase = np.angle(fourier)
        power += np.random.normal(loc=0, scale=self.sigma, size=power.shape)
        new_fourier = power * np.exp(1j * phase)
        return fftpack.ifft(new_fourier).real
