import numpy as np
import cv2

NAME = "Shannon-Fano Compression"

PARAMS = {
    "channel_mode": {
        "type": "select",
        "label": "Image Mode",
        "options": [
            "Grayscale",
            "RGB"
        ],
        "default": "Grayscale"
    }
}


def shannon_fano(symbols, codes, prefix=""):
    if len(symbols) == 1:
        codes[symbols[0][0]] = prefix if prefix else "0"
        return

    total = sum(freq for _, freq in symbols)

    current = 0
    split = 0
    min_difference = float("inf")

    for i in range(len(symbols) - 1):
        current += symbols[i][1]

        difference = abs(
            current - (total - current)
        )

        if difference < min_difference:
            min_difference = difference
            split = i + 1

    left = symbols[:split]
    right = symbols[split:]

    shannon_fano(
        left,
        codes,
        prefix + "0"
    )

    shannon_fano(
        right,
        codes,
        prefix + "1"
    )


def create_codes(image):
    pixels = image.flatten()

    values, counts = np.unique(
        pixels,
        return_counts=True
    )

    symbols = list(
        zip(
            values.tolist(),
            counts.tolist()
        )
    )

    symbols.sort(
        key=lambda x: x[1],
        reverse=True
    )

    codes = {}

    shannon_fano(
        symbols,
        codes
    )

    return codes, symbols


def apply(image, params):

    mode = params["channel_mode"]

    if mode == "Grayscale":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        codes, symbols = create_codes(gray)

        original_bits = gray.size * 8

        compressed_bits = sum(
            frequency * len(codes[value])
            for value, frequency in symbols
        )

        compression_ratio = (
            original_bits / compressed_bits
            if compressed_bits > 0
            else 1
        )

        decoded = gray.copy()

        return {
            "image": decoded,
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": compression_ratio,
            "unique_values": len(symbols)
        }

    else:

        channels = []

        total_original = 0
        total_compressed = 0
        total_unique = 0

        for channel in range(3):

            current = image[:, :, channel]

            codes, symbols = create_codes(
                current
            )

            original_bits = current.size * 8

            compressed_bits = sum(
                frequency * len(codes[value])
                for value, frequency in symbols
            )

            channels.append(current)

            total_original += original_bits
            total_compressed += compressed_bits
            total_unique += len(symbols)

        compression_ratio = (
            total_original / total_compressed
            if total_compressed > 0
            else 1
        )

        decoded = np.stack(
            channels,
            axis=2
        )

        return {
            "image": decoded,
            "original_bits": total_original,
            "compressed_bits": total_compressed,
            "compression_ratio": compression_ratio,
            "unique_values": total_unique
        }