import numpy as np
import cv2
import heapq
from collections import Counter


NAME = "Huffman Compression"

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


class Node:

    def __init__(
        self,
        symbol=None,
        frequency=0,
        left=None,
        right=None
    ):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other):

        return self.frequency < other.frequency


def build_tree(image):

    pixels = image.flatten()

    frequency = Counter(
        pixels.tolist()
    )

    heap = []

    for symbol, freq in frequency.items():

        node = Node(
            symbol=symbol,
            frequency=freq
        )

        heapq.heappush(
            heap,
            node
        )

    if len(heap) == 1:

        return heap[0]

    while len(heap) > 1:

        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        parent = Node(
            frequency=
            left.frequency +
            right.frequency,
            left=left,
            right=right
        )

        heapq.heappush(
            heap,
            parent
        )

    return heap[0]


def generate_codes(
    node,
    codes,
    prefix=""
):

    if node is None:
        return

    if (
        node.left is None
        and node.right is None
    ):

        codes[node.symbol] = (
            prefix if prefix else "0"
        )

        return

    generate_codes(
        node.left,
        codes,
        prefix + "0"
    )

    generate_codes(
        node.right,
        codes,
        prefix + "1"
    )


def encode_image(image):

    tree = build_tree(image)

    codes = {}

    generate_codes(
        tree,
        codes
    )

    pixels = image.flatten()

    encoded_bits = sum(
        len(codes[value])
        for value in pixels
    )

    original_bits = (
        image.size * 8
    )

    return (
        codes,
        original_bits,
        encoded_bits
    )


def apply(image, params):

    mode = params["channel_mode"]

    if mode == "Grayscale":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        codes, original_bits, encoded_bits = (
            encode_image(gray)
        )

        compression_ratio = (
            original_bits / encoded_bits
            if encoded_bits > 0
            else 1
        )

        return {
            "image": gray,
            "original_bits": original_bits,
            "compressed_bits": encoded_bits,
            "compression_ratio": compression_ratio,
            "unique_values": len(codes)
        }

    else:

        total_original = 0
        total_encoded = 0
        total_unique = 0

        for channel in range(3):

            current = image[:, :, channel]

            codes, original_bits, encoded_bits = (
                encode_image(current)
            )

            total_original += original_bits
            total_encoded += encoded_bits
            total_unique += len(codes)

        compression_ratio = (
            total_original / total_encoded
            if total_encoded > 0
            else 1
        )

        return {
            "image": image,
            "original_bits": total_original,
            "compressed_bits": total_encoded,
            "compression_ratio": compression_ratio,
            "unique_values": total_unique
        }