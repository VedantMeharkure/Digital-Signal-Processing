
import hashlib
import importlib.util
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Digital Image Processing",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Digital Image Processing")
st.write("Upload an image and select an operation.")


# ============================================================
# LOAD OPERATIONS
# ============================================================
OPERATIONS_DIR = Path(__file__).parent / "operations"


def load_operations():
    operations = []
    errors = []

    if not OPERATIONS_DIR.is_dir():
        return [], [f"Operations folder not found: {OPERATIONS_DIR}"]

    for file in sorted(OPERATIONS_DIR.glob("*.py")):
        if file.name == "__init__.py":
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                f"dip_operation_{file.stem}", file
            )

            if spec is None or spec.loader is None:
                raise ImportError("Could not load module")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "NAME"):
                raise ValueError("NAME is missing")

            if not callable(getattr(module, "apply", None)):
                raise ValueError("apply(image, params) is missing")

            operations.append(module)

        except Exception as exc:
            errors.append(
                f"{file.name}: {type(exc).__name__}: {exc}"
            )

    return operations, errors


operations, errors = load_operations()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Options")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

    selected_operation = None

    if operations:
        operation_names = [op.NAME for op in operations]

        selected_name = st.selectbox(
            "Operation",
            operation_names
        )

        selected_operation = next(
            op for op in operations if op.NAME == selected_name
        )
    else:
        st.error("No operations found. Check your operations folder.")

if errors:
    with st.expander("Some operations could not be loaded"):
        for error in errors:
            st.error(error)


# ============================================================
# LOAD IMAGE
# ============================================================
if uploaded_file is None:
    st.info("Upload an image from the sidebar to begin.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.getvalue()

    image = np.array(
        Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    )

except Exception as exc:
    st.error(f"Could not read image: {exc}")
    st.stop()

image_id = hashlib.sha256(uploaded_bytes).hexdigest()


# ============================================================
# OPERATION PARAMETERS
# ============================================================
params = {}

if selected_operation is not None:
    st.subheader(selected_operation.NAME)

    definitions = getattr(selected_operation, "PARAMS", {})

    for name, definition in definitions.items():
        parameter_type = definition.get("type")

        if parameter_type == "select":
            options = definition["options"]
            default = definition.get("default", options[0])

            params[name] = st.selectbox(
                definition.get("label", name),
                options,
                index=options.index(default)
                if default in options else 0,
                key=f"{selected_operation.NAME}_{name}"
            )

        elif parameter_type == "slider":
            params[name] = st.slider(
                definition.get("label", name),
                min_value=definition["min"],
                max_value=definition["max"],
                value=definition.get(
                    "default", definition["min"]
                ),
                step=definition.get("step", 1),
                key=f"{selected_operation.NAME}_{name}"
            )

        elif parameter_type == "checkbox":
            params[name] = st.checkbox(
                definition.get("label", name),
                value=definition.get("default", False),
                key=f"{selected_operation.NAME}_{name}"
            )


# ============================================================
# IMAGE CONVERSION AND VALIDATION
# ============================================================
def to_display_array(output):
    """Convert supported operation results into displayable arrays."""

    if isinstance(output, Image.Image):
        output = np.array(output)

    elif isinstance(output, (bytes, bytearray)):
        output = np.array(
            Image.open(io.BytesIO(output))
        )

    array = np.asarray(output)

    if array.size == 0:
        raise ValueError("The operation returned an empty image.")

    if array.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image dimensions: {array.shape}"
        )

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(
            array, nan=0.0, posinf=255.0, neginf=0.0
        )

        if array.min() >= 0 and array.max() <= 1:
            array = array * 255

        array = np.clip(array, 0, 255).astype(np.uint8)

    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] not in (3, 4):
            raise ValueError(
                f"Unsupported channel count: {array.shape}"
            )

    return array


def show_image(output, caption):
    array = to_display_array(output)

    st.image(
        array,
        caption=caption,
        use_container_width=True
    )


# ============================================================
# HISTOGRAM
# ============================================================
def plot_histogram(img, title):
    array = to_display_array(img)

    fig, ax = plt.subplots(figsize=(7, 3))

    if array.ndim == 2:
        ax.hist(
            array.ravel(),
            bins=256,
            range=(0, 256)
        )

    else:
        if array.shape[2] == 4:
            array = array[:, :, :3]

        channels = [
            ("Red", 0),
            ("Green", 1),
            ("Blue", 2)
        ]

        for channel_name, channel_index in channels:
            ax.hist(
                array[:, :, channel_index].ravel(),
                bins=256,
                range=(0, 256),
                alpha=0.5,
                label=channel_name
            )

        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ============================================================
# APPLY OPERATION
# ============================================================
if selected_operation is not None:
    if st.button("Apply Operation", type="primary"):
        try:
            with st.spinner("Processing image..."):
                result = selected_operation.apply(
                    image.copy(),
                    params.copy()
                )

            if result is None:
                raise ValueError(
                    "The operation returned no output."
                )

            # Validate results before storing them.
            if isinstance(result, dict):
                if "image" not in result:
                    raise ValueError(
                        "Dictionary result has no 'image' entry."
                    )

                to_display_array(result["image"])

            elif isinstance(result, list):
                if not result:
                    raise ValueError("The result list is empty.")

                for item in result:
                    to_display_array(item)

            else:
                to_display_array(result)

            st.session_state["dip_result"] = {
                "image_id": image_id,
                "operation": selected_operation.NAME,
                "params": params.copy(),
                "result": result
            }

            st.success("Operation completed successfully!")

        except Exception as exc:
            st.session_state.pop("dip_result", None)

            st.error(
                f"Error while processing image: "
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# DISPLAY ORIGINAL IMAGE
# ============================================================
st.subheader("Original Image")
show_image(image, "Original")


# ============================================================
# DISPLAY PROCESSED RESULT
# ============================================================
saved = st.session_state.get("dip_result")
current_result = None

if saved is not None:
    if saved["image_id"] != image_id:
        st.info("Click Apply Operation to process this image.")

    elif (
        selected_operation is not None
        and saved["operation"] != selected_operation.NAME
    ):
        st.info("Click Apply Operation to update the result.")

    elif saved["params"] != params:
        st.info("Parameters changed. Click Apply Operation again.")

    else:
        current_result = saved["result"]


if current_result is not None:
    st.subheader("Processed Result")

    # Compression operations return a dictionary.
    if isinstance(current_result, dict):
        processed_image = current_result["image"]

        col1, col2 = st.columns(2)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(
                processed_image,
                "Processed / Reconstructed"
            )

        # Display compression information when available.
        if all(
            key in current_result
            for key in (
                "original_bits",
                "compressed_bits",
                "compression_ratio"
            )
        ):
            st.subheader("Compression Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Original Bits",
                f"{current_result['original_bits']:,}"
            )

            col2.metric(
                "Compressed Bits",
                f"{current_result['compressed_bits']:,}"
            )

            col3.metric(
                "Compression Ratio",
                f"{current_result['compression_ratio']:.2f}"
            )

            if "unique_values" in current_result:
                st.write(
                    "Unique intensity values: "
                    f"{current_result['unique_values']}"
                )

    # Operations returning multiple images.
    elif isinstance(current_result, list):
        columns = st.columns(min(len(current_result), 3))

        for index, output in enumerate(current_result):
            with columns[index % len(columns)]:
                show_image(output, f"Result {index + 1}")

    # Normal image-processing operations.
    else:
        col1, col2 = st.columns(2)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(current_result, "Processed")


    # ========================================================
    # HISTOGRAMS
    # ========================================================
    st.subheader("Histograms")

    hist_col1, hist_col2 = st.columns(2)

    with hist_col1:
        plot_histogram(image, "Original Histogram")

    with hist_col2:
        if isinstance(current_result, dict):
            plot_histogram(
                current_result["image"],
                "Processed Histogram"
            )

        elif isinstance(current_result, list):
            st.caption(
                "Multiple outputs returned; "
                "processed histogram omitted."
            )

        else:
            plot_histogram(
                current_result,
                "Processed Histogram"
            )


    # ========================================================
    # DOWNLOAD RESULT
    # ========================================================
    if not isinstance(current_result, list):
        try:
            if isinstance(current_result, dict):
                output = current_result["image"]
            else:
                output = current_result

            output_array = to_display_array(output)

            output_image = Image.fromarray(output_array)

            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")

            st.download_button(
                label="Download Result",
                data=buffer.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )

        except Exception as exc:
            st.warning(
                f"Could not prepare the download: {exc}"
            )

else:
    st.info(
        "Choose an operation and parameters, "
        "then click Apply Operation to see the result."
    )
