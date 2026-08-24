import streamlit as st
import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Digital Image Processing",
    layout="wide"
)


# ============================================================
# SIMPLE CSS
# ============================================================

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


# ============================================================
# TITLE
# ============================================================

st.title("Digital Image Processing")

st.write(
    "Upload an image and select an operation."
)


# ============================================================
# LOAD OPERATIONS
# ============================================================

OPERATIONS_DIR = Path(__file__).parent / "operations"


def load_operations():

    operations = []
    errors = []

    for file in OPERATIONS_DIR.glob("*.py"):

        if file.name == "__init__.py":
            continue

        try:

            spec = importlib.util.spec_from_file_location(
                file.stem,
                file
            )

            module = importlib.util.module_from_spec(
                spec
            )

            spec.loader.exec_module(module)

            if not hasattr(module, "NAME"):
                raise ValueError(
                    "NAME is missing"
                )

            if not hasattr(module, "apply"):
                raise ValueError(
                    "apply() function is missing"
                )

            operations.append(module)

        except Exception as e:

            errors.append(
                f"{file.name}: {e}"
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
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff"
        ]
    )

    if operations:

        operation_names = [
            operation.NAME
            for operation in operations
        ]

        selected_name = st.selectbox(
            "Operation",
            operation_names
        )

        selected_operation = next(
            operation
            for operation in operations
            if operation.NAME == selected_name
        )

    else:

        st.error(
            "No operations found."
        )

        selected_operation = None


# ============================================================
# ERRORS
# ============================================================

if errors:

    with st.expander(
        "Some operations could not be loaded"
    ):

        for error in errors:

            st.write(error)


# ============================================================
# IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an image from the sidebar to begin."
    )

    st.stop()


image = Image.open(
    uploaded_file
).convert("RGB")

image = np.array(image)


# ============================================================
# PARAMETERS
# ============================================================

params = {}


if selected_operation is not None:

    st.subheader(
        selected_operation.NAME
    )

    parameter_definitions = getattr(
        selected_operation,
        "PARAMS",
        {}
    )

    for parameter_name, definition in parameter_definitions.items():

        parameter_type = definition["type"]

        # --------------------------------
        # SELECT
        # --------------------------------

        if parameter_type == "select":

            params[parameter_name] = st.selectbox(
                definition["label"],
                definition["options"],
                index=definition["options"].index(
                    definition.get(
                        "default",
                        definition["options"][0]
                    )
                )
            )

        # --------------------------------
        # SLIDER
        # --------------------------------

        elif parameter_type == "slider":

            params[parameter_name] = st.slider(
                definition["label"],
                min_value=definition["min"],
                max_value=definition["max"],
                value=definition.get(
                    "default",
                    definition["min"]
                ),
                step=definition["step"]
            )

        # --------------------------------
        # CHECKBOX
        # --------------------------------

        elif parameter_type == "checkbox":

            params[parameter_name] = st.checkbox(
                definition["label"],
                value=definition.get(
                    "default",
                    False
                )
            )


# ============================================================
# APPLY
# ============================================================

if st.button(
    "Apply Operation",
    type="primary"
):

    try:

        result = selected_operation.apply(
            image,
            params
        )

        st.session_state["result"] = result

    except Exception as e:

        st.error(
            f"Error while processing image: {e}"
        )


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.subheader("Original Image")

st.image(
    image,
    use_container_width=True
)


# ============================================================
# RESULT
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.subheader("Result")


    # ========================================
    # MULTIPLE RESULTS
    # ========================================

    if isinstance(result, list):

        columns = st.columns(4)

        for i, plane in enumerate(result):

            with columns[i % 4]:

                st.image(
                    plane,
                    caption=f"Bit Plane {i}",
                    use_container_width=True
                )

    # ========================================
    # SINGLE RESULT
    # ========================================

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original",
                use_container_width=True
            )

        with col2:

            st.image(
                result,
                caption="Processed",
                use_container_width=True
            )


# ============================================================
# HISTOGRAM FUNCTION
# ============================================================

def plot_histogram(img, title):

    fig, ax = plt.subplots(
        figsize=(7, 3)
    )

    if len(img.shape) == 2:

        ax.hist(
            img.ravel(),
            bins=256,
            range=(0, 256)
        )

    else:

        channels = [
            ("Red", 0),
            ("Green", 1),
            ("Blue", 2)
        ]

        for _, channel in channels:

            ax.hist(
                img[:, :, channel].ravel(),
                bins=256,
                range=(0, 256),
                alpha=0.5,
                label=_
            )

        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# HISTOGRAMS
# ============================================================

st.subheader("Histograms")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:

    plot_histogram(
        image,
        "Original Histogram"
    )

with hist_col2:

    if "result" in st.session_state:

        result = st.session_state["result"]

        if not isinstance(result, list):

            plot_histogram(
                result,
                "Processed Histogram"
            )


# ============================================================
# DOWNLOAD
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    if not isinstance(result, list):

        if len(result.shape) == 2:

            output_image = Image.fromarray(
                result
            )

        else:

            output_image = Image.fromarray(
                result
            )

        import io

        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            "Download Result",
            data=buffer.getvalue(),
            file_name="processed_image.png",
            mime="image/png"
        )