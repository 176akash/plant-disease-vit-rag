
import os
import sys

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# ------------------------------------------------------------
# Project directory
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if BASE_DIR not in sys.path:
    sys.path.insert(
        0,
        BASE_DIR
    )

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ------------------------------------------------------------
# Gemini secret
# ------------------------------------------------------------

try:

    if "GEMINI_API_KEY" in st.secrets:

        os.environ["GEMINI_API_KEY"] = (
            st.secrets["GEMINI_API_KEY"]
        )

except Exception:
    pass

# ------------------------------------------------------------
# Load pipeline
# ------------------------------------------------------------

@st.cache_resource(
    show_spinner="Loading AI model and RAG system..."
)
def load_pipeline():

    import pipeline

    return pipeline


pipeline = load_pipeline()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title(
    "🌿 Plant Disease Detection"
)

st.markdown(
    """
### Vision Transformer + Explainable AI + RAG + Gemini

Upload a plant leaf image to get:

- 🔍 Disease prediction
- 📊 Confidence score
- 🥇 Top-3 predictions
- 🔥 XAI saliency map
- 📚 RAG-based disease information
- 🤖 Gemini-generated explanation
"""
)

st.divider()

# ------------------------------------------------------------
# Upload image
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

# ------------------------------------------------------------
# Main application
# ------------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        caption="Plant leaf",
        use_container_width=True
    )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing leaf..."
    ):

        prediction = pipeline.predict_image(
            image,
            top_k=3
        )

    disease = prediction[
        "predicted_class"
    ]

    confidence = prediction[
        "confidence"
    ]

    # --------------------------------------------------------
    # Prediction section
    # --------------------------------------------------------

    st.subheader(
        "🔍 Prediction"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Disease",
            disease.replace(
                "___",
                " — "
            )
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Top-3 predictions
    # --------------------------------------------------------

    st.subheader(
        "🥇 Top-3 Predictions"
    )

    for i, item in enumerate(
        prediction["top_predictions"],
        start=1
    ):

        st.write(
            f"**{i}. "
            f"{item['class_name']}** — "
            f"{item['confidence'] * 100:.2f}%"
        )

        st.progress(
            float(
                item["confidence"]
            )
        )

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    st.subheader(
        "🔥 Explainable AI"
    )

    with st.spinner(
        "Generating saliency map..."
    ):

        class_index = prediction[
            "top_predictions"
        ][0]["class_index"]

        saliency = (
            pipeline.generate_saliency_map(
                image,
                class_index
            )
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        fig, ax = plt.subplots()

        ax.imshow(
            saliency,
            cmap="jet"
        )

        ax.axis("off")

        ax.set_title(
            "XAI Saliency Map"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    st.info(
        "The saliency map highlights image regions "
        "that contributed strongly to the model's prediction."
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    st.subheader(
        "📚 Retrieved Knowledge"
    )

    with st.spinner(
        "Retrieving relevant disease information..."
    ):

        rag_results = (
            pipeline.retrieve_knowledge(
                disease,
                top_k=5
            )
        )

    for i, result in enumerate(
        rag_results,
        start=1
    ):

        document = result[
            "document"
        ]

        with st.expander(
            f"{i}. {document['title']} "
            f"(score: {result['score']:.4f})"
        ):

            st.write(
                document["content"]
            )

            st.caption(
                f"Source: "
                f"{document.get('source', 'Project Agricultural Knowledge Base')}"
            )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    st.subheader(
        "🤖 Gemini + RAG Explanation"
    )

    if os.environ.get(
        "GEMINI_API_KEY"
    ):

        with st.spinner(
            "Generating explanation..."
        ):

            gemini_result = (
                pipeline.generate_gemini_explanation(
                    disease,
                    confidence,
                    rag_results
                )
            )

        if gemini_result["available"]:

            st.markdown(
                gemini_result["text"]
            )

        else:

            st.warning(
                gemini_result["text"]
            )

    else:

        st.warning(
            "Gemini API key is not configured. "
            "RAG results are still available."
        )

    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "⚠️ This result is an AI prediction and "
        "not a definitive professional diagnosis."
    )

else:

    st.info(
        "👆 Upload a plant leaf image to begin."
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Plant Disease Detection — "
    "Vision Transformer + XAI + RAG + Gemini"
)
