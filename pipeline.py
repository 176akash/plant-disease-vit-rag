
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import timm
import faiss

from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO_ID = "176santu/plant-disease-vit-38"
MODEL_FILENAME = "plant_disease_vit_best.pth"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "classes.json"
)

RAG_PATH = os.path.join(
    BASE_DIR,
    "rag_documents.json"
)

FAISS_PATH = os.path.join(
    BASE_DIR,
    "faiss",
    "plant_disease.index"
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "faiss",
    "embeddings.npy"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CLASSES
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:

    CLASSES = json.load(f)


# ============================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================

def download_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    return model_path


def create_model():

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=38
    )

    return model


def load_model():

    model_path = download_model()

    model = create_model()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    model.to(DEVICE)

    return model


MODEL = load_model()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# LOAD RAG DOCUMENTS
# ============================================================

with open(
    RAG_PATH,
    "r",
    encoding="utf-8"
) as f:

    RAG_DOCUMENTS = json.load(f)


# ============================================================
# LOAD FAISS
# ============================================================

FAISS_INDEX = faiss.read_index(
    FAISS_PATH
)


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    image,
    top_k=3
):

    if isinstance(
        image,
        str
    ):

        image = Image.open(
            image
        )

    image = image.convert(
        "RGB"
    )

    tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.no_grad():

        logits = MODEL(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    top_values, top_indices = torch.topk(
        probabilities,
        k=top_k
    )

    predictions = []

    for probability, index in zip(
        top_values,
        top_indices
    ):

        class_index = int(
            index.item()
        )

        predictions.append({
            "class_index": class_index,
            "class_name": CLASSES[class_index],
            "confidence": float(
                probability.item()
            )
        })

    return {
        "predicted_class": predictions[0]["class_name"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
        "tensor": tensor
    }


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_knowledge(
    query,
    top_k=5
):

    query_embedding = EMBEDDING_MODEL.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = FAISS_INDEX.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        document = RAG_DOCUMENTS[
            int(index)
        ]

        results.append({
            "score": float(score),
            "document": document
        })

    return results


# ============================================================
# XAI SALIENCY MAP
# ============================================================

def generate_saliency_map(
    image,
    class_index=None
):

    if isinstance(
        image,
        str
    ):

        image = Image.open(
            image
        )

    image = image.convert(
        "RGB"
    )

    tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    tensor.requires_grad_(True)

    MODEL.zero_grad()

    logits = MODEL(
        tensor
    )

    if class_index is None:

        class_index = int(
            logits.argmax(
                dim=1
            ).item()
        )

    score = logits[
        0,
        class_index
    ]

    score.backward()

    gradients = tensor.grad[
        0
    ]

    saliency = gradients.abs().max(
        dim=0
    )[0]

    saliency = saliency.detach().cpu().numpy()

    saliency -= saliency.min()

    max_value = saliency.max()

    if max_value > 0:

        saliency /= max_value

    return saliency


# ============================================================
# GEMINI + RAG
# ============================================================

def generate_gemini_explanation(
    disease,
    confidence,
    rag_results
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return {
            "available": False,
            "text": (
                "Gemini API key is not configured. "
                "RAG information is available below."
            )
        }

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        knowledge = []

        for i, result in enumerate(
            rag_results,
            start=1
        ):

            document = result[
                "document"
            ]

            knowledge.append(
                f"""
SOURCE {i}
Title: {document.get('title', '')}
Source: {document.get('source', '')}
Knowledge:
{document.get('content', '')}
"""
            )

        knowledge_text = "\n".join(
            knowledge
        )

        prompt = f"""
You are an agricultural plant disease assistant.

A Vision Transformer classified a plant leaf image.

Predicted disease:
{disease}

Model confidence:
{confidence * 100:.2f}%

Use ONLY the retrieved knowledge below as the factual source
for the disease explanation.

Retrieved knowledge:
{knowledge_text}

Provide a concise explanation with exactly these sections:

Disease:
Confidence:
Symptoms:
Cause:
Management:
Prevention:
Sources:

Do not invent treatments, chemicals, dosages, or unsupported facts.
Mention that the result is an AI prediction and not a definitive
professional diagnosis.
"""

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return {
            "available": True,
            "text": interaction.output_text
        }

    except Exception as error:

        return {
            "available": False,
            "text": (
                "Gemini request failed: "
                + str(error)
            )
        }


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_pipeline(
    image,
    use_gemini=True
):

    prediction = predict_image(
        image,
        top_k=3
    )

    disease = prediction[
        "predicted_class"
    ]

    confidence = prediction[
        "confidence"
    ]

    class_index = prediction[
        "top_predictions"
    ][0][
        "class_index"
    ]

    saliency = generate_saliency_map(
        image,
        class_index
    )

    rag_results = retrieve_knowledge(
        disease,
        top_k=5
    )

    gemini_result = None

    if use_gemini:

        gemini_result = (
            generate_gemini_explanation(
                disease,
                confidence,
                rag_results
            )
        )

    return {
        "prediction": prediction,
        "saliency": saliency,
        "rag": rag_results,
        "gemini": gemini_result
    }


# ============================================================
# PIPELINE STATUS
# ============================================================

def pipeline_status():

    return {
        "model": "VisionTransformer",
        "classes": len(CLASSES),
        "rag_documents": len(
            RAG_DOCUMENTS
        ),
        "faiss_vectors": FAISS_INDEX.ntotal,
        "device": str(DEVICE),
        "gemini": bool(
            os.getenv(
                "GEMINI_API_KEY"
            )
        )
    }
