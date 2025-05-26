import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from models import load_model  # Ensure your model loading function is available
from utils import load_data  # Ensure your dataset helper functions are available


def test():
    # ✅ Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Define image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # ✅ Load dataset first
    full_train_dataset = load_data(
        os.path.join("/content/fathom-net-kaggle-comp/dataset/", "train"),
        "annotations.csv",
        batch_size=16,
        use_roi=True,
    )

    # ✅ Pass the same taxonomy tree to the dataset to avoid duplicate calls
    taxonomy_tree = full_train_dataset.dataset.taxonomy_tree
    label_mapping = full_train_dataset.dataset.label_mapping

    # ✅ Load trained model
    # taxonomy_tree = {}  # Load your taxonomy tree if needed
    # label_mapping = {}  # Load label mapping for decoding predictions
    model = load_model(
        model_name="classifier", with_weights=True, taxonomy_tree=taxonomy_tree
    ).to(device)
    model.eval()

    # ✅ Load test annotations
    test_images_dir = "/content/fathom-net-kaggle-comp/dataset/test/rois/"
    annotations_file = "/content/fathom-net-kaggle-comp/dataset/test/annotations.csv"
    test_annotations = pd.read_csv(annotations_file)

    # ✅ Overwrite the existing "label" column with predictions
    for idx, image_name in enumerate(test_annotations["path"]):
        image_path = os.path.join(
            test_images_dir, os.path.basename(image_name)
        )  # Use os.path.basename to get the filename
        print("image_path:", image_path)
        break
        if not os.path.exists(image_path):
            print(f"[WARNING] Missing image: {image_path}")
            continue

        # ✅ Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # ✅ Run inference
        with torch.inference_mode():
            outputs = model(image)

        # ✅ Extract predicted class index (species-level used for labeling)
        pred_class_idx = outputs["species"].argmax(dim=1).cpu().item()

        # ✅ Convert predicted index to taxonomic name
        predicted_label = next(
            (
                name
                for name, idx in label_mapping["species"].items()
                if idx == pred_class_idx
            ),
            "UNKNOWN",
        )

        # ✅ Overwrite existing label with predicted name
        test_annotations.at[idx, "label"] = predicted_label

    # ✅ Save updated annotations file
    test_annotations.to_csv(annotations_file, index=False)
    print(f"[INFO] Updated annotations saved to {annotations_file}")
