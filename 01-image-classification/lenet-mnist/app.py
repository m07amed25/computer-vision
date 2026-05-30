import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms

ARTIFACTS = Path(__file__).parent / "artifacts"
device = "cuda" if torch.cuda.is_available() else "cpu"
STATE: dict = {}


class LeNet(nn.Module):
    """Self-contained copy of the architecture trained in model.ipynb."""

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_channels, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        meta = json.loads((ARTIFACTS / "metadata.json").read_text())
        model = LeNet(num_classes=meta["num_classes"], in_channels=meta["in_channels"]).to(device)
        model.load_state_dict(torch.load(ARTIFACTS / meta["weights"], map_location=device))
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing artifact {e.filename!r}. Run model.ipynb to generate artifacts/.") from e
    except RuntimeError as e:
        raise RuntimeError(f"Could not load weights (architecture/metadata mismatch?): {e}") from e
    model.eval()
    STATE.update(meta=meta, model=model, preprocess=transforms.Compose([
        transforms.Grayscale(num_output_channels=meta["in_channels"]),
        transforms.Resize(tuple(meta["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(meta["normalize"]["mean"], meta["normalize"]["std"]),
    ]))
    yield
    STATE.clear()


app = FastAPI(title="LeNet Image Classifier", version="1.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Health(BaseModel):
    status: str
    device: str
    architecture: str
    accuracy: float | None


class Prediction(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health", response_model=Health)
def health():
    meta = STATE["meta"]
    return Health(status="ok", device=device, architecture=meta["architecture"],
                  accuracy=meta.get("best_accuracy"))


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file")
    x = STATE["preprocess"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(STATE["model"](x), dim=1)[0]
    conf, idx = probs.max(0)
    classes = STATE["meta"]["classes"]
    return Prediction(
        prediction=classes[idx],
        confidence=round(conf.item(), 4),
        probabilities={c: round(p, 4) for c, p in zip(classes, probs.tolist())},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000)
