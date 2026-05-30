# LeNet (Modernized) — MNIST Classifier + API

A stronger take on the classic **LeNet-5**: same conv → pool → FC backbone,
upgraded with **BatchNorm**, **ReLU**, wider feature maps, and **Dropout**.
Includes training that saves serving artifacts and a **FastAPI** inference service.

## Files
| File | Purpose |
|------|---------|
| `model.py` | Modernized LeNet architecture (configurable classes/channels) |
| `train.py` | Trains on MNIST, saves best weights + `metadata.json` to `artifacts/` |
| `app.py` | FastAPI service that loads artifacts and serves `/predict` |
| `requirements.txt` | Pinned dependencies |

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py
```
Produces:
```
artifacts/
├── lenet_mnist.pt    # best model weights
└── metadata.json     # classes, input size, normalization (used by the API)
```

## Serve
```bash
uvicorn app:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, device, best accuracy |
| `/predict` | POST | Upload an image, returns predicted digit + confidence |

Example:
```bash
curl -X POST -F "file=@digit.png" http://127.0.0.1:8000/predict
```
```json
{
  "prediction": "7",
  "confidence": 0.9985,
  "probabilities": { "0": 0.0001, "7": 0.9985, ... }
}
```

> **Security note:** `/predict` is unauthenticated. Add auth (e.g. an API key
> or gateway) before exposing it publicly.
