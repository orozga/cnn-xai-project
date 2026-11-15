# CNN XAI Project

Projekt poświęcony wyjaśnialnej sztucznej inteligencji (**XAI**) dla konwolucyjnych sieci neuronowych (**CNN**) na przykładzie architektury **ResNet-50**.

---

## Zakres Fazy I i Fazy II (Tygodnie 1–7)

- **Problem badawczy:** Ilościowa weryfikacja wierności wyjaśnień decyzji sieci CNN oraz identyfikacja zjawiska *shortcut learning* (wyjaśnienia wskazujące tło zamiast obiektu).
- **Model bazowy:** Pre-trenowana sieć **ResNet-50** z mechanizmem *forward/backward hooks* (`src/model.py`, domyślnie na warstwie `layer4` o wymiarach $7 \times 7 \times 2048$, z możliwością dynamicznego wyboru warstwy).
- **Zbiór danych:** Zbalansowany (stratyfikowany) podzbiór 10 klas **Imagenette** (`data/sample_images/` wraz z `metadata.json`).
- **Zaimplementowane metody XAI (`src/xai_methods.py`):**
  - `saliency` — *Vanilla Saliency Maps*
  - `gradcam` — *Grad-CAM*
  - `gradcam_pp` — *Grad-CAM++*
  - `scorecam` — *Score-CAM* (bezgradientowa, przetwarzanie wsadowe)
- **Kryteria ewaluacji (`src/metrics.py`):** *Deletion AUC*, *Insertion AUC*, *Pointing Game Hit*, *Energy in Bounding Box*, *Gini Index*, *Znormalizowana Entropia Shannona* oraz *Kaskadowa Randomizacja Wag (Test Adebayo)*.

---

## Uruchomienie

1. Instalacja zależności:
```bash
python -m pip install -r requirements.txt
```

2. Pobranie zbalansowanej próbki obrazów testowych Imagenette:
```bash
python -m src.download_data
```

3. Weryfikacja modelu ResNet-50 i ekstrakcji aktywacji oraz gradientów przez hooki:
```bash
python -m src.model
```

4. Weryfikacja modułu metryk ilościowych:
```bash
python -m src.metrics
```

5. Uruchomienie ujednoliconego potoku XAI dla wszystkich 4 metod oraz wygenerowanie raportu porównawczego:
```bash
python main.py
```

Opcjonalne flagi CLI:
```bash
python main.py --methods saliency gradcam gradcam_pp scorecam --max-images 10 --scorecam-max-channels 256
```

Wygenerowane wizualizacje i metryki zapisywane są w katalogach `results/<metoda>/`, a zbiorczy raport porównawczy w `results/comparison/RAPORT_FAZA_2.md`.
