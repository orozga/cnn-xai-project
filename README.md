# CNN XAI Project

Projekt poświęcony wyjaśnialnej sztucznej inteligencji (**XAI**) dla konwolucyjnych sieci neuronowych (**CNN**) na przykładzie architektury **ResNet-50**.

Szczegółowa dokumentacja założeń badawczych i harmonogramu znajduje się w:
[docs/FAZA_1_CEL_I_ZAKRES.md](docs/FAZA_1_CEL_I_ZAKRES.md)

---

## Cel i zakres projektu (Tydzień 1)

- **Problem badawczy:** Ilościowa weryfikacja wierności wyjaśnień decyzji sieci CNN oraz identyfikacja zjawiska *shortcut learning* (wyjaśnienia wskazujące tło zamiast obiektu).
- **Model bazowy:** Pre-trenowana sieć **ResNet-50** z mechanizmem *forward/backward hooks* na warstwie `layer4` ($7 \times 7 \times 2048$).
- **Zbiór danych:** Referencyjny podzbiór klas ImageNet (`data/sample_images/`).
- **Badane metody XAI:** Metody gradientowe (*Vanilla Saliency*, *Grad-CAM*, *Grad-CAM++*), metoda bezgradientowa (*Score-CAM*) oraz autorskie usprawnienie.
- **Kryteria ewaluacji:** Ilościowe metryki wierności i lokalizacji zaimplementowane w `src/metrics.py` (*Pointing Game*, *Deletion AUC*, *Insertion AUC*).

---

## Uruchomienie

1. Instalacja zależności:
```bash
python -m pip install -r requirements.txt
```

2. Pobranie obrazów testowych:
```bash
python -m src.download_data
```

3. Weryfikacja modelu ResNet-50 i rejestracji hooków:
```bash
python -m src.model
```

4. Generowanie map atrybucji (Saliency Maps):
```bash
python main.py
```

5. Weryfikacja modułu metryk ilościowych:
```bash
python -m src.metrics
```

Wygenerowane wizualizacje zapisywane są w katalogu `results/saliency/`.
