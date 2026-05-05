# HemoScan 
- Realizat de: Țarălungă Egor || Mironenco Alexandra

Un sistem de clasificare a celulelor din sange bazat pe imagini microscopice. Modelul analizeaza o poza cu un frotiu de sange si returneaza tipul celulei, bolile posibile asociate si ce actiune medicala este recomandata.

---

## Ce face

- Recunoaste tipuri de celule din sange (globule albe, globule rosii, paraziti)
- Arata cat de sigur este modelul pe raspuns
- Indica severitatea (normal, scazut, mediu, ridicat, critic)
- Sugereaza actiuni medicale pentru fiecare rezultat
- Are o interfata web unde poti incarca o imagine si vezi rezultatul instant
- Poate rula si din terminal pe un folder intreg de imagini

---

## Clase suportate

| Categorie | Clase |
|---|---|
| Globule albe / Leucemie | myeloblast, lymphoblast, neutrophil, eosinophil, basophil, monocyte, lymphocyte, reactive lymphocyte, normoblast |
| Anomalii globule rosii |
| Trombocite | platelet clump |
| Paraziti | malaria ring |
| Normal | normal |

---

## Structura proiectului

```
project/
  train.py              # Antreneaza modelul
  dataset.py            # Incarca si proceseaza imaginile
  predict.py            # Ruleaza modelul pe imagini noi
  disease_mapping.py    # Leaga fiecare clasa de boli si actiuni
  app.py                # Serverul web Flask
  static/
    index.html          # Interfata web
  data/
    blood_cells/        # Datele de antrenament (un folder per clasa)
  checkpoints/
    best_model.pt       # Modelul salvat
    classes.json        # Lista claselor
    confusion_matrix.png
```

---

## Instalare

**Cerinte**

```
Python 3.10+
torch
torchvision
flask
pillow
scikit-learn
matplotlib
numpy
```

## Pregatirea datelor

Pune imaginile in foldere separate, cate unul per clasa:

```
data/blood_cells/
  myeloblast/
  lymphoblast/
  neutrophil/
  normal/
  ...
```

---

## Antrenament

```bash
python train.py --data_dir data/blood_cells --epochs 30 --batch_size 32
```

Argumente principale:

| Argument | Valoare implicita | Descriere |
|---|---|---|
| `--data_dir` | `data/blood_cells` | Folderul cu datele de antrenament |
| `--epochs` | `30` | Numarul total de epoci |
| `--warmup_epochs` | `5` | Epoci cu backbone inghetat (Etapa 1) |
| `--batch_size` | `32` | Marimea unui batch |
| `--lr` | `1e-4` | Rata de invatare pentru Etapa 1 |
| `--checkpoint_dir` | `checkpoints` | Unde se salveaza modelul |

Dupa antrenament, se salveaza automat o matrice de confuzie in `checkpoints/confusion_matrix.png` ca sa poti vedea unde greseste modelul.

---

## Inferenta din terminal

**O singura imagine:**

```bash
python predict.py --image cale/catre/imagine.tif
```

**Un folder intreg:**

```bash
python predict.py --folder cale/catre/folder/ --output_json rezultate.json
```

| Argument | Valoare implicita | Descriere |
|---|---|---|
| `--checkpoint` | `checkpoints/best_model.pt` | Calea catre model |
| `--top_k` | `3` | Cate predictii sa afiseze |
| `--output_json` | None | Salveaza rezultatele ca JSON |

---

## Interfata web

Porneste serverul:

```bash
python app.py
```

Deschide `http://localhost:5000` in browser. Poti incarca o imagine JPG, PNG, TIF sau BMP si vei vedea rezultatul cu predictia principala, scorurile alternative si avertismentele pentru clasele ambigue.

Daca nu exista un checkpoint salvat, serverul ruleaza in mod demo cu rezultate simulate.

---

## Cum functioneaza clasificarea bolilor

Fisierul `disease_mapping.py` leaga fiecare clasa prezisa de un profil cu boli asociate, nivel de severitate, descriere si actiune recomandata. Poate fi extins usor pe masura ce adaugi clase noi.

Niveluri de severitate:

| Nivel | Semnificatie |
|---|---|
| none | Fara anomalii |
| low | De obicei benign, de urmarit |
| medium | Necesita investigatie |
| high | Evaluare urgenta necesara |
| critical | Interventie imediata necesara |

---

## Atentie

HemoScan este destinat exclusiv scopurilor de cercetare si educatie. Rezultatele nu trebuie folosite ca unica baza pentru decizii clinice. Orice constatare trebuie validata de un medic specialist.

---

## Licenta

MIT
