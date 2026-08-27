# SEPIAS — Système d'exploration patrimoniale par images et analyse segmentée

Moteur de recherche d'images par le contenu (CBIR) pour les arts visuels et le patrimoine.
Travail de fin d'études — Master ingénieur civil en informatique et gestion, UMONS,
Faculté Polytechnique, 2026.

Le système combine deux représentations complémentaires (DINOv2 et CLIP) au moyen d'une
fusion pondérée, étend la recherche globale par une recherche hiérarchique sur des
sous-images issues d'une segmentation automatique, et applique le feedback utilisateur au
niveau de la région responsable d'un résultat plutôt qu'au niveau de l'image entière.

---

## Installation

```bash
pip install -r requirements.txt
```

Les scripts de segmentation nécessitent en outre les poids des modèles correspondants
(SAM, checkpoints SegNet et U-Net), non versionnés ici.

---

## Inventaire des scripts

Les scripts sont présentés dans l'ordre du pipeline. Chacun peut être exécuté
indépendamment, ce qui permet de reconstruire une seule partie de la chaîne.

### 1 · Construction du corpus

| Script | Rôle |
|---|---|
| `build_benchs_from_lists.py` | Constitue le corpus expérimental en copiant les images depuis l'archive WikiArt vers `benchs_full/`, selon les listes de `scene_lists/`. Produit les 615 images réparties en cinq scènes iconographiques. |

### 2 · Représentations globales

| Script | Rôle |
|---|---|
| `build_dino_dump.py` | Calcule l'embedding DINOv2 de chaque image complète et sérialise l'index global (`dump_dino_scenes_full.pk1`). Dimension 768. |
| `build_clip_dump.py` | Équivalent pour CLIP (`dump_clip_scenes_full.pk1`). Dimension 512. |

### 3 · Segmentation — génération des régions

| Script | Rôle |
|---|---|
| `segment_objects.py` | Segmentation d'instances par **Mask R-CNN** pré-entraîné sur COCO. Filtre les détections par seuil de confiance, plafonne le nombre d'objets par image, élargit les boîtes par un padding. Produit les sous-images et un manifest JSONL. |
| `segment_sam_auto.py` | Segmentation générique par **SAM** en mode automatique. Filtre les masques par surface minimale et relative maximale, ainsi que par les seuils de qualité et de stabilité propres au modèle. |
| `segment_segnet.py` | Inférence **SegNet** : seuillage de la carte d'objectness, post-traitement morphologique (fermeture, ouverture, remplissage de trous), extraction des composantes connexes en boîtes englobantes. |
| `segment_unet.py` | Idem pour **U-Net**. |
| `segment_sam3.py` | Segmentation **promptable par concept textuel** (« angel », « cross », etc.). Développé comme perspective, non évalué dans le mémoire. |

### 4 · Entraînement des réseaux d'objectness

| Script | Rôle |
|---|---|
| `pseudo_masks_sam_union.py` | Génère les pseudo-masques d'entraînement à partir de l'union des meilleurs masques SAM. C'est la supervision utilisée par SegNet et U-Net, en l'absence de masques annotés manuellement. |
| `train_segnet_objectness.py` | Entraîne SegNet à prédire une carte binaire d'objectness à partir de ces pseudo-masques (perte BCE + Dice). |
| `train_unet_objectness.py` | Idem pour U-Net. |

### 5 · Représentations locales

| Script | Rôle |
|---|---|
| `build_object_dump.py` | Calcule l'embedding DINOv2 de chaque sous-image et construit l'index local. Conserve pour chaque région ses métadonnées : image parente, boîte englobante, chemin de la sous-image, méthode de segmentation. Ce sont ces métadonnées qui permettent la remontée vers l'œuvre et l'affichage de la région responsable. |
| `build_object_clip_dump.py` | Équivalent pour CLIP. |
| `make_rectangular_crops_from_manifest.py` | Régénère les sous-images rectangulaires à partir des boîtes d'un manifest existant, avec un padding paramétrable. Utilisé pour tester l'effet du contexte autour de la région sans relancer la segmentation. |

### 6 · Évaluation *(scripts ayant produit les résultats du mémoire)*

| Script | Rôle | Produit |
|---|---|---|
| `evaluate_global_alpha.py` | Balayage du paramètre de fusion α sur la recherche globale, en leave-one-out sur les 615 images, avec intervalles de confiance par bootstrap. | **Tableau 4.3** |
| `evaluate_pre_post_v2.py` | Évaluation appariée pré-segmentation (recherche globale) contre post-segmentation (recherche hiérarchique). Implémente la véritable agrégation des sous-images vers l'image parente par maximum, ainsi que les variantes moyenne et top-*k*. Option `--cap_per_parent` pour plafonner le nombre de régions. Recalcule la référence globale sur le sous-ensemble couvert par chaque méthode, pour une comparaison appariée. | **Tableaux 4.4, 4.5 et 4.6** |
| `make_results_table.py` | Consolide les sorties CSV en un tableau unique, exportable en LaTeX. | — |

### 7 · Interface

| Script | Rôle |
|---|---|
| `app_query2parent_feedback_streamlit_compact_v2.py` | Interface web expérimentale (Streamlit). Segmente l'image requête à la volée, laisse l'utilisateur sélectionner les sous-images pertinentes, compare côte à côte recherche globale et recherche par sous-images, affiche la région responsable de chaque résultat et applique le feedback local. C'est la version utilisée pour toutes les captures du mémoire. |

Lancement :

```bash
streamlit run app_query2parent_feedback_streamlit_compact_v2.py
```

### 8 · Génération des figures

| Script | Rôle | Produit |
|---|---|---|
| `make_segmentation_comparison.py` | Planche comparant les régions produites par les quatre méthodes sur une même œuvre. | **Figure 3.2** |
| `make_global_dino_clip_fusion_example.py` | Exemple qualitatif comparant les résultats de DINOv2, CLIP et de la fusion. | **Figure 4.1** |
| `make_feedback_before_after.py` | Illustration de l'évolution du classement avant et après une itération de feedback local. | **Figure du feedback** |
| `find_global_qualitative_cases.py` | Recherche automatique de cas qualitatifs intéressants dans le corpus. | — |

---

## Scripts non utilisés dans les résultats finaux

Ces fichiers sont conservés pour la traçabilité du travail, mais **ne produisent pas les
chiffres du mémoire**. Les relancer donnerait des valeurs différentes de celles publiées.

| Script | Statut |
|---|---|
| `evaluate_scene.py` | Ancienne évaluation globale, sur 100 requêtes tirées aléatoirement. Remplacée par `evaluate_global_alpha.py` en leave-one-out. |
| `evaluate_scene_objects.py` | Ancienne évaluation locale. Moyennait les embeddings des sous-images d'une image avant une recherche image à image, ce qui ne correspond pas au mécanisme décrit dans le mémoire. **Remplacée par `evaluate_pre_post_v2.py`.** |
| `evaluate_global_fusion.py` | Ancienne évaluation de la fusion α, sur 100 requêtes aléatoires. Remplacée par `evaluate_global_alpha.py`. |
| `evaluate.py`, `evaluate_artist.py` | Évaluations par style et par artiste, menées sur le corpus WikiArt complet. Hors du périmètre du mémoire. |
| `annotate_benchmark.py`, `evaluate_benchmark.py` | Outillage d'un banc d'essai de motifs locaux annotés manuellement. Développé mais non exploité, faute de temps d'annotation. Identifié comme perspective principale dans la conclusion. |
| `main.py`, `query_dino.py` | Prototype initial fourni au début du travail (recherche globale DINOv2 seule). Conservé comme point de départ historique. |

---

## Données

Les images proviennent de **WikiArt** et ne sont pas redistribuées dans ce dépôt. Les
index sérialisés (`.pk1`), les sous-images et les poids de modèles ne sont pas versionnés
non plus : ils sont régénérables via les scripts des sections 1 à 5.

Le corpus expérimental compte 615 peintures réparties en cinq scènes iconographiques :
annonciation (168), crucifixion (125), cène (48), vierge à l'enfant (201), nativité (73).

---

## Reproduire les résultats du mémoire

```bash
# 1. Corpus et index globaux
python build_benchs_from_lists.py
python build_dino_dump.py
python build_clip_dump.py

# 2. Segmentation et index locaux (exemple avec Mask R-CNN)
python segment_objects.py
python build_object_dump.py
python build_object_clip_dump.py

# 3. Tableau 4.3 — recherche globale et balayage de alpha
python evaluate_global_alpha.py \
    --dino_dump dump_dino_scenes_full.pk1 \
    --clip_dump dump_clip_scenes_full.pk1 \
    --kmax 20 --out eval_global_alpha.csv

# 4. Tableaux 4.4 à 4.6 — comparaison pré/post-segmentation
python evaluate_pre_post_v2.py \
    --global_dump dump_dino_scenes_full.pk1 \
    --obj_dump dump_obj_maskrcnn_scenes_full.pk1 \
    --restrict_to_covered --kmax 20 --per_class \
    --out ppp_maskrcnn_dino.csv
```

À répéter pour chaque méthode de segmentation (`samauto`, `segnet`, `unet`) et pour la
représentation CLIP.
