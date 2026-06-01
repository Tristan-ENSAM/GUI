# Abaqus Cutting Pre-processor — Itération 1

Première itération : **onglet Geometry + preview 2D live + panneau de quantités dérivées**.

Les autres onglets (Materials, Interaction, BCs/ICs, Mesh, Job, Optimization) sont
des placeholders pour cette itération.

## Installation

```
pip install PySide6 matplotlib numpy
```

## Lancement (Windows, recommandé)

Double-cliquez sur **`run_gui.bat`** (situé à côté du dossier `gui/`).

Au premier lancement, le script crée un environnement virtuel `.venv\` à
côté du `.bat` et y installe `PySide6`, `matplotlib`, `numpy`
(durée : 2-5 min, taille : ~250 Mo). Les lancements suivants sont
instantanés et n'utilisent que ce venv local — votre Python système
n'est pas touché.

Pour forcer une réinstallation propre : supprimez le dossier `.venv\`
et relancez le `.bat`.

Si la GUI ne se lance pas, utilisez **`run_gui_debug.bat`** qui garde la
console ouverte et affiche les éventuelles erreurs Python.

## Lancement (autre OS / manuel)

```
pip install PySide6 matplotlib numpy
python -m gui.main
```
ou
```
python gui/main.py
```

## Arborescence

```
gui/
├── main.py                       Entrée: QMainWindow + QTabWidget
├── core/
│   └── model_config.py           Dataclass ModelConfig + sérialisation
│                                 → dict params au format abq_odb_generator.py
├── tabs/
│   └── geometry_tab.py           Formulaire Tool / WP / Euler / BBox
├── widgets/
│   ├── param_field.py            NumField / IntField / BoolField (réutilisables)
│   └── geometry_preview.py       Canvas Matplotlib 2D, mis à jour en live
```

## Ce qui marche

- Édition live des paramètres géométriques (outil, pièce, domaine eulérien,
  bbox ROI), preview Matplotlib synchronisée à chaque caractère tapé.
- Reproduction fidèle de la géométrie d'outil construite par
  `abq_odb_generator.py` (quadrilatère + filet de rayon `r_tool` au coin
  inférieur gauche = arête de coupe, angles de coupe et de dépouille
  appliqués via la même convention `90 + angle`).
- Discrétisation des dimensions du domaine eulérien au multiple inférieur
  de `elem_size` (cf. `discretize()` de votre script), visualisée en live.
- Quantités dérivées affichées :
    * Dimensions effectives après discrétisation
    * Nombre d'éléments eulériens estimé (mailles hex structurées,
      une seule à travers l'épaisseur, comme dans votre modèle)
    * **Δt stable estimé** : c_d = √(E/ρ), Δt ≈ elem_size / c_d
      (estimation premier ordre — Abaqus utilise une formule plus
      complexe documentée dans le Theory Manual, section "Stability
      limit for the explicit operator")
- `ModelConfig.to_params_dict()` produit le dict exactement au format
  attendu par `abq_odb_generator.py`, prêt à être passé à `ABQ.run_simul`.

## Limites connues / différences avec Abaqus

- La géométrie d'outil est une approximation 2D fidèle, mais l'esquisse
  Abaqus utilise un solveur de contraintes qui peut différer de quelques
  micromètres si l'on combine `rake` et `clear` extrêmes.
- L'estimation Δt utilise uniquement les propriétés du matériau eulérien
  (le pas le plus contraignant en pratique). L'outil pourrait être plus
  raide mais sa maille est plus grossière par construction.
- La `translateTo` qui colle la pièce contre l'outil n'est pas appliquée
  dans la preview ; on affiche la pièce à sa position de référence.

## Prochaines étapes proposées

1. **Onglet Materials** : éditeurs Johnson-Cook (Plastic / RateDependent /
   DamageInitiation / DamageEvolution) + Élastique / Conductivité / etc.
   avec menus déroulants de matériaux préenregistrés (cuivre, Ti-6Al-4V,
   acier 42CrMo4...). Sauvegarde/chargement JSON.
2. **Onglet Mesh** : raffinement par arête (vous utilisez déjà `seedEdgeByBias`
   sur l'outil), prévisualisation des semis.
3. **Onglet Job** : champs `cpus`, `domains`, choix workdir, bouton
   "Run" qui appelle `ABQ.run_simul`, log console intégrée.
4. **Onglet Interaction** : loi de frottement (coeff μ, formulation,
   `fraction` du `TangentialBehavior`).
5. **Optimisation** : on choisira l'algo quand vous aurez décidé de la
   fonction objectif (effort de coupe minimal ? calage expérimental ?).

Dites-moi par quoi on continue.
