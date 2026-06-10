# Passation / TODO — GUI Abaqus CEL (coupe orthogonale / rabotage)

Ce fichier sert à reprendre le projet dans une nouvelle session. Il décrit
l'état réel du code, ce qui reste à finir hors gros chantiers, puis les gros
chantiers à venir. Conventions clés rappelées en fin de fichier.

---

## 1. État actuel (fait)

- **Étage 1 — initialisation des paramètres simu** : onglets Analysis,
  Geometry, Materials, Interaction, BCs/ICs, Mesh, Step, Job, Results.
- **Mass scaling CEL** : appliqué par mise à l'échelle de la densité
  (`Density(rho * ms)` et `SpecificHeat(Cp / ms)`) car le `*Fixed Mass
  Scaling` natif ne s'applique pas aux éléments eulériens EC3D8R. ρ·Cp
  conservé, température physique correcte. La simulation tourne.
- **Extraction** : ROI via `getByBoundingBox`, réservé à l'instance
  eulérienne ; l'outil (Tool) est extrait entier. Champs : EVF/TEMP/S/PEEQ au
  CENTROID, V au NODAL moyenné par élément (magnitude `|V|`).
- **Affichage Results** : face robuste, masquage des cellules EULER où
  `EVF <= 1e-3`.
- **Étage 2 — étude de sensibilité (jacobien seul)** : tableau
  `Vary | Parameter | Ref | Min | Max | Delta | Delta% | Norm | Unit`, sync
  Delta↔Delta%, détrompeur trust-region (Delta en rouge si Ref±Delta sort de
  [Min,Max]), QoI scalaires (Fx/Fy/T/PEEQ) + Field QoI SSD (EVF/V/TEMP) sur la
  ZOI, runner Abaqus en thread (`run_worker`), estimation du temps live via le
  `.sta` du run en cours + durées mesurées. Morris retiré de l'UI (module
  conservé, import SALib paresseux → la GUI démarre sans SALib).
  - **Correctif** : `runner_core.eulerian_instance` appelait
    `bundle.instance_names()` (méthode) alors que `ResultsBundle`
    expose `instance_names` en *property* (liste). Sur un vrai bundle le
    `try/except` avalait le `TypeError` et renvoyait `None`, ce qui
    faisait disparaître **silencieusement** les Field QoI SSD. Corrigé
    (helper tolérant property/méthode) et couvert par un test sur un vrai
    bundle. Le mock historique masquait le bug (méthode au lieu de
    property).
  - **Cancel** durci : le sous-processus est lancé dans son propre
    groupe (POSIX `start_new_session` / Windows
    `CREATE_NEW_PROCESS_GROUP`) et l'annulation tue **tout l'arbre**
    (POSIX : SIGTERM puis escalade SIGKILL ; Windows : `taskkill /F /T`).
  - **Run échoué signalé en direct** : le signal `runDone(i, ok)` est
    désormais connecté ; un run KO est tracé immédiatement dans le log et
    le compteur d'échecs apparaît dans la ligne d'estimation.
  - **Export CSV** (`gui/sensitivity/export_results.py` + bouton « Save
    results… ») : une ligne par (QoI, paramètre), triée par
    |sensibilité| décroissante (= tableau + classement field-SSD).
- **Profil** : sauvegarde / chargement (`ModelConfig`, JSON), drapeau "dirty".
- **Preferences** : chemins normalisés en séparateurs natifs (Windows →
  backslash) à l'affichage, après Browse et à l'enregistrement.
- **Déploiement PC distant** : venv recréé depuis Anaconda + correctif SSL
  (ajout de `Library\bin` au PATH pour pip) ; lanceurs `run_gui.bat` et
  `run_gui_debug.bat` (chemin rapide si dépendances présentes, host Python
  requis uniquement pour créer le venv).

---

## 2. Reste à faire — hors gros chantiers

- [ ] **Validation bout-en-bout avec le vrai Abaqus** (reste à faire sur le
      PC distant). Un harnais de **répétition à blanc** sans Abaqus a été
      ajouté pour dérisquer (faux solveur `tests/abaqus_stub.py` +
      `tests/test_e2e_dryrun.py`) : il exerce le vrai worker, un vrai
      sous-processus, un vrai `.sta`, la relecture `.npz` et les Field QoI
      sur de vrais bundles. Le run réel doit encore confirmer :
  - lancement du sous-processus + streaming de sortie (cp1252) ;
  - écriture puis relecture du `.npz` (`ResultsBundle.load`) ;
  - estimation live via `.sta` : format `wall_time` « HH:MM:SS » et que
    `sens_run000.sta` est bien trouvé dans le workdir ;
  - calcul des Field QoI SSD (EVF/V/TEMP) sur la ZOI — **NB** : le
    dry-run couvre EVF/TEMP ; `V` n'est pas produit par `fake_builder`,
    donc la Field QoI `V` ne se valide qu'avec le vrai Abaqus ;
  - remplissage du tableau Results et du Chart ;
  - plausibilité physique des QoI (Fc/Ff, T) avec le mass-scaling ;
  - **Cancel** : confirmer le chemin Windows `taskkill /F /T` (non
    testable hors Windows ; le chemin POSIX est couvert par le dry-run).
- [ ] **Déploiement** : renseigner le vrai `abaqus.bat` dans Preferences
      (`where /r C:\ abaqus.bat`), créer le working directory, confirmer
      `run_gui_debug.bat`, vérifier l'affichage backslash sous Windows.
- [x] **Finitions Sensitivity (Cancel + échec)** : **Cancel** interrompt
      désormais tout l'arbre du sous-processus (groupe de processus +
      escalade SIGTERM→SIGKILL / `taskkill /T`) et un run en échec est
      signalé en direct dans l'UI (signal `runDone` connecté). Restent
      cosmétiques : proportions du layout côte-à-côte, rendu de la colonne
      Norm (élasticité), miroir CPU depuis l'onglet Job.
- [x] **Export des résultats** : bouton « Save results… » +
      `gui/sensitivity/export_results.py` (CSV : tableau de sensibilité et
      classement field-SSD, trié par |sensibilité|, utf-8-sig pour Excel).

---

## 3. Gros chantiers à venir

### 3.1 Acquisition / import des données expérimentales — onglet "Experimental Data"
Aujourd'hui : placeholder vide dans la catégorie "Experimental Data".
- [ ] **DIC** : import du champ de vitesse (corrélation d'images) ; lien avec
      `V`. Prévoir d'exposer `V1`/`V2` séparés (et pas seulement `|V|`), et la
      géométrie du copeau (à confronter à la frontière EVF du modèle).
- [ ] **IRT** : import du champ / relevé de température (lien `TEMP`).
- [ ] **Efforts** : import de Fc et Ff (lien `Fx=RF1=Fc`, `Fy=RF2=Ff`).
- [ ] **Points ouverts** : formats de fichiers, recalage spatial (repère
      DIC/IRT ↔ repère modèle), recalage temporel (fenêtre de régime établi),
      ré-échantillonnage / interpolation sur la ZOI.

### 3.2 Post-traitement / mise en correspondance simu ↔ essai
- [ ] Aligner les champs simulés (extraits sur la ZOI) et les mesures : même
      grille, même fenêtre temporelle (warmup / régime permanent), mêmes unités.
- [ ] Métriques d'écart simu↔essai en réutilisant `field_metrics`
      (SSD / L2 / RMSE) pour la vitesse et la température, et un écart scalaire
      pour Fc/Ff.

### 3.3 Calibration / identification inverse — onglet "Inverse identification"
Aujourd'hui : placeholder "Inverse identification — coming later".
- [ ] **Boucle d'optimisation** : paramètres à identifier (loi de
      comportement, frottement, …), fonction coût = écart pondéré simu↔essai
      (Fc/Ff + champs DIC/IRT), bornes = trust region du tableau, gradient
      fourni par le jacobien déjà en place.
- [ ] **Algorithme** : choisir (moindres carrés type Levenberg–Marquardt avec
      jacobien, ou sans gradient type Nelder-Mead / CMA-ES selon le bruit).
- [ ] **Évaluation des configs** : réutiliser `runner_core` / `run_worker`.
- [ ] **Optimisation des dimensions** : géométrie outil + domaine eulérien.
      Les paramètres géométriques sont aujourd'hui **exclus** de la sensibilité
      (`_EXCLUDED_CATEGORIES`, `_EXCLUDED_PATHS={"elem_size"}`) ; ils seront
      réintégrés ici.
- [ ] **Points ouverts** : pondération de QoI hétérogènes (efforts vs champs),
      normalisation / régularisation, critères d'arrêt, gestion des runs
      échoués, reprise.

---

## 4. Conventions / décisions à ne pas perdre

- **Mass scaling** : scaler ρ ET Cp du même facteur ; ne pas double-scaler
  (pas de pré-scaling manuel + facteur GUI en même temps). `improvedDtMethod=ON`.
- **ZOI** = la bbox ROI (éditée par le groupe "ROI" de l'onglet Geometry), pas
  le Set interne 'ROI'. Réservée à l'instance EULER (type d'élément en "EC").
- **QoI** : `Fx = RF1 = Fc`, `Fy = RF2 = Ff`.
- **Venv non portable** : recréer sur chaque machine ; venv issu d'Anaconda →
  ajouter `<base>\Library\bin` au PATH pour que pip ait SSL.
- **SALib** : importé paresseusement dans `morris_plan` → la GUI démarre sans.
- **Lancement Abaqus** : `<abaqus_cmd> cae noGUI=<script> -- --model_cfg ...
  --run_cfg ...`, cwd = workdir, résultats relus depuis `<job>.results.npz`.

---

## 5. Repères de code

- `gui/sensitivity/` : `param_registry.py`, `jacobian_plan.py`,
  `field_metrics.py`, `runner_core.py`, `run_worker.py`, `morris_plan.py`,
  `export_results.py`.
- `gui/results/` : `qoi.py`, lecture `ResultsBundle` (`reader.py`),
  `fake_builder.py` (option `field_scale` pour le dry-run).
- `gui/core/sta_parser.py` (parseur du `.sta` ; était listé à tort sous
  `gui/results/`).
- `gui/tabs/sensitivity_tab.py` : UI de l'étude (tableau, QoI, run, estimation).
- `gui/main.py` : onglets imbriqués — "Numerical Model", "Experimental Data"
  (placeholder), "Optimization" (Sensitivity + placeholder inverse).
- `abaqus_scripts/run_simul.py`, `abaqus_scripts/extract_odb.py`.
- Tests de non-régression : `tests/test_lot2a.py`, `test_lot2b.py`,
  `test_lot2c.py`, `test_runner_core.py`.
- Répétition à blanc (sans Abaqus) : `tests/abaqus_stub.py` (faux solveur,
  non collecté par pytest) + `tests/test_e2e_dryrun.py`.
