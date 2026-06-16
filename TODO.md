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
- **Ajouts UI (session courante)** :
  - Onglet Sensitivity : la colonne **Ref** est resynchronisée avec le
    Numerical Model courant. `showEvent` n'étant pas fiable pour une page
    de tab doublement imbriquée, le déclenchement passe par
    `currentChanged` des deux QTabWidget (méthode publique
    `SensitivityTab.refresh_from_model`, câblée dans `MainWindow`). Les
    lignes cochées préservent leur trust region et leur Delta%.
  - Onglet Materials : boutons **Save** (écrase le preset utilisateur
    sélectionné) et **Delete** (supprime un preset utilisateur ; les
    presets fournis sont protégés). Backend déjà présent
    (`PresetLibrary.save_user_preset` / `delete_user_preset`).
  - Onglet Step : estimation live du **stable increment time**
    Δt ≈ Lₑ/√(E/ρ) (matériau eulérien, taille de maille, mass scaling)
    et du **nombre d'incréments** ≈ sim_time/Δt.
  - Onglet Step : **V** et **A** reclassés comme sorties **nodales**
    (catégorie « Nodal »), plus dans la catégorie élément. Côté
    `run_simul.py`, la `FieldOutputRequest` ne force aucune position →
    Abaqus route déjà V/A à leur position nodale native.
  - Onglet Step : **time scaling** (κ_t, d'après Hammelmüller & Zehetner).
    `StepCfg.time_scaling_enabled/_factor`. Appliqué dans
    `to_params_dict` : vitesse de coupe ×κ_t, vitesse eulérienne initiale
    ×κ_t, sim_time ÷κ_t (longueur usinée conservée) ; et dans
    `run_simul.py` : Cp ÷κ_t (cumulé au mass scaling → Cp/(κ_m·κ_t)). ρ et
    E inchangés. Combinable avec le mass scaling. Indicateur speed-up
    ≈ κ_t (rappel κ_m = κ_t²) + avertissement si Johnson-Cook C ≠ 0
    (rate-dependent). **À valider sur Abaqus** : les modifications de
    `run_simul.py` (Cp) ne sont pas exécutables hors Abaqus.
  - **Système d'unités configurable** (`gui/core/unit_system.py`) : 4 unités
    de base (masse {kg,g,t}, longueur {m,mm,µm}, temps {ms,s}, température
    {°C,K}) d'où dérivent dimensionnellement densité, vitesse, taux de
    déformation, dilatation ; + surcharges nommées (module, résistance,
    conductivité, chaleur spécifique, énergie de rupture, vitesse). Presets
    (Engineering SI, SI strict, Millimetre). `units.py` délègue à un
    « système actif » ; le preset par défaut reproduit exactement les
    facteurs historiques. Enregistré dans le profil (`cfg.units`, round-trip
    JSON, fallback hérité depuis `ui.temp_unit`). Réglage via Preferences →
    « Unit system… » (remplace le toggle °C/K). Branché dans Materials, BCs
    (vitesse + température) et la colonne Ref de Sensitivity. **Limite
    connue** : le paramètre *vitesse* de l'onglet Sensitivity reste en
    m/min (spec non-matériau à facteur figé) ; les paramètres matériaux
    suivent le système.
  - Onglet Job : bouton **« Write .inp only »** — construit le modèle et
    écrit le deck `.inp` dans le workdir sans lancer le solveur
    (`run_simul.py` : flag `RUN_CFG["write_inp_only"]` →
    `myJob.writeInput()` puis `sys.exit(0)`, avant submit/extraction).
  - Onglet Sensitivity : **suivi temps réel** par lecture du `.sta`.
    L'estimation du temps total est construite explicitement comme
    (temps par frame mesuré = wall/frames_faites) × (frames/run) ×
    (nb de runs), affinée par les durées des runs terminés. Affiche
    « run i/N · frame f/F · ~x/frame · ~y/run · ~reste · est. total ·
    elapsed » ; barre de progression lissée (0–1000) incluant la fraction
    du run courant.
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

## Revue projet — recommandations priorisées appliquées

- **(1) `except Exception` resserrés.** Nouveau `gui/core/logging_util.py`
  (`log_swallowed(context, level)`, niveau via `GUI_ABAQUS_LOG`, défaut
  WARNING) : les avalements silencieux deviennent visibles **sans changer
  le flux**. Convertis dans le chemin critique (`runner_core`, `qoi`,
  `reader`, `run_worker`) et l'UI logique (`sensitivity_tab`, `results_tab`,
  `export_txt`, `field_viewer`). Les `except` qui remontaient déjà via
  QMessageBox / log panel laissés tels quels.
- **(2a) `extract_odb.py` supprimé** (code mort dupliqué) + préférence
  périmée `abaqus_extract_script` retirée (`preferences.py`) et son champ
  dans `preferences_dialog.py`. `load_preferences` filtre les clés inconnues
  → pas de casse des profils existants.
- **(2b) Morris rebranché dans l'UI Sensitivity.** Sélecteur de méthode
  Jacobian/Morris, contrôles N (trajectoires) + grid levels, `n_runs`/coût/
  génération branchés sur la méthode ; runs Morris = N×(k+1). Bug latent
  corrigé : `_collect_morris` lisait Min dans la colonne Ref (col 2) au lieu
  de col 3. Morris reste scalaire (les Field QoI sont jacobien-only) ; si
  SALib absent, message d'aide explicite.
- **(3) Paramètres de Job persistés.** `JobCfg{job_name, cpus}` dans
  `ModelConfig` (round-trip JSON, fallback défauts pour profils hérités).
  `job_tab` lit/écrit `cfg.job` (`apply_from_cfg` réel, câblé dans
  `_rebind_cfg`). Le workdir reste hors profil (spécifique machine, défaut
  via Preferences).
- **(4) `requirements.txt` épinglé** (PySide6 6.11.1, numpy 2.4.4,
  matplotlib 3.10.8, pytest 9.0.3 ; SALib optionnel pour Morris) + test
  « literal-only » sur `to_params_dict` (`ast.literal_eval(repr(d)) == d`).
- **(5) Checklist de validation Abaqus** : `docs/abaqus_validation_checklist.md`
  (mass/time scaling, write .inp, run, sensitivité, Cancel Windows, round-trip
  profil). Le stub `tests/abaqus_stub.py` reste pour les répétitions à blanc.

Tests : 30 verts (ajout : literal-only, persistance Job).

## Lot de simplifications par onglet

- **Analyse** : CEL uniquement. `analysis_tab` réécrit (plus de choix de
  formulation ni d'options Lagrangiennes) ; `formulation` forcé à "CEL".
  Les branches `is_lagrangian` des autres onglets deviennent inertes
  (toujours CEL), laissées en place (churn risqué).
- **Géométrie** : ROI sans zmin/zmax (face z=0 systématique). Les champs
  `BBox.zmin/zmax` restent au défaut pour capter la face z=0.
- **Materials** : chargement des profils refondu en système fichier — un
  dossier dédié (`presets.profiles_dir()`), boutons **Load…** (choisir un
  fichier profil) et **Save as…** (écrit/écrase un fichier). Boutons
  **Save**, **Delete**, **Copy** retirés (suppression = effacer le fichier).
- **Interaction** : pressure-overclosure toujours **Hard** (option retirée) ;
  **Fraction to master tool** retirée (master = 1 − slave).
- **Mesh** : onglet réécrit en minimal — seuls **elem_size** + **discretize**
  (+ aperçu + quantités dérivées). Toutes les sous-options d'éléments
  retirées (à régler par un expert dans le source) ; `MeshElementCfg` reste
  au défaut et est toujours émis au solveur.
- **Step** : groupes Field Output et History Output retirés (ce qui est
  calculé est fixé dans le source) ; extraction par défaut fixe (V + NT11
  nodaux, EVF élément, RF1/RF2 history synchronisés sur les frames Field
  Output, preselect). **Time scaling retiré** (UI, application dans
  `to_params_dict`, et Cp/κ_t dans run_simul neutralisé κ_t=1). Tests
  time-scaling supprimés.
- **Job** : choix du working directory retiré (toujours `prefs.default_workdir`,
  affiché en lecture seule).
- **Extraction (run_simul)** : sets **ZOI_nodes** / **ZOI_elems** créés depuis
  la ROI (face z=0) côté modèle ; à l'extraction, si la ZOI eulérienne est
  vide → arrêt avec une erreur explicite demandant d'agrandir la ROI
  (`sys.exit(2)`). L'ensemble de champs extrait était déjà fixe (EVF, V, TEMP).

Tests : 28 verts (les 2 tests time-scaling retirés).

- **Job — contrat par onglet (option 2, fait).** `to_params_dict` réécrit en
  une sous-config nommée par onglet (CEL-only) : `analysis`, `geometry`
  (tool/euler position+geometry, bbox), `materials` (euler/tool), `mesh`
  (elem_size, discretize, euler_element, tool_element), `interaction`, `bcs`,
  `step`. La clé `process` est supprimée (cutting_speed vient de `bcs`,
  sim_time/n_frames de `step`). Tous les chemins `cfg_get(MODEL_CFG, …)` de
  `run_simul.py` ont été remappés en conséquence (vérifié : chaque chemin lu
  se résout dans la nouvelle structure). L'aperçu dry-run montre une config
  par onglet.

## Sensibilité — colonne "ΔV % (rel)" (variation relative de champ)

- Nouvelle fonction `field_metrics.field_rel_change_pct(base, pert)` :
  variation relative du champ en %, **pondérée (moyenne) sur les nœuds et
  les frames** (RMS) — donc indépendante du nombre de points/frames et
  NaN-safe : `100 * sqrt(mean((pert-base)^2)) / sqrt(mean(base^2))`.
- `runner_core.jacobian_field_analysis` calcule `rel_pct` en plus de la
  sensibilité, et `analyze()` ajoute une colonne parallèle par variable de
  champ, id `"<var> Δ% (rel)"` (ex. `V Δ% (rel)`), affichée dans le tableau
  de résultats, le sélecteur de graphe et l'export CSV.
- Régression : `test_runner_core` vérifie rel(E)=100/E0 et rel(mu)=0.

## Experimental Data — incrément 1 (Acquisition / Import)

- **`gui/core/experiment_session.py`** : `ExperimentSession` (un `.json` par
  essai, séparé du profil modèle) — métadonnées (nom, matériau, vitesse
  nominale, notes), `trigger_offset_s` (t0 commun), `StreamCfg` visible/IR
  (path, noload_path, fps), `ForceCfg` (path, noload, fps, mapping colonnes
  t/Fc/Ff), refs vers les `.npz` DIC/IRT, et conteneurs ouverts pour
  calibration/géométrie de référence (remplis par les onglets suivants).
  Round-trip JSON tolérant (clés inconnues ignorées, blocs manquants → défauts).
- **`gui/core/sequence_io.py`** : `ImageSequence` (backend tableau testable +
  chargement best-effort dossier/tiff/vidéo/npz via imageio) et `load_forces`
  (CSV/texte → t, Fc, Ff, t dérivé de fps si pas de colonne temps).
- **Widgets** : `image_sequence_viewer.py` (imshow + curseur de frame, label
  « frame i/N · t »), `force_viewer.py` (Fc/Ff vs t).
- **`gui/tabs/experimental_data_tab.py`** : conteneur `ExperimentalDataTab`
  (8 sous-onglets, 7 placeholders) + `AcquisitionTab` (barre essai
  Load/Save session, import des 3 flux + acquisitions à vide, fps par flux,
  mapping colonnes efforts, prévisualisation visible/IR/efforts).
- **`main.py`** : placeholder « Experimental Data » remplacé par
  `ExperimentalDataTab` (possède sa propre `ExperimentSession`).
- **Contrat de champ** documenté dans `gui/results/FORMAT.md` (section
  expérimentale : `.npz` x,y (mm, repère modèle), t, V1/V2/Vmag ou T,
  `(n_frames, n_points)` ; `.json` = paramètres de calcul + bruit à vide).
- Tests : `tests/test_experimental.py` (9) + `tests/conftest.py` (fixture
  `qapp` partagée). **37 verts** au total.

### À discuter au moment venu (onglets suivants)
- DIC : détails (ROI/subset/maille Q4, critère de corrélation, sortie V1/V2).
- Calibration visible (Zhang/`q4dic`, mire R2L2S6N) et thermique (radiométrie
  + recalage IR↔visible).
- Alignement : bouton « Écrire dans Numerical Model » (rake/clear angles,
  (x0,y0) outil/pièce) — nécessitera de passer `cfg` + le geometry_tab au
  conteneur.
