# Revue de fiabilisation — GUI_Abaqus

Date : 2026-09-15 (phase 1), mise à jour phase 2 le même jour.

## Statut des constats

| ID | Constat | Sévérité | Statut |
|---|---|---|---|
| M1 | Cancel : le repli ne tue pas l'arbre de processus | Majeur | **CORRIGÉ** (commit `7a7631c`) |
| M2 | `domain_jacobian` livré sans câblage GUI | Majeur | **CORRIGÉ** — fonctionnalité abandonnée sur décision de Tristan, code supprimé (commit `70b43c0`) |
| M3 | Le Cancel gèle l'UI (les deux chemins) — ampleur révisée à la baisse | Majeur | **CLOS SANS CORRECTION** sur décision de Tristan : `terminate` s'exécute vite, le gel est jugé acceptable |
| M4 | `MASSEUL`/`VOLEUL` absents de l'ODB : le contrôle de conservation n'existe pas | Majeur | **OUVERT — cause CONFIRMÉE** : la requête lève à la construction, message d'Abaqus encore à récupérer |
| M5 | Le filtre Butterworth de champ ne s'applique PAS à `TEMP` ni `EVF`, contrairement à l'intention documentée | Majeur | **CORRIGÉ** — requêtes filtrée/non filtrée séparées + extraction rendue explicite (à vérifier par un `Write .inp only`) |
| m6 | `COORD` demandé mais indisponible pour `EC3D8RT` (toute la pièce) | Mineur | **CLOS — NON-PROBLÈME** : Tristan confirme que `COORD` est bien dans l'ODB et que ses display groups fonctionnent. L'avertissement ne porte que sur la variante élémentaire |
| m5 | Le repli `dataDouble` de `_read_data` reposerait sur une prémisse fausse | Mineur | **INFIRMÉ** — la vérification donne tort à mon hypothèse, le code est correct |
| m1 | Duplication de la construction de la commande Abaqus | Mineur | **CORRIGÉ** (`edf0cf6`) — `build_abaqus_args()` unique, 3 tests |
| m2 | Code d'extraction mort (`_TENSOR_REDUCERS`, von Mises) | Mineur | **CORRIGÉ** — supprimé sur décision de Tristan |
| m3 | `try/except: pass` autour d'une assignation qui ne peut échouer | Mineur | **CORRIGÉ** (`bb31f67`) |
| m4 | Suite de tests non tolérante à l'absence d'`imageio` | Mineur | **CORRIGÉ** (`5ac4255`) — `importorskip`, suite verte |

## Vérification introspective — résultats réels (Abaqus 2022 HF8 de Tristan)

`_review/check_api.py` v1 a été exécuté sur l'installation réelle
(`abaqus cae noGUI=_review/check_api.py`). Sortie complète :
`_review/check_api_report.txt` côté machine Abaqus.

**VÉRIFIÉ (preuve = sortie du script) :**

| Élément | Résultat |
|---|---|
| Interpréteur Abaqus | **Python 2.7.15** (MSC v.1928, 64 bit) — confirme la contrainte 2.7 de `cel_common.py:15-24`, qui n'était qu'une hypothèse jusqu'ici |
| Les 44 constantes `abaqusConstants` utilisées | **toutes FOUND**, sans exception (`EC3D8RT`, `C3D8RT`, `NON_REFLECTING`, `ZERO_PRESSURE`, `EQUILIBRIUM`, `INFLOW`/`OUTFLOW`/`BOTH`, `JOHNSON_COOK`, `CONSTANTPRESSURE`, `MISES`, `PRESS`, `PRESELECT`, …) |
| `abaqus.mdb`, `mdb.Model`, `mdb.Job` | FOUND |
| `regionToolset.Region`, `mesh.ElemType`, `odbAccess.openOdb` | FOUND |

C'est la confirmation la plus utile de l'audit : **aucune constante symbolique
inventée ou mal nommée** dans tout le modèle CEL.

**Les trois `[MISSING]` du rapport sont un DÉFAUT DU SCRIPT DE VÉRIFICATION,
pas un constat sur le projet.** `sketch.ConstrainedSketch`, `part.Part` et
`material.Material` sont signalés absents parce que v1 testait des noms au
niveau MODULE que le code de production n'utilise jamais : `cel_model.py`
appelle `model.ConstrainedSketch(...)`, `model.Part(...)`,
`model.Material(...)` — des **méthodes de l'objet Model**. v1 vérifiait une
façade inexistante au lieu de l'API réellement employée. Aucune conclusion
sur le projet ne peut être tirée de ces trois lignes.

**Ce que v1 n'a PAS couvert du tout** (il se contentait d'imprimer une note
« à vérifier à la main ») — c'est-à-dire l'essentiel de l'inventaire :
les 18 méthodes de `Model`, toutes les méthodes de `rootAssembly`
(`seedEdgeByBias`, `DiscreteFieldByVolumeFraction`, `generateMesh`,
`setElementType`…), les méthodes de `Material`, celles de `ContactProperty`,
de `Job` (`writeInput`/`submit`/`waitForCompletion`), et **toute l'API
d'extraction ODB** de `cel_results.py`.

**`check_api.py` v2** corrige les deux problèmes : il instancie un modèle
jetable en mémoire et introspecte les objets réels (puis le supprime ; il ne
construit aucune géométrie, ne maille rien, ne soumet rien), et accepte un
`.odb` existant pour couvrir `cel_results.py` :

```
abaqus cae noGUI=_review/check_api.py -- --odb C:\TEMP\ABQ_wd\<job>.odb
```

La branche ODB est celle qui compte le plus pour les points restants : elle
seule peut confirmer `getSubset` / `getScalarField` / `dataDouble`, et
surtout lister les `historyOutputs` réels — ce qui vérifie d'un coup
**`MASSEUL`/`VOLEUL`** (cel_model.py:615-624, le point le plus risqué de
l'inventaire, entouré d'un `try/except` justement parce que le doute
existait), `RF1`/`RF2` et `ALLKE`/`ALLIE`, ainsi que les noms suffixés par
les filtres Butterworth (`RF1_SENSORBAND`, `V_CAMERABAND`) que
`_find_history_key` et `_resolve_fo_name` tentent de résoudre.

### Résultats de `check_api.py` v2 (exécuté avec `--odb GCI_run000.odb`)

**Désormais VÉRIFIÉ — l'essentiel de l'inventaire de construction du modèle :**

| Groupe | Résultat |
|---|---|
| 14 méthodes `Model` (`ConstrainedSketch`, `Part`, `Material`, `EulerianSection`, `HomogeneousSolidSection`, `ContactProperty`, `ContactExp`, `TempDisplacementDynamicsStep`, `ButterworthFilter`, `EulerianBC`, `VelocityBC`, `Velocity`, `MaterialAssignment`, `Temperature`, `rootAssembly`) | **FOUND** |
| **Les 17 méthodes de `rootAssembly`** — dont `seedEdgeByBias`, `DiscreteFieldByVolumeFraction`, `setElementType`, `setMeshControls`, `generateMesh`, `ReferencePoint`, `Surface` | **toutes FOUND** |
| 10 méthodes `ConstrainedSketch` (dont `FilletByRadius`, `ObliqueDimension`, `AngularDimension`) | **toutes FOUND** |
| 3 méthodes `ContactProperty` (`TangentialBehavior`, `NormalBehavior`, `HeatGeneration`) | **toutes FOUND** |
| `Part.BaseSolidExtrude`, `Part.SectionAssignment`, `Part.cells` | **FOUND** |
| `Job.writeInput`, `.submit`, `.waitForCompletion`, `.status`, `.messages` | **FOUND** |
| `Odb.steps/.rootAssembly/.close`, `FieldOutput.getSubset/.getScalarField/.componentLabels/.values`, `FieldValue.data/.elementLabel/.nodeLabel` | **FOUND** |

**La justification de `_check_job_succeeded` est VÉRIFIÉE.** Le script a lu, en
mode `noGUI`, `job.status = None` et `job.messages = []` — exactement ce que
décrit le commentaire de `cel_model.py:810-824`. Écarter `job.status` au
profit du `.sta` n'était donc pas une superstition : c'était nécessaire.

**Le mécanisme de suffixage par les filtres est VÉRIFIÉ**, et il valide le
code de résolution :
- `V` n'existe QUE sous `V_CAMERABAND` (le nom nu est absent) ;
- `RF1`/`RF2` n'existent QUE sous `RF1_SENSORBAND`/`RF2_SENSORBAND` ;
- `EVF` se résout en `['EVF_ASSEMBLY_EULER_EULER-1', 'EVF_VOID']` — exactement
  la forme que décrit la docstring de `_resolve_fo_name` (cel_results.py:192-202).

Sans `_resolve_fo_name` / `_find_history_key`, un `"RF1"` ou `"V"` codé en dur
ne trouverait rien. Ces deux fonctions ne sont pas défensives « au cas où » :
elles sont indispensables sur cet ODB.

**Trois `[MISSING]` + un `[ERROR]` sont encore des DÉFAUTS DE MON SCRIPT**, pas
des constats sur le projet — v3 les corrige :

| Symptôme v2 | Cause réelle |
|---|---|
| `Model.RigidBody`, `Model.FieldOutputRequest`, `Model.HistoryOutputRequest` MISSING | v2 n'importait que `abaqus`. Ces méthodes ne sont greffées sur `Model` qu'en important les modules CAE (`step`, `interaction`…), ce que `cel_model.py:17-25` fait. **Preuve qu'elles marchent : l'ODB contient 501 frames de sortie de champ et `RF1`/`RF2` sur le point de référence du corps rigide.** |
| `model.Material(...)` → `invalid name` | Abaqus refuse un nom commençant par `_`. Mon nom d'objet jetable était `_check_api_mat`. Les méthodes `Material` restent donc NON VÉRIFIÉES. |
| `FieldValue.dataDouble` MISSING | Sondé sur `CPRESS General_Contact_Domain`, une sortie de contact que le projet n'extrait pas. Ne permet de conclure ni dans un sens ni dans l'autre (voir m5). |

### Résultats de `check_api.py` v3 — inventaire clos

Exécuté avec le même ODB. Résumé du script : **`constants missing : 0`,
`[MISSING] rows : 0`, `[ERROR] rows : 0` — « Every symbol checked exists on
this installation. »**

Les trois défauts de script sont confirmés comme tels : après l'ajout des
imports CAE (`step`, `interaction`, `load`, …, tous `[OK]`),
`Model.RigidBody`, `Model.FieldOutputRequest` et `Model.HistoryOutputRequest`
passent en **FOUND**. Le diagnostic était le bon. Avec un nom d'objet valide,
les 8 méthodes `Material` et la chaîne `Plastic(...).RateDependent`
ressortent **toutes FOUND**.

**Bilan de l'inventaire API : VÉRIFIÉ.** Tous les symboles employés par
`cel_model.py` et `cel_results.py` — 44 constantes, 18 méthodes `Model`,
17 méthodes `rootAssembly`, 10 de `ConstrainedSketch`, 8 de `Material`,
3 de `ContactProperty`, 3 de `Part`, 5 de `Job`, et l'API d'extraction ODB —
existent sur Abaqus 2022 HF8.

**Reste NON VÉRIFIÉ — et ne le sera pas par cette voie : les noms de
mots-clés.** `hasattr` prouve qu'une méthode existe, pas que
`secondOrderAccuracy=`, `improvedDtMethod=` ou `nodalOutputPrecision=` sont
les bons noms d'arguments. Cela dit, ces arguments-là sont indirectement
attestés : le modèle se construit, tourne, et produit l'ODB examiné ici —
un nom de mot-clé erroné ferait lever l'appel. La réserve porte donc sur les
chemins non exercés par ce run précis (les branches `ROUGH`/`FRICTIONLESS`,
les lois de contact autres que `HARD`, `HeatGeneration`, les types
d'inflow/outflow non utilisés).

**Et une réserve qui, elle, s'est matérialisée : `MASSEUL`/`VOLEUL`.** Un nom
de VARIABLE DE SORTIE n'est vérifiable par aucun `hasattr` — seul l'ODB le
dit. C'est précisément là qu'un défaut se cachait (M4).

---

## Phase 1 — audit en lecture seule

Date : 2026-09-15
Portée : `abaqus_scripts/`, `gui/`, `tests/`, `docs/abaqus_validation_checklist.md`.
Aucune modification de code de production n'a été faite dans cette phase ; seuls
`_review/REVIEW.md` et `_review/check_api.py` (nouveaux fichiers) ont été créés.

**Limite majeure de cet audit, à lire avant tout le reste** : l'environnement
dans lequel cette revue a été effectuée est un conteneur Linux **sans Abaqus
installé** (`which abaqus` → introuvable, aucun `C:\SIMULIA`). Le point 3 de
la mission ("Vérification contre l'installation réelle") n'a donc **pas pu
être exécuté**. Tout ce qui concerne l'existence et la signature exacte des
symboles de l'API Scripting Abaqus (`mdb.Model`, `ElemType`, `EulerianBC`,
`odbAccess.openOdb`, etc.) est classé **NON VÉRIFIÉ** ci-dessous : c'est une
lecture du code, pas une confirmation par introspection. Un script
d'introspection prêt à l'emploi est fourni (`_review/check_api.py`) — voir
section "Vérification introspective" plus bas pour la procédure à exécuter
sur la machine Abaqus.

Ce qui a en revanche pu être fait dans cet environnement, et l'a réellement
été :
- lecture complète de `cel_model.py`, `cel_results.py`, `cel_common.py`,
  `run_simul.py`, `job_tab.py`, `run_worker.py`, `runner_core.py`,
  `sta_parser.py`, `mass_scaling.py`, `model_config.py`, `qoi.py`, et une
  revue ciblée (grep + lecture partielle) du reste de `gui/` ;
- exécution réelle de la suite de tests, headless, en deux temps, après
  installation de `requirements.txt` dans un venv dédié et des bibliothèques
  système Qt manquantes (`libegl1` et dépendances) — sorties réelles
  rapportées en fin de document, pas de simulation.

## Résumé (5–10 lignes)

Le pipeline GUI → `run_simul.py` → `cel_model.py`/`cel_results.py` → `.npz`/
`.json` → viewers est cohérent et repose exclusivement sur l'API Scripting
Abaqus « directe » (aucune réécriture texte du `.inp`, aucun `keywordBlock`,
aucune réimportation CAE d'un `.inp` généré) : c'est le point le plus
rassurant de cette revue. Deux problèmes concrets, reproduits localement,
méritent une correction avant la phase 2 : (1) l'annulation d'un run lancé
depuis l'onglet Job (`JobTab._cancel_run`) ne tue pas l'arbre de processus
sur le fallback, contrairement à `SensitivityRunWorker.cancel()` qui le fait
— risque de processus solveur orphelins documenté par le projet lui-même
dans le checklist mais non implémenté dans ce chemin ; (2) la fonctionnalité
« Domain sizing by Jacobian » a été livrée avec son moteur et ses tests mais
sans câblage dans `OptimizationTab` — 6 tests échouent avec `AttributeError`
sur le commit courant (`HEAD` = `79b66ad`). La suite de tests (597 tests,
deux passes headless demandées) donne 588 réussites / 9 échecs ; 3 des 9
échecs sont une dépendance de test optionnelle absente (`imageio`, non fautif
côté production) et 6 sont le défaut ci-dessus. Aucune méthode Abaqus
« inventée » n'a été trouvée ; le seul point ouvert est que rien n'a pu être
confirmé contre une installation réelle dans cet environnement.

## Cartographie (fait)

- **Point d'entrée Abaqus** : `abaqus_scripts/run_simul.py`, lancé par
  `abaqus cae noGUI=run_simul.py -- --model_cfg <repr> --run_cfg <repr>`
  (`gui/tabs/job_tab.py:521-529`, `gui/sensitivity/run_worker.py:223-226`).
  Exécuté sous l'interpréteur Python embarqué d'Abaqus (Python 2.7 pour
  Abaqus 2022 HF8 — **NON VÉRIFIÉ** dans cet environnement, aucun accès à
  `abaqus python -c "import sys; print(sys.version)"`; affirmation reprise
  du commentaire du projet lui-même, `abaqus_scripts/cel_common.py:15-24`
  et `requirements.txt:3-4`).
- **Côté GUI (Python 3.x)** : `gui/tabs/*.py` (onglets), `gui/core/*.py`
  (config, helpers), `gui/sensitivity/*.py` (campagnes multi-runs,
  sizing, mass scaling), `gui/results/*.py` (lecture `.npz`/`.json`,
  QoI), `gui/widgets/*.py` (viewers).
- **Côté Abaqus (Python 2.7)** : `abaqus_scripts/cel_model.py` (construction
  du modèle CEL + job), `abaqus_scripts/cel_results.py` (extraction ODB →
  `.npz`/`.json`), `abaqus_scripts/cel_common.py` (helpers purs partagés,
  volontairement sans dépendance 3.x — voir son docstring).
- **Chemin complet paramètres → résultats** :
  1. `ModelConfig.to_params_dict()` (`gui/core/model_config.py`) sérialise
     l'état des onglets en un dict `model_cfg` (clés à points, ex.
     `"geometry.bbox.xmin"`) ;
  2. `JobTab._launch_abaqus` (ou `SensitivityRunWorker._abaqus_solve`)
     construit `run_cfg` et lance `abaqus cae noGUI=run_simul.py -- ...` via
     `QProcess` (Job tab) ou `subprocess.Popen` (campagnes) ;
  3. `run_simul.py:parse_arguments` fait `ast.literal_eval` sur les deux
     `repr()` reçus en argument ;
  4. `cel_model.prepare_parameters` normalise, `build_model` construit le
     modèle CEL, `create_job`+`run_job` écrit le `.inp` ou soumet et attend
     la fin (`job.waitForCompletion()`), avec une vérification de succès
     indépendante de `job.status` fondée sur le `.sta` (voir plus bas) ;
  5. `cel_results.extract_results` ouvre le `.odb` et écrit
     `<job>.results.npz` + `<job>.meta.json` ;
  6. côté GUI, `gui/results/reader.py` (`ResultsBundle.load`) relit le
     bundle, `gui/results/qoi.py` réduit aux QoI scalaires, les widgets
     (`field_viewer.py`, `force_viewer.py`, `time_series_viewer.py`)
     affichent.
  7. En parallèle, `gui/core/sta_parser.py` lit le `.sta` pendant
     l'exécution pour alimenter la barre de progression (`JobTab._poll_sta`,
     timer 800 ms).

## Tableau d'inventaire des appels API Abaqus

Le statut **NON VÉRIFIÉ** des lignes ci-dessous date de la phase 1. Il a été
**partiellement levé depuis** — voir la section « Vérification introspective
— résultats réels » en tête de document :
- les **constantes symboliques** de toutes ces lignes (`EULERIAN`, `EC3D8RT`,
  `JOHNSON_COOK`, `NON_REFLECTING`, `PRESELECT`, `MISES`, …) sont désormais
  **VÉRIFIÉES** : les 44 existent sur l'installation de Tristan ;
- les points d'entrée `mdb.Model`, `mdb.Job`, `regionToolset.Region`,
  `mesh.ElemType`, `odbAccess.openOdb` sont **VÉRIFIÉS** ;
- les **méthodes** (`model.EulerianBC`, `assembly.seedEdgeByBias`,
  `job.writeInput`, `FieldOutput.getScalarField`, …) et surtout les **noms
  de mots-clés** restent **NON VÉRIFIÉS** : `check_api.py` v1 ne les
  couvrait pas. v2 le fait, il reste à l'exécuter.

La colonne « lecture » n'indique que la cohérence apparente avec l'API
Scripting — ce n'est pas une preuve.

### Construction du modèle — `abaqus_scripts/cel_model.py`

| Appel | fichier:ligne | Usage | Statut |
|---|---|---|---|
| `mdb.Model(name=, absoluteZero=-273.15)` | cel_model.py:779 | crée le modèle | NON VÉRIFIÉ |
| `model.ConstrainedSketch(...)` / `.rectangle` | cel_model.py:281-282, 287-288 | sketches Euler/WP | NON VÉRIFIÉ |
| `model.Part(..., type=EULERIAN\|DEFORMABLE_BODY)` + `BaseSolidExtrude` | cel_model.py:283-315 | parts Euler/WP/Tool | NON VÉRIFIÉ |
| `sketch.Spot/FixedConstraint/Line/HorizontalConstraint/VerticalConstraint/FilletByRadius/CoincidentConstraint/ObliqueDimension/AngularDimension` | cel_model.py:294-313 | sketch paramétrique de l'outil | NON VÉRIFIÉ |
| `model.Material(...)` + `.Density/.Elastic/.Conductivity/.SpecificHeat/.Expansion/.InelasticHeatFraction/.Plastic(...).RateDependent(...)` | cel_model.py:326-353 | matériaux Euler/Tool | NON VÉRIFIÉ |
| `Material.JohnsonCookDamageInitiation(...).DamageEvolution(...)` | cel_model.py:343-347 | **commenté / inactif** | non applicable |
| `model.EulerianSection` / `model.HomogeneousSolidSection` / `Part.SectionAssignment(region=Region(cells=...))` | cel_model.py:360-363 | sections | NON VÉRIFIÉ |
| `assembly = model.rootAssembly` ; `assembly.Instance(..., dependent=OFF)` ; `assembly.translate` ; `Instance.translateTo(movableList=, fixedList=, direction=, clearance=)` ; `assembly.excludeFromSimulation` | cel_model.py:376-389 | assemblage + positionnement | NON VÉRIFIÉ |
| `assembly.seedPartInstance` ; `mesh.ElemType(elemCode=EC3D8RT\|C3D8RT, elemLibrary=EXPLICIT, secondOrderAccuracy=OFF, hourglassControl=DEFAULT)` ; `assembly.Set` ; `assembly.setElementType` ; `assembly.setMeshControls(elemShape=HEX, technique=STRUCTURED\|SWEEP)` ; `assembly.seedEdgeBySize/seedEdgeByNumber/seedEdgeByBias` ; `assembly.generateMesh` | cel_model.py:400-443 | maillage Euler + Tool | NON VÉRIFIÉ |
| `assembly.ReferencePoint` ; `assembly.referencePoints[...]` ; `assembly.Set(referencePoints=...)` ; `assembly.DiscreteFieldByVolumeFraction` ; `Instance.nodes.getByBoundingBox` / `.elements.getByBoundingBox` | cel_model.py:456-483 | RP outil, VolFraction Eulérien, sets ROI_node/ROI_elem | NON VÉRIFIÉ |
| `model.ContactProperty` + `.TangentialBehavior(formulation=PENALTY\|ROUGH\|FRICTIONLESS)` + `.NormalBehavior(pressureOverclosure=...)` + `.HeatGeneration` ; `model.ContactExp(contactPropertyAssignments=((GLOBAL, SELF, name),))` ; `model.RigidBody` | cel_model.py:500-535 | contact général + corps rigide outil | NON VÉRIFIÉ |
| `model.TempDisplacementDynamicsStep(nlgeom=ON, linearBulkViscosity=0.06, quadBulkViscosity=1.2, improvedDtMethod=ON)` | cel_model.py:545-549 | step Explicit thermo-mécanique | NON VÉRIFIÉ |
| `model.ButterworthFilter` ; `model.FieldOutputRequest(..., filter=)` ; `model.HistoryOutputRequest(..., filter=)` | cel_model.py:563-624 | filtres + sorties champ/historique | NON VÉRIFIÉ |
| `HistoryOutputRequest(region=assembly.sets['Euler'], variables=('MASSEUL','VOLEUL'))` dans un `try/except Exception` non-fatal | cel_model.py:615-624 | garde-fou conservation masse | NON VÉRIFIÉ (nom de variable `MASSEUL`/`VOLEUL` en particulier) |
| `Instance.faces.getByBoundingBox` ; `assembly.Surface(side1Faces=)` ; `model.EulerianBC(definition=INFLOW\|OUTFLOW\|BOTH, inflowType=, outflowType=)` ; addition de `FaceArray` (`+`) | cel_model.py:672-695, 707-724 | BC Eulériennes par face | NON VÉRIFIÉ |
| `model.VelocityBC(v1=SET,...)` + `.setValuesInStep` ; `model.Velocity(distributionType=MAGNITUDE)` ; `model.MaterialAssignment(useFields=True, fieldList=...)` ; `model.Temperature` | cel_model.py:737-773 | vitesse de coupe, vitesse initiale, affectation matière par champ, température initiale | NON VÉRIFIÉ |
| `mdb.Job(type=ANALYSIS, numCpus=, numDomains=, explicitPrecision=DOUBLE, nodalOutputPrecision=FULL)` | cel_model.py:798-806 | création du job | NON VÉRIFIÉ |
| `job.writeInput(consistencyChecking=OFF)` | cel_model.py:958 | écriture `.inp` seule | NON VÉRIFIÉ |
| `job.submit(consistencyChecking=OFF)` ; `job.waitForCompletion()` | cel_model.py:966-967 | soumission + attente | NON VÉRIFIÉ |
| `job.status` / `job.messages` | **délibérément NON utilisé** — voir cel_model.py:810-824 | — | voir section "méthode la plus directe" |

### Extraction ODB — `abaqus_scripts/cel_results.py`

| Appel | fichier:ligne | Usage | Statut |
|---|---|---|---|
| `odbAccess.openOdb(path, readOnly=True)` | cel_results.py:13, 502 | ouverture ODB en lecture seule | NON VÉRIFIÉ |
| `odb.steps[name]` ; `step.frames` ; `frame.frameValue` | cel_results.py:504-507 | pas de temps | NON VÉRIFIÉ |
| `odb.rootAssembly.instances` ; `Instance.nodes`/`.elements` ; `Node.coordinates`/`.label` ; `Element.type`/`.connectivity`/`.label` | cel_results.py:70-95, 514-530 | géométrie par instance | NON VÉRIFIÉ |
| `FieldOutput.getSubset(region=)` / `.getSubset(position=CENTROID\|NODAL)` / `.getScalarField(invariant=MISES\|PRESS)` / `.componentLabels` / `Value.data` / `Value.dataDouble` / `.elementLabel` / `.nodeLabel` | cel_results.py:210-408 | résolution et lecture des champs (EVF/TEMP/V) | NON VÉRIFIÉ |
| `step.historyRegions` ; `HistoryRegion.historyOutputs` ; `HistoryOutput.data` | cel_results.py:444-477 | RF1/RF2, ALLKE/ALLIE | NON VÉRIFIÉ |
| `odb.close()` | cel_results.py:692 (dans un `finally`) | fermeture propre | NON VÉRIFIÉ |

### Commandes CLI Abaqus

| Commande | fichier:ligne | Usage | Statut |
|---|---|---|---|
| `<abaqus_cmd> cae noGUI=<script> -- --model_cfg <repr> --run_cfg <repr>` | job_tab.py:521-529 ; run_worker.py:223-226 | lancement build+solve+extract, ou write-inp-only (`--run_cfg` porte `write_inp_only=True`) | NON VÉRIFIÉ (syntaxe `cae noGUI=` documentée publiquement mais non confirmée sur cette install) |
| `<abaqus_cmd> job=<name> continue cpus=<n>` | job_tab.py:519 | reprise d'un job interrompu depuis ses fichiers de restart | NON VÉRIFIÉ |
| `<abaqus_cmd> terminate job=<name>` (exécuté avec `cwd=workdir`, lit `<job>.cid`) | run_worker.py:43-74 (utilisé aussi par job_tab.py:745) | arrêt propre + libération des jetons de licence | NON VÉRIFIÉ |
| `taskkill /F /T /PID <pid>` (fallback Windows, hors Abaqus) | run_worker.py:90-95 | tue l'arbre de process quand `terminate` échoue | commande Windows standard, pas Abaqus — NON VÉRIFIÉ en exécution (le projet le dit lui-même : "cannot be exercised in the Linux dev/CI environment", run_worker.py:86-87) |

## Constats par sévérité

### Critique

Aucun constat classé Critique n'a été identifié dans le code — sous réserve
que la vérification introspective (section suivante) ne révèle pas un
symbole d'API mal nommé, ce qui reste possible tant qu'elle n'a pas été
exécutée sur l'installation réelle.

### Majeur

**M1 — `JobTab._cancel_run` ne tue pas l'arbre de processus sur le chemin de
repli, contrairement à `SensitivityRunWorker.cancel()`.** — **CORRIGÉ**,
commit `7a7631c` : ajout de `kill_process_tree_by_pid()`
(`gui/sensitivity/run_worker.py`), appelée par `JobTab._cancel_run` avant le
repli `terminate()`/`kill()`. `abaqus terminate job=<name>` reste la première
route, conformément à la confirmation de Tristan. La fonction est
volontairement Windows-only : sur POSIX un enfant `QProcess` partage le
groupe de processus de la GUI, donc `killpg` tuerait la GUI elle-même — elle
renvoie `False` et l'appelant retombe sur son kill mono-processus. 4 tests de
non-régression ajoutés dans `tests/test_abaqus_terminate.py`.
- Fichier : `gui/tabs/job_tab.py:711-757`.
- Statut : **fait** (comparaison directe de deux implémentations dans le
  même dépôt) + **interprétation** sur la conséquence côté OS (le
  comportement de `QProcess.kill()`/`.terminate()` sur Windows — qu'il
  n'agit que sur le process direct, pas sur ses enfants — est un fait
  documenté par Qt en général, non vérifié spécifiquement ici).
- Preuve : `_cancel_run` appelle d'abord `abaqus_terminate_job(...)` (la
  route propre) puis, en repli, seulement
  `self._proc.terminate()` / `self._proc.kill()` (job_tab.py:754-756) —
  ce sont des méthodes `QProcess`, qui sur Windows envoient
  `WM_CLOSE`/appellent `TerminateProcess` **uniquement sur le processus
  fils direct** (`abaqus.bat`/`cae.exe`), pas sur les processus qu'il a
  lui-même engendrés (pre/package/standard.exe/explicit.exe). À l'inverse,
  `gui/sensitivity/run_worker.py:77-131`
  (`_terminate_process_tree`) fait explicitement
  `taskkill /F /T /PID <pid>` sur Windows — le `/T` tue l'arbre — et le
  commentaire du fichier de checklist (`docs/abaqus_validation_checklist.md:80-83`)
  décrit précisément ce comportement ("`taskkill /F /T`, no orphaned
  standard.exe/explicit.exe") comme le comportement ATTENDU d'un Cancel,
  mais seulement `SensitivityRunWorker` l'implémente. Il y a donc deux
  mécanismes de cancel dans le dépôt, un correct (campagnes de
  sensibilité), un incomplet (onglet Job, chemin le plus utilisé en usage
  interactif).
- Conséquence si confirmé : un Cancel depuis l'onglet Job peut laisser un
  process solveur (standard.exe/explicit.exe) tourner en arrière-plan sur
  Windows après que l'utilisateur croit l'avoir arrêté, avec jeton de
  licence non libéré et fichiers `.odb`/`.lck` potentiellement encore
  écrits.
- Corrections possibles (alternatives, sans trancher) :
  (a) faire appeler `_terminate_process_tree`-équivalent (ou une variante
  utilisant `QProcess.processId()` + `taskkill /F /T`) depuis
  `job_tab.py:754-756`, en réutilisant le PID exposé par `QProcess` ;
  (b) factoriser un seul chemin de cancel partagé entre `JobTab` et
  `SensitivityRunWorker` (actuellement dupliqué, cf. M2) pour éliminer le
  risque de divergence future.
- Ce point ne peut pas être testé en CI Linux (le projet le reconnaît
  lui-même pour le chemin `taskkill`) ; le test qui existe
  (`tests/test_abaqus_terminate.py`) couvre la fonction `abaqus_terminate_job`
  et `_terminate_process_tree` (POSIX) isolément, mais pas
  `JobTab._cancel_run` lui-même.

**M2 — La fonctionnalité « Domain sizing by Jacobian » est livrée
incomplète : moteur + tests présents, câblage GUI absent.** — **CORRIGÉ**,
commit `70b43c0` : Tristan a confirmé que l'étude est ABANDONNÉE et qu'aucun
câblage n'est envisagé. `gui/sensitivity/domain_jacobian.py`,
`gui/sensitivity/domain_jacobian_worker.py`, `tests/test_domain_jacobian.py`
et `tests/test_domain_jacobian_ui.py` ont donc été supprimés (26 tests
retirés), ce qui achève un nettoyage déjà entamé — `gui/core/domain_sizing.py:160`
portait déjà la mention « Relocated from domain_jacobian (now removed) ».
Les docstrings de `domain_convergence.py` / `domain_convergence_worker.py`
conservent la justification du choix de la méthode par convergence mais ne
renvoient plus vers un module inexistant.
- Fichiers : `gui/tabs/optimization_tab.py` (aucune référence à
  `domain_jacobian`/`DomainJacobianWorker`/`_on_run_domain_jacobian`/
  `_on_domain_jacobian_done` — recherche exhaustive, zéro résultat) vs
  `gui/sensitivity/domain_jacobian.py`, `gui/sensitivity/domain_jacobian_worker.py`
  (classe `DomainJacobianWorker`, existe et est complète) et
  `tests/test_domain_jacobian_ui.py` (teste `OptimizationTab._on_run_domain_jacobian`
  et `._on_domain_jacobian_done` comme s'ils existaient).
- Statut : **fait**, reproduit par l'exécution réelle de la suite de
  tests (section "État des tests").
- Preuve : `git show 79b66ad --stat` (dernier commit de la branche) montre
  que `gui/tabs/optimization_tab.py` (+809/-… lignes) et
  `gui/sensitivity/domain_jacobian_worker.py` (nouveau fichier, 48 lignes)
  ont été modifiés/ajoutés dans le MÊME commit, mais le second n'est
  importé nulle part dans le premier. Le seul mécanisme de "domain sizing"
  câblé dans `OptimizationTab` est `_on_run_domain_convergence` /
  `DomainConvergenceWorker` (gui/tabs/optimization_tab.py:44, 272, 853,
  888) — une fonctionnalité voisine mais distincte.
- Comparer avec `docs/abaqus_validation_checklist.md` section 8, qui documente
  un "pending wiring" mais pour `run_domain_convergence` (ZOI), PAS pour
  domain_jacobian — donc ce n'est pas un manque déjà connu/documenté, c'est
  un point mort non signalé.
- Correction : soit câbler `OptimizationTab` (bouton + handlers, sur le
  modèle de `_on_run_domain_convergence`), soit — si la fonctionnalité est
  volontairement mise en pause — marquer `tests/test_domain_jacobian_ui.py`
  en `xfail`/`skip` explicite avec la raison, pour que la suite de tests
  reflète l'état réel du produit plutôt qu'une régression silencieuse à
  chaque exécution.

**M3 — Le Cancel bloque le thread GUI jusqu'à ~30 s, sur les DEUX chemins
d'annulation.** (constat ajouté en phase 2, non présent dans l'audit initial)
- Fichiers : `gui/sensitivity/run_worker.py:67-69` (la fonction bloquante),
  appelée depuis `gui/tabs/job_tab.py:745` et depuis
  `gui/sensitivity/run_worker.py:181` via
  `gui/tabs/sensitivity_tab.py:783-785`.
- Statut : **fait** pour le caractère bloquant et le thread d'exécution ;
  **calcul** (et non mesure) pour la borne de ~30 s, obtenue en sommant les
  timeouts écrits dans le code.
- Preuve : `abaqus_terminate_job` fait
  `subprocess.run(..., timeout=20.0)` — un appel synchrone. Il est atteint :
  (a) depuis `JobTab._cancel_run`, qui est le slot de `btn_cancel.clicked`
  (`job_tab.py:211`) donc s'exécute dans le thread GUI, suivi de
  `waitForFinished(10000)` puis, en repli, `waitForFinished(2000)` →
  20 + 10 + 2 = **~32 s** ;
  (b) depuis `SensitivityTab._on_cancel` (`sensitivity_tab.py:783-785`) qui
  appelle `self._worker.cancel()` en **appel de méthode direct**. Bien que
  `SensitivityRunWorker` vive dans un QThread (`moveToThread`), un appel
  direct s'exécute dans le thread de l'APPELANT, donc le thread GUI ici
  aussi : 20 s (`subprocess.run`) + 10 s (`p.wait(timeout=10.0)`,
  run_worker.py:186) + la boucle de grâce de `_terminate_process_tree`
  (2 s) → même ordre de grandeur.
- Conséquence : après un clic sur Cancel, la fenêtre ne se redessine plus et
  Windows peut afficher « ne répond pas », alors même que l'annulation se
  déroule correctement. L'utilisateur peut croire à un plantage et tuer la
  GUI — ce qui le ramène précisément au problème d'orphelins de M1.

**RÉVISION de l'ampleur (mesures réelles sur la machine de Tristan).**
Le « ~32 s » initial était la somme de tous les timeouts en supposant que
chacun atteigne son plafond. Deux mesures le rendent improbable :

| Commande | Durée mesurée |
|---|---|
| `abaqus.bat information=release` | **4,90 s** |
| `abaqus.bat terminate job=<job inexistant>` | **0,63 s** |

La seconde **n'est pas** le coût d'un vrai terminate (aucun job ne tournait,
la commande a échoué vite), mais elle établit que `abaqus.bat` peut démarrer
et rendre la main en 0,63 s. Les 4,90 s de `information=release` sont donc le
coût PROPRE de cette commande, pas un surcoût de lanceur : le plancher que
j'avais supposé à ~5 s ne tient pas, et mon étape de mesure était un mauvais
proxy.

**SECONDE RÉVISION — le `waitForFinished` n'est pas non plus le coupable.**
Le run `cancel_test` du 15/09/2026 le montre. Chronologie tirée du `.log` et
du `.sta` :

| Horodatage | Événement |
|---|---|
| 18:07:48 | `explicit_dp.exe` démarre |
| 18:07:54 | `Terminate request received from 2019-0357 on CL-CHENEVEZ-01` |
| 18:07:54 | `job aborted` — **même seconde** |

Le `.sta` s'arrête sur `***ERROR: Process terminated by external request`
après la frame 39/500. Le solveur meurt donc **en moins d'une seconde** après
réception de la demande. `waitForFinished(10000)` rend la main presque
immédiatement : il n'atteindra jamais son plafond de 10 s dans ce scénario.

Il ne reste donc qu'un seul terme au gel : la durée du sous-processus
`abaqus.bat terminate` lui-même — toujours non mesurée contre un job vivant
(la mesure à 0,63 s portait sur un job inexistant). **Ampleur probable de M3 :
quelques secondes, pas quelques dizaines.** Le mécanisme (appel bloquant sur
le thread GUI) reste établi ; son coût réel est vraisemblablement modeste.
À confirmer par l'observation directe du gel, seule donnée encore manquante.

**Au passage, ce run VALIDE la route propre d'annulation** : `abaqus
terminate job=` a bien été reçu par le solveur, qui s'est arrêté proprement
et a libéré ses 8 jetons de licence. C'est exactement le comportement que
Tristan décrivait comme le bon.

**CLOS SANS CORRECTION.** Décision de Tristan : « la commande terminate
s'exécute vite en effet donc ça me semble une bonne solution pour cancel un
job en cours, notamment lorsqu'un pipeline entier doit être cancel ». Le
mécanisme (appel bloquant sur le thread GUI) reste réel et documenté
ci-dessus, mais son coût est jugé acceptable en usage. Réserve d'honnêteté :
la durée exacte du `terminate` contre un job VIVANT n'a jamais été
chronométrée — le 0,63 s mesuré portait sur un job inexistant (chemin
d'erreur). Ce qui est établi, c'est que le solveur meurt dans la seconde
(log du run `cancel_test`), donc que le `waitForFinished(10000)` n'est pas le
terme dominant. Rien à corriger tant que l'usage ne remonte pas de gêne.
- Corrections possibles (alternatives, à arbitrer) :
  (a) **Minimal** : réduire les timeouts (ex. 20 s → 5 s pour
  `abaqus terminate`, qui rend la main en général en moins d'une seconde
  puisqu'il se contente d'écrire un fichier de signal). Ne supprime pas le
  gel, le raccourcit.
  (b) **Correct mais plus invasif** : exécuter `abaqus terminate` dans un
  QThread / `QProcess` asynchrone et faire du Cancel une machine à états
  (bouton passe en « Cancelling… », le repli est armé par un `QTimer`
  plutôt que par un `waitForFinished`). Supprime réellement le gel, mais
  change la structure des deux chemins d'annulation.
  (c) Faire émettre à `SensitivityTab._on_cancel` un signal vers le worker
  au lieu de l'appel direct (corrige uniquement le chemin (b) du constat,
  pas celui de l'onglet Job).

**M4 — `MASSEUL`/`VOLEUL` sont absents de l'ODB : le contrôle de conservation
de masse eulérienne n'existe pas dans les résultats.**
- Fichier : `abaqus_scripts/cel_model.py:615-624`.
- Statut : **fait** pour l'absence (constatée sur un ODB réel) ; **hypothèse**
  pour la cause.
- Preuve : `check_api.py v2 --odb C:\TEMP\ABQ_wd\GCI_run000.odb` liste
  l'intégralité des régions d'historique du step `Cut` (501 frames) :

  ```
  region 'Assembly ASSEMBLY': ['ALLAE','ALLCD','ALLDMD','ALLFD','ALLHF',
                               'ALLIE','ALLIHE','ALLKE','ALLPD','ALLSE',
                               'ALLVD','ALLWK','ETOTAL']
  region 'Node ASSEMBLY.1'  : ['RF1_SENSORBAND','RF2_SENSORBAND']
  ```

  Deux régions, aucune ne porte `MASSEUL` ni `VOLEUL`. `ALLKE`/`ALLIE`
  (garde-fou énergétique) et `RF1`/`RF2` (efforts de coupe) sont bien là :
  H-Output-1 et H-Output-2 fonctionnent, seul **H-Output-3 ne produit rien**.
- **L'âge de l'ODB n'explique pas l'absence.** Vérifié par `git log -S` :
  la requête `MASSEUL`/`VOLEUL` a été introduite en `e9e967f`, et son code est
  identique (au `COORD` près, ajouté plus tard) entre `e9e967f` et HEAD. Or
  cet ODB porte les sorties filtrées (`V_CAMERABAND`, `RF1_SENSORBAND`)
  introduites par ce même commit `e9e967f`, et pas `COORD` (ajouté en
  `79b66ad`) : il a donc été produit dans l'intervalle, par un code qui
  **contenait déjà** la requête.
- Conséquence : le projet croit disposer d'un indicateur de conservation
  (« is material leaving the domain, or being lost numerically? »,
  cel_model.py:610-614). Il n'en dispose pas. Ce n'est pas une erreur de
  physique, mais un garde-fou de diagnostic silencieusement inopérant — et
  le `try/except` qui l'entoure garantit que personne ne le remarque, la
  mise en garde partant sur stdout au milieu d'un log de run.
- Causes possibles (HYPOTHÈSES — je n'ai pas de quoi trancher, et je ne
  veux pas inventer la bonne forme de l'appel) :
  (a) la requête lève à la construction et le `try/except` l'avale — le
  suspect principal étant `region=assembly.sets['Euler']` : `MASSEUL`/`VOLEUL`
  sont des grandeurs par **instance de matériau eulérien**, et un set de
  cellules d'assemblage n'est peut-être pas une région recevable ;
  (b) la requête est acceptée à la construction mais Abaqus ne produit rien
  au solve.
- **CAUSE CONFIRMÉE — hypothèse (a).** Le `.dat` du job `cancel_test`
  (15/09/2026 18:07) le prouve sans ambiguïté :
  - `grep -i "masseul\|voleul"` sur le `.dat` → **aucune occurrence** ;
  - le deck ne contient que **deux** blocs `*output, history` dans le step :
    `*output, history, filter=SENSORBAND` (avec `*nodeoutput,
    nset=ASSEMBLY_RP` → RF1/RF2) et `*output, history, variable=PRESELECT`.

  H-Output-3 n'a donc jamais été écrit dans l'input deck : l'appel
  `model.HistoryOutputRequest(...)` de cel_model.py:616-620 **lève à la
  construction du modèle**, et le `try/except` de la ligne 621 l'avale. Le
  solveur n'est pas en cause, il n'a jamais reçu la demande.
- **Ce qui manque encore pour corriger** : le message d'exception exact. Il a
  été imprimé pendant ce run-là sous la forme
  `[WARNING] MASSEUL/VOLEUL history not created: <message>` dans le panneau
  de sortie de l'onglet Job. Le récupérer coûte quelques secondes : un
  **Write .inp only** (construction seule, pas de solveur) puis le bouton
  **Copy output**, et chercher `MASSEUL` dans le texte collé. Sans ce
  message je ne peux pas proposer la bonne forme d'appel sans l'inventer.
- Corrections possibles, à décider APRÈS ce test : corriger la région /
  la forme de la requête si (a) ; ou retirer la requête et le `try/except`
  si la conservation eulérienne ne s'obtient pas ainsi, plutôt que de
  garder un garde-fou qui n'en est pas un. Dans les deux cas, remplacer
  l'`except` muet par une trace que le pipeline remonte.

**M5 — Le filtre Butterworth de sortie de champ ne s'applique PAS à `TEMP`
ni à `EVF` : la garantie anti-repliement documentée ne vaut que pour `V`.**
- Fichiers : `abaqus_scripts/cel_model.py:111-127` (l'intention),
  `:551-585` (la requête filtrée), `cel_results.py:497` (les champs extraits).
- Statut : **fait** — Abaqus le dit lui-même, et l'ODB le confirme.
- Preuve n°1, le `.sta` du job `cancel_test` :

  ```
  ***WARNING: Nodal Output for coordinates and temperatures are not
              (digitally) filtered.
  ***WARNING: Element Output for Equivalent plastic strains, Status, ...,
              Coordinates, Temperatures and Field Variables are not
              (digitally) filtered.
  ```

- Preuve n°2, les clés de l'ODB : `V_CAMERABAND`, `U_CAMERABAND`,
  `UR_CAMERABAND`, `VR_CAMERABAND`, `ERV_CAMERABAND` portent le suffixe du
  filtre — mais `TEMP`, `TEMP_ASSEMBLY_EULER_EULER-1`,
  `EVF_ASSEMBLY_EULER_EULER-1` et `EVF_VOID` **ne le portent pas**.
- Le pipeline extrait exactement trois champs (`cel_results.py:497`) :
  `EVF`, `TEMP`, `V`. **Un seul des trois est effectivement filtré.**
- Pourquoi c'est un vrai constat et pas un détail : le commentaire de
  `create_step` (cel_model.py:551-558) justifie le filtre en disant qu'Abaqus
  filtre « at the SOLVER increment, BEFORE writing to the ODB -- the only
  stage where aliasing can still be prevented (once aliased data is written,
  no post-processing recovers it) ». Cette protection est réelle pour `V`.
  Elle est **inexistante pour `TEMP`**, alors que la température est
  précisément l'observable comparée aux mesures IRT.
- Nuance à ne pas écraser : pour `EVF` (fraction volumique, indicateur de
  matière), filtrer ne serait sans doute même pas souhaitable — un lissage
  de l'interface matière/vide n'a pas de sens physique. Le point porte
  surtout sur `TEMP`.
- Ce que je ne tranche pas : si l'absence de filtrage sur `TEMP` est
  acceptable dépend de la bande passante réelle de ta chaîne IRT et du taux
  d'échantillonnage (500 frames ici) — c'est ton arbitrage, pas le mien.
- **CORRIGÉ** — décision de Tristan : garder TOUJOURS les sorties non
  filtrées dans l'ODB, et ajouter les filtrées par-dessus quand un filtre
  est demandé ; la comparaison filtré/brut se fait à la main dans
  Abaqus/Viewer, et l'extraction prend la série filtrée si elle existe.
  Deux changements, indissociables :
  1. `cel_model.create_step` émet désormais **deux** requêtes de champ :
     `F-Output-1` (toutes les variables, jamais filtrée, toujours émise) et
     `F-Output-Filtered` (uniquement `_FILTERABLE_FO_VARIABLES = ('V','ERV')`,
     avec le filtre, seulement si un cutoff est réglé). Même forme pour
     l'historique : `H-Output-1` brut toujours émis, `H-Output-1-Filtered`
     en plus. Le code ne prétend donc plus filtrer ce qu'Abaqus ne filtre
     pas, et les avertissements « not (digitally) filtered » doivent
     disparaître du `.sta`.
  2. **Conséquence non évidente qu'il fallait traiter en même temps** :
     `_resolve_fo_name` et `_find_history_key` testaient le nom EXACT en
     premier. Tant que `V` nu n'existait pas, ils tombaient sur
     `V_CAMERABAND`. Dès que la série brute coexiste, ils auraient choisi la
     série NON filtrée — l'inverse de l'intention, et en silence. Les deux
     fonctions prennent maintenant un `filter_suffix` et préfèrent la série
     filtrée quand le modèle en a demandé une, avec repli sur le nom nu.
     `extract_results` dérive ce suffixe de `model_cfg`, en miroir exact des
     conditions sous lesquelles `create_step` crée chaque filtre.
  4 tests de non-régression ajoutés (`tests/test_cel_results_history.py`),
  dont celui qui garde précisément la bascule silencieuse ci-dessus.
- **Reste à vérifier côté Abaqus** (je ne peux pas le faire d'ici) : un
  `Write .inp only` doit montrer **deux** blocs `*output, field` et deux
  `*output, history` sur le RP ; et un run court doit faire coexister `V` et
  `V_CAMERABAND` dans l'ODB. C'est une HYPOTHÈSE tant que ce n'est pas
  constaté : je n'ai aucune preuve qu'Abaqus accepte deux requêtes portant
  la même variable, l'une filtrée et l'autre non.
- Ce que cette correction NE tranche pas : savoir si `TEMP` devrait être
  band-limitée physiquement. Elle rend seulement l'ODB complet et le code
  honnête ; l'arbitrage sur la bande passante IRT reste ouvert.

### Mineur

**m6 — `COORD` est demandé alors qu'il n'existe pas pour `EC3D8RT`.**
- Fichier : `abaqus_scripts/cel_model.py:92-104` (ajouté en `79b66ad`).
- Statut : **fait**.
- Preuve, `.dat` :
  `***WARNING: OUTPUT REQUEST COORD IS NOT AVAILABLE FOR ELEMENT TYPE EC3D8RT`.
  `EC3D8RT` est le type d'élément de TOUT le domaine eulérien, c'est-à-dire
  de la pièce. La demande est donc sans effet là où elle aurait servi, et
  génère un avertissement à chaque run.
- À noter, le même `.dat` porte deux messages voisins mais LÉGITIMES, qu'il
  ne faut pas confondre avec celui-ci : `EVF IS NOT AVAILABLE FOR ELEMENT
  TYPE C3D8RT` (EVF n'a pas de sens sur l'outil lagrangien) et les `NOTE`
  sur `DMICRT`/`SDEG` (pas de modèle d'endommagement actif — cohérent avec
  `JohnsonCookDamageInitiation` commenté en cel_model.py:343-347). Ceux-là
  sont le prix normal d'une liste de variables unique pour deux corps.
- **CONSERVÉ sur décision de Tristan** : `COORD` sert à l'inspection manuelle
  de l'ODB — construire des display groups à partir des coordonnées de
  nœuds dans Abaqus/Viewer. Le retirer coûterait cet usage.
- **Nuance qui change la lecture de l'avertissement** : il porte sur
  `ELEMENT TYPE EC3D8RT`, donc sur la variante aux POINTS D'INTÉGRATION. Le
  `.sta` parle par ailleurs de « **Nodal** Output for coordinates », ce qui
  implique que la sortie NODALE de `COORD` existe — et c'est celle dont
  l'usage ci-dessus a besoin. **HYPOTHÈSE non vérifiée** : aucun ODB
  disponible ne permet de le confirmer (`COORD` a été ajouté en `79b66ad`,
  après l'ODB `GCI_run000` examiné ici).
- **CLOS — NON-PROBLÈME, vérifié par Tristan** : « COORD est bien dans
  `cancel_test.odb`, je peux donc faire des display groups pour inspecter
  les résultats. » L'hypothèse était la bonne : l'avertissement ne porte
  que sur la variante aux points d'intégration, la sortie NODALE arrive
  bien et sert son usage.
- **Aucune correction.** Retirer `COORD` supprimerait un usage réel ; lui
  dédier une requête scindée ajouterait de la complexité pour faire taire
  un avertissement sur une variante dont personne ne se sert. L'entrée est
  conservée ici pour documenter pourquoi cet avertissement du `.dat` est
  attendu et n'a pas à être « corrigé ».

**Note sans conséquence — `order=2` n'apparaît pas dans le deck.**
`ButterworthFilter(..., order=2)` (cel_model.py:563-570) produit
`*filter, name=CAMERABAND, type=BUTTERWORTH` sans paramètre d'ordre, et
Abaqus avertit : « NO VALUE WAS SPECIFIED FOR THE ORDER OF FILTER CAMERABAND.
A DEFAULT SECOND ORDER WILL BE USED. » Le résultat est donc **identique à
l'intention** (l'ordre 2 est le défaut), la couche CAE omettant simplement
les paramètres égaux au défaut. Consigné ici pour que personne ne « corrige »
un comportement qui est déjà le bon.

**Observation de configuration (pas un défaut de code) — BC eulérienne
écrasée par la BC de vitesse.** Le `.sta` porte :
`***WARNING: Both the *EULERIAN BOUNDARY, INFLOW=NONE option and the
*BOUNDARY option are specified at the same nodes. In case of conflict
*BOUNDARY will override the Eulerian boundary condition.` Sur cette
configuration, une face portait à la fois une `EulerianBC` et la BC de
vitesse de coupe ; c'est la seconde qui gagne. Cela dépend des cases cochées
dans l'onglet BCs, pas du code — mais rien dans la GUI ne le signale.

**m5 — INFIRMÉ. Mon hypothèse était fausse : le repli `dataDouble` est
indispensable, pas du code mort.**
- Fichier : `abaqus_scripts/cel_results.py:182-189`.
- Ce que j'avais avancé (sur la base d'une sonde v2 portant sur `CPRESS`,
  une sortie de contact que le pipeline n'extrait jamais) : `data`
  fonctionne, `dataDouble` est absent, donc le repli serait du code mort.
- **Ce que v3 mesure sur les variables réellement extraites, instance
  eulérienne :**

  ```
  EVF (EVF_ASSEMBLY_EULER_EULER-1): data=True   dataDouble=False
        v.data reads OK -> 1.0
  V   (V_CAMERABAND)              : data=False  dataDouble=True
  ```

  Pour `V`, **`data` est ABSENT et `dataDouble` est PRÉSENT**. Le
  `try: v.data / except: v.dataDouble` de `_read_data` est donc
  effectivement emprunté, et c'est la seule branche qui permet de lire la
  vitesse — le champ de comparaison avec la DIC. Sans ce repli, `V` ne
  serait pas extractible.
- La docstring dit vrai sur le fond, elle généralise seulement un peu trop
  (« `value.data` raises » vaut pour les champs NODAUX écrits en pleine
  précision — `nodalOutputPrecision=FULL`, cel_model.py:805 — et non pour
  les champs élémentaires comme EVF, où `data` fonctionne). Les deux cas
  sont correctement traités par le code tel qu'il est. **Aucune correction
  nécessaire.**
- Leçon pour cet audit : le constat initial venait d'une sonde non
  représentative, que j'avais signalée comme telle. La vérification l'a
  tranché contre moi — c'est le résultat attendu d'un tel processus.

**VALIDATION SUPPLÉMENTAIRE — `_resolve_fo_name` est nécessaire, preuve à
l'appui** (ce n'est pas un constat, c'est une confirmation que le code
existant est bien fondé).
- Fichier : `abaqus_scripts/cel_results.py:192-233`.
- v3 montre que pour `TEMP`, deux clés coexistent dans l'ODB :
  `['TEMP', 'TEMP_ASSEMBLY_EULER_EULER-1']`, et que la clé NUE `TEMP`
  **ne porte aucune valeur sur l'instance eulérienne** (elle porte celles
  de l'outil lagrangien). C'est exactement le cas décrit par la docstring
  de `_resolve_fo_name`.
- Conséquence : le test `_has_inst_values` appliqué aux correspondances
  exactes AVANT de se rabattre sur les noms suffixés n'est pas une
  précaution décorative. Une résolution naïve « nom exact d'abord »
  choisirait `TEMP` et extrairait un champ de température entièrement
  NaN pour la pièce, sans erreur visible.

**m1 — Duplication de la construction de la commande Abaqus entre
`JobTab._dry_run`/`_launch_abaqus` et `SensitivityRunWorker._abaqus_solve`.**
- Fichiers : `gui/tabs/job_tab.py:326-335` et `:521-529`, vs
  `gui/sensitivity/run_worker.py:223-226`.
- Statut : fait (même séquence `[cmd, "cae", f"noGUI={script}", "--",
  "--model_cfg", repr(...), "--run_cfg", repr(...)]` écrite deux fois,
  indépendamment).
- Risque : une évolution du contrat (ex. nouvel argument) faite dans un
  seul des deux endroits romprait silencieusement l'autre chemin de
  lancement — ce type de divergence s'est déjà produit sur le cancel (M1).
- Correction possible : extraire un seul `build_abaqus_args(abaqus_cmd,
  script, model_cfg, run_cfg)` partagé (ex. dans un module commun aux
  deux, `gui/core/` ou `gui/sensitivity/`).

**m2 — Code d'extraction mort dans le pipeline réel (`_TENSOR_REDUCERS`,
`_STRESS_INVARIANT`, réduction von Mises).**
- Fichier : `abaqus_scripts/cel_results.py:171-179, 236-326`.
- Statut : fait + interprétation.
- Preuve : `extract_results` fige `_field_vars = ["EVF", "TEMP", "V"]`
  (cel_results.py:497), documenté comme intentionnel
  ("the extraction pipeline only reads EVF, TEMP and V", cel_model.py:83-91).
  Or `_extract_field` gère aussi `"MISES"`/`"S_VM"`/`"S_P"` via
  `_TENSOR_REDUCERS`/`_STRESS_INVARIANT`/`_reduce_VM`, et ces chemins ne
  sont exercés par aucun appelant réel (seul `gui/results/fake_builder.py`,
  utilisé par les tests et le stub, produit du `S_VM` synthétique — grep
  exhaustif du dépôt). Le commentaire de `results_tab.py:372-375` ("the
  Eulerian [instance]... carries PEEQ/TEMP/MISES/EVF") est de ce fait
  légèrement trompeur pour un run réel : PEEQ et MISES ne quittent jamais
  l'ODB aujourd'hui.
- Ce n'est pas un bug fonctionnel (rien ne dépend de ce chemin en usage
  réel), mais une source de confusion pour la maintenance et un risque
  latent si quelqu'un active `S_VM`/`PEEQ` côté GUI en pensant que
  l'extraction suit.
- **CORRIGÉ** — Tristan : « je ne me sers en effet pas de ça, je regarde
  juste les contraintes dans l'ODB en mode inspection, c'est du code mort
  que tu peux nettoyer. » Supprimés : `_reduce_VM`, `_TENSOR_REDUCERS`,
  `_STRESS_INVARIANT`, et dans `_extract_field` la résolution d'invariant
  et l'appel `getScalarField`.
- **`'S'` reste demandé dans `fo_variables`** (cel_model.py:111) : c'est
  précisément ce qui alimente l'inspection manuelle des contraintes dans
  l'ODB. Seul le code d'EXTRACTION vers le `.npz` était mort, pas la sortie
  elle-même. Le chemin réellement emprunté (`EVF`, `TEMP` via
  `_reduce_identity`) est inchangé.

**m3 — `try/except Exception: pass` autour d'une assignation qui ne peut
pas échouer.**
- Fichier : `gui/sensitivity/mass_scaling.py:289-293`.
- Statut : fait.
- Preuve : `cfg.step.output.ho_preselect = True` est entouré d'un
  `try/except Exception: pass`, alors que `OutputCfg.ho_preselect` est un
  champ de dataclass déclaré avec une valeur par défaut `True`
  (`gui/core/model_config.py:109`) — l'assignation ne peut pas lever sauf
  si `cfg.step.output` n'existe pas, ce qui n'arrive jamais avec
  `ModelConfig()` standard. Inoffensif (et le défaut est déjà `True`), mais
  un `except` sans commentaire sur CE qu'il est censé absorber est
  difficile à auditer plus tard.
- Correction : soit retirer le `try/except`, soit expliciter en commentaire
  quel scénario précis il couvre.

**m4 — Suite de tests non tolérante à l'absence d'`imageio`, une
dépendance volontairement optionnelle côté production.**
- Fichiers : `tests/test_experimental.py` (imports directs `import
  imageio.v3 as iio` sans garde), vs `requirements.txt` qui ne liste PAS
  `imageio` (contrairement à Pillow/opencv/scipy) et
  `gui/core/sequence_io.py` qui implémente un repli explicite
  Pillow → imageio → matplotlib précisément pour fonctionner SANS
  imageio (TODO.md:350-353 : "Dépendance imageio retirée du chemin
  critique").
- Statut : fait, reproduit par l'exécution réelle
  (`ModuleNotFoundError: No module named 'imageio'`, 3 tests de
  `test_experimental.py`).
- Ce n'est pas un défaut du produit — c'est un défaut de robustesse de la
  suite de tests vis-à-vis d'un environnement conforme à
  `requirements.txt`. Correction : `pytest.importorskip("imageio")` en
  tête des tests concernés (ou dans `tests/conftest.py`).

### Style

Rien de notable au-delà des points ci-dessus. Le style du code Abaqus
(`cel_model.py`, `cel_results.py`) est homogène, commenté sur le POURQUOI
plutôt que le QUOI, et les fonctions sont bien découpées par responsabilité
(`create_parts`, `create_materials`, `create_mesh`, ...), ce qui a
considérablement facilité cette revue.

## Méthodes Abaqus contournées ou inventées

**Aucune méthode contournée ou inventée n'a été identifiée.** En particulier
sur les points que la mission demandait de vérifier spécifiquement :

- **Édition de mots-clés `.inp`** : le projet n'édite JAMAIS le texte du
  `.inp` et n'utilise jamais `keywordBlock` (recherche exhaustive du dépôt,
  zéro résultat). Le modèle est construit intégralement via l'API
  Scripting (`mdb.Model`, `model.Part`, etc.) puis soit écrit directement
  (`job.writeInput`), soit soumis (`job.submit`) — il n'y a à aucun moment
  de réimport du `.inp` dans CAE. Le risque historique documenté par la
  mission ("la réimportation du .inp dans CAE perd les Section Controls
  (secondOrderAccuracy)") **ne peut donc pas se produire dans ce pipeline**,
  puisque `secondOrderAccuracy=OFF` est fixé directement sur l'`ElemType`
  au moment de la création du maillage (`cel_model.py:403, 436`) et n'est
  jamais reperdu par un aller-retour CAE. C'est, à la lecture, la méthode
  la plus directe et la plus robuste possible pour ce risque précis.
- **Suivi de succès du job** : `_check_job_succeeded` (cel_model.py:810-849)
  explique explicitement, en commentaire, pourquoi `job.status` /
  `job.messages` sont écartés au profit d'une lecture du `.sta` — "Job
  messages are not returned if a script is run without the Abaqus/CAE GUI"
  et `job.status` documenté `NONE` dans ce cas. C'est un contournement
  DOCUMENTÉ et justifié d'une limitation connue de l'API en mode `noGUI`,
  pas un raccourci arbitraire — mais son fondement ("documented as NONE")
  n'a pas pu être confirmé contre la doc Abaqus dans cet environnement
  (NON VÉRIFIÉ, à confirmer via `_review/check_api.py` + doc Abaqus).
- **Annulation** : `abaqus terminate job=<name>` est utilisé en premier
  (route documentée pour libérer les jetons de licence), avec repli sur
  kill process — c'est l'ordre attendu. Voir cependant M1 : le repli n'est
  pas UNIFORME entre les deux implémentations du dépôt.

## Constat transversal — les avertissements `.dat`/`.sta` ne sont lus par personne

M4, M5 et m6 ont tous les trois été trouvés en lisant les avertissements
qu'Abaqus écrit lui-même dans le `.dat` et le `.sta` d'un run ordinaire. Le
pipeline **conserve** délibérément ces deux fichiers
(`_DIAGNOSTIC_EXTENSIONS`, cel_model.py:876-877, avec un commentaire qui dit
qu'ils sont « the files that actually let you find out WHY a run
misbehaved »), mais **aucun code ne les lit jamais**, et la GUI ne les
affiche nulle part.

Conséquence mesurable : trois défauts — dont un garde-fou de conservation
inopérant et une protection anti-repliement absente sur la température —
étaient annoncés noir sur blanc par Abaqus à chaque exécution depuis
`e9e967f`, sans que rien ne les remonte.

Piste (c'est une fonctionnalité, donc hors du périmètre « fiabiliser sans
ajouter » de cette mission — à toi de dire si tu la veux) : après un run,
balayer `<job>.dat` et `<job>.sta` pour les lignes `***WARNING` / `***ERROR`
et les afficher dans l'onglet Job ou Results. Une quinzaine de lignes de
code auraient fait remonter M4, M5 et m6 automatiquement, le jour même.

## Questions pour Tristan

1. ~~Exécuter `check_api.py`~~ — **CLOS**. v1, v2 et v3 exécutés.
   L'inventaire API est vérifié (0 MISSING, 0 ERROR en v3), m5 est infirmé,
   et M4 est confirmé. Plus rien à demander de ce côté.
2bis. **M4 — le seul test qui reste, et il est gratuit** : onglet Job →
   **Write .inp only**, puis chercher `MASSEUL` dans le `.inp` produit.
   Présent → Abaqus a accepté la requête et l'a abandonnée au solve ;
   absent → la requête lève à la construction, et le log du run porte alors
   `[WARNING] MASSEUL/VOLEUL history not created:` suivi du message d'Abaqus,
   qui nomme la cause exacte. Sans cette information je ne peux pas proposer
   de correction sans inventer.
2. ~~Version du Python embarqué~~ — **répondu** : **2.7.15** (MSC v.1928,
   64 bit). La contrainte 2.7 de `cel_common.py` est donc justifiée, ce
   n'est plus une hypothèse.
3. M1 (cancel process tree) : confirmes-tu que le Cancel depuis l'onglet
   Job a déjà laissé un `standard.exe`/`explicit.exe` orphelin dans le
   Gestionnaire des tâches, ou est-ce un risque théorique jamais observé en
   pratique chez toi ?
4. ~~M2 (domain_jacobian non câblé)~~ — **répondu** : abandonné, aucun
   câblage envisagé. Code supprimé (commit `70b43c0`).
5. ~~Cancel : `taskkill /F /T` est-il la spécification voulue ?~~ —
   **répondu** : la meilleure route est `abaqus terminate job=<name>`
   exécutée dans le dossier de travail du job. C'était déjà l'étape 1 des
   deux implémentations ; seul le repli a été corrigé (M1, commit `7a7631c`),
   car `abaqus terminate` ne peut répondre qu'une fois le `<job>.cid` écrit
   par le solveur — un Cancel pendant la construction du modèle ou pendant
   l'extraction n'a pas d'autre recours.
6. **M3 (gel de l'UI pendant le Cancel)** : quelle option préfères-tu ?
   (a) réduire simplement les timeouts (correction de quelques lignes, le
   gel passe de ~30 s à ~7 s mais ne disparaît pas) ; (b) rendre le Cancel
   asynchrone (supprime le gel, mais restructure les deux chemins
   d'annulation) ; (c) ne rien faire si un gel de quelques secondes au
   Cancel ne te gêne pas en pratique. Je n'ai pas tranché seul : c'est un
   compromis ergonomie / risque de régression sur un chemin que je ne peux
   pas tester sous Windows.
7. Reste-t-il des constats mineurs (m1 à m4) que tu veux voir corrigés dans
   cette passe ? m4 (`pytest.importorskip("imageio")`) est le moins risqué :
   3 lignes, il rend la suite verte dans un environnement conforme à
   `requirements.txt`.

## État des tests

Exécutés réellement dans cet environnement (Linux, headless,
`QT_QPA_PLATFORM=offscreen`), après création d'un venv dédié
(`.venv_review/`, non versionné) et `pip install -r requirements.txt`, plus
installation des bibliothèques système Qt manquantes (`libegl1`,
`libegl-mesa0`, `libxcb-cursor0`, `libxcb-image0`, `libxcb-render-util0`,
`libxcb-util1` — absentes de l'image de base, sans rapport avec le code du
projet).

**Passe 1 — toute la suite sauf `test_mesh_pipeline.py`** :
```
577 passed, 9 failed in 213.37s (0:03:33)
```
Détail des 9 échecs :
- 6× `tests/test_domain_jacobian_ui.py` — `AttributeError:
  'OptimizationTab' object has no attribute '_on_run_domain_jacobian'` (ou
  `_on_domain_jacobian_done`) → constat M2 ci-dessus.
- 3× `tests/test_experimental.py` (`test_image_sequence_folder_standard_formats`,
  `test_image_sequence_folder_jpeg`, `test_image_sequence_dir_natural_sort`)
  — `ModuleNotFoundError: No module named 'imageio'` → constat m4 ci-dessus.

**Passe 2 — `test_mesh_pipeline.py` seul** :
```
11 passed in 209.82s (0:03:29)
```

**Total réel : 588 réussis / 9 échoués sur 597 tests collectés.**

### Après les corrections de phase 2 (M1 + M2)

Mêmes conditions (venv `requirements.txt`, headless, deux passes).

**Passe 1 — toute la suite sauf `test_mesh_pipeline.py`** :
```
3 failed, 561 passed in 189.06s (0:03:09)
```
Les 3 échecs restants sont ceux d'`imageio` (constat m4, non corrigé à ce
stade). Les 6 échecs `AttributeError` de `test_domain_jacobian_ui.py` ont
disparu avec la suppression de la fonctionnalité.

**Passe 2 — `test_mesh_pipeline.py` seul** :
```
11 passed in 212.08s (0:03:32)
```

**Total après phase 2 : 572 réussis / 3 échoués sur 575 tests collectés.**

### Après les correctifs m1 / m3 / m4

```
567 passed, 3 skipped in 138.55s   (passe 1, hors test_mesh_pipeline)
11 passed in 172.87s               (passe 2, test_mesh_pipeline seul)
```

**578 réussis, 3 ignorés, ZÉRO échec.** La suite est verte pour la première
fois de cette revue : les 3 « échecs » restants étaient les tests `imageio`
de m4, qui sont désormais correctement IGNORÉS dans un environnement conforme
à `requirements.txt` au lieu d'échouer. (+3 tests par rapport au relevé
précédent : ceux ajoutés pour `build_abaqus_args`.)

Réconciliation du nombre de tests (pour vérifier qu'aucun test n'a été perdu
silencieusement) : 586 collectés en passe 1 avant, moins 26 tests supprimés
avec la fonctionnalité abandonnée (19 dans `test_domain_jacobian.py` + 7 dans
`test_domain_jacobian_ui.py`, comptés sur les fichiers via `git show`), plus
4 tests de non-régression ajoutés pour M1 = 564 collectés, ce qui correspond
exactement aux 561 + 3 observés.

Zones critiques non couvertes par la suite automatisée (le projet le
documente déjà en grande partie dans `docs/abaqus_validation_checklist.md`) :
tout ce qui nécessite un Abaqus réel — construction de modèle, solveur,
lecture ODB réelle, `JobTab._cancel_run` en conditions réelles Windows (le
test `tests/test_abaqus_terminate.py` couvre les fonctions unitaires
`abaqus_terminate_job`/`_terminate_process_tree`, pas le code de
`JobTab._cancel_run` lui-même — voir M1), et tout le chemin
`_review/check_api.py` par construction.
