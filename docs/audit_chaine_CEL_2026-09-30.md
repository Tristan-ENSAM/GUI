# Audit de la chaîne simulation / post-traitement CEL — 2026-09-30

Portée : dépôt `Tristan-ENSAM/GUI`, commit `8b91a85` (branche de travail
`claude/compassionate-babbage-omd35o`). Audit en lecture seule : aucun fichier
de code n'a été modifié. Seul ce rapport a été ajouté.

## 0. Méthode et niveau de preuve

Chaque constat porte un code de vérification :

| Code | Signification |
|---|---|
| **L** | lecture du code (chaîne suivie : commande UI → validation → config → moteur → sortie) |
| **T** | couvert par la suite de tests existante, **exécutée** ici : `714 passed, 4 skipped` (Python 3.11, PySide6 6.11.1, offscreen, 6 min 35 s) |
| **X** | script ciblé exécuté pendant l'audit (hors dépôt, reproductible, voir annexe A) |
| **U** | interface instanciée en offscreen (`MainWindow`) : onglets, boutons, cases relevés |
| **NV** | non vérifiable ici : Abaqus, licences, ODB réels et données expérimentales sont absents de l'environnement |

Catégories de défaut : **ABS** fonctionnalité absente · **NR** moteur existant
non raccordé à l'interface · **CALC** erreur de calcul ou de définition ·
**DATA** problème de données · **NV** élément seulement non vérifié.

Hypothèse de lecture (interprétation, pas un fait) : la campagne de 13 runs
correspond au schéma **central** sur k = 6 paramètres (2k+1 = 13,
`jacobian_plan.n_runs`). Rien dans le dépôt ne contient ses résultats. Les
constats qui la concernent décrivent donc ce que le code a produit, pas des
valeurs relues.

---

## 1. Diagnostic

### Réalisable immédiatement (avec les réserves indiquées)

- **Relancer ou exploiter les cartes de sensibilité signées** : les CSV
  `sensitivity_maps/map_<champ>_pNN_<param>.csv` contiennent la dérivée
  centrale signée par élément et par frame. Réserves : champ `V` = norme
  uniquement, pas de masque EVF, toutes les frames sont incluses (t = 0 compris).
- **Sensibilités scalaires des efforts** (`Fx_mean`, `Fy_mean`). Réserve de
  définition : moyenne de |RF| sur tout l'historique, transitoire d'entrée
  compris, en N pour une tranche d'épaisseur `elem_size` (§2, A3).
- **Étude GCI du maillage eulérien** et **dimensionnement du domaine par
  convergence** depuis *Optimization › Model*. Réserves bloquantes pour la
  traçabilité : les deux études **modifient le profil actif** (A1), leurs
  résultats ne sont écrits **que dans le panneau de log** (A6), et les
  tolérances relatives de 2 % sont des valeurs par défaut sans provenance.
- **Mesure DIC** (moteurs local ZNCC et global Q4) et export `.npz/.json`.

### Bloqué

| Tâche du planning | Blocage | Nature |
|---|---|---|
| Définir ε_q à partir des incertitudes DIC | aucun outil de statistiques d'erreur (biais, σ, RMSE, cartes), onglet *Noise* vide, incertitude analytique désactivée | ABS + NR |
| Tolérances ε_q en température et en efforts | pas de données IRT ni de bruit d'efforts ; onglets *IRT*, *Forces*, *Calib. thermal* et *Noise* vides | DATA + ABS |
| Identification du mass scaling depuis l'interface | moteur `mass_scaling.py` présent mais non raccordé ; seul un facteur manuel est disponible (onglet *Step*) | NR |
| Robustesse χ⁻/χ₀/χ⁺ et calcul de Φ_L | ni états χ, ni réutilisation de runs, ni Φ_L exporté, ni tableau χ → Φ_L → L** | ABS |
| σ_Φ, hystérésis, journal de décisions | rien d'implémenté | ABS |
| Boucle identification–dimensionnement | onglet « Inverse identification — coming later » ; aucun solveur ; pas de données cibles T et efforts | ABS + DATA |

---

## 2. Anomalies susceptibles de fausser les conclusions scientifiques

Classées par gravité. Chaque ligne donne une preuve vérifiable.

**A1 — Les études de dimensionnement modifient le profil actif (X, L).**
`optimization_tab.py:936` et `:1020` passent `base_cfg=self.cfg` (le profil
affiché) aux workers. `domain_convergence.py:349` écrit
`cfg.euler_geometry.*` et `mesh_gci.py:337` écrit `base_cfg.elem_size` avant
chaque run, depuis un thread de travail. Vérifié (annexe A, `v_mutation`) :
après un GCI interrompu, `elem_size` passe de 0.005 à 0.0025 ; après une
convergence de domaine, `h_wp` et `l_wp` passent de 0.2 à 0.06. Le profil
n'est pas marqué modifié et l'onglet *Geometry* n'est pas rafraîchi.
Conséquence : une campagne de sensibilité lancée ensuite tourne sur un L
différent de L₀** **sans aucun signal**. Le profil sauvegardé peut aussi
contenir le dernier L essayé au lieu du L retenu.

**A2 — L'indicateur `[field]` n'est pas comparable entre paramètres,
maillages ou fenêtres (X, L).** `runner_core.py:87` n'utilise que le run +δ,
même en schéma central. La valeur vaut `SSD(F+, F0)/δ²` (`field_metrics.py:83`).
Vérifié : pour la même réponse, `[field]` passe de 5.556e4 à 8.889e5 quand
(éléments × frames) passe de 50 à 800, soit un facteur 16. Le run −δ est
ignoré : pour une réponse asymétrique (+1 / −5), `[field]` ne voit que +1,
alors que la carte centrale vaut 100 = 6/(2δ). L'unité est
(unité du champ)²/(unité du paramètre)², sans normalisation possible. Le
graphique *Chart* classe pourtant les paramètres par `|sensitivity|` sur
cette colonne. Tout classement A/B/n/C/m/μ fondé sur `[field]` est donc
invalide.

**A3 — L'« effort moyen » cumule plusieurs choix implicites (L).**
`qoi.py:122` calcule `nanmean(|RF1_RP|)` : c'est la **moyenne des valeurs
absolues**. `warmup_frac` n'est jamais transmis (`sensitivity_tab.py:844`, la
valeur par défaut 0.0 s'applique), donc l'entrée de l'outil est incluse.
L'historique n'a que `n_frames` intervalles (`cel_model.py:123`) et il est
filtré (série `SENSORBAND`) si le filtre est actif. La force est en N pour
une tranche d'épaisseur `elem_size` (`cel_model.py:298/304/329`, extrusion
`depth=elem_size`). Les autres modules définissent l'effort autrement :

- convergence de domaine : moyenne **signée** de RF1 seul, sur la fenêtre
  30–100 %, sans normalisation ;
- GCI : moyenne signée sur la fenêtre, divisée par h (N/mm), pour RF1 et RF2.

Trois définitions différentes circulent donc pour « Fc ».

**A4 — `V` désigne la norme du vecteur moyen, et la sensibilité porte sur la
variation de cette norme (X, L).** L'extraction moyenne les composantes nodales
signées sur chaque élément, puis calcule `V = sqrt(V1² + V2²)`
(`cel_results.py:379`). L'onglet *Sensitivity* ne propose que `EVF`, `V` et
`TEMP` (`sensitivity_tab.py:203`, relevé U) : V1 et V2 sont dans le bundle
mais ne sont pas sélectionnables. Toutes les colonnes et cartes « V »
calculent donc d|V|, et non |dV|. Vérifié : une rotation de 90° à norme
constante donne d|V| = 0 alors que |dV| = 141 mm/s.

**A5 — Pas de masque EVF ni de fenêtre temporelle dans la sensibilité de
champ (L).** `jacobian_field_analysis`, `jacobian_field_maps` et
`map_export.time_aggregates` prennent toutes les frames, t = 0 compris, et
tous les éléments ROI, vide compris. De plus, `field_metrics._align`
(`:20`) tronque **silencieusement** deux champs de formes différentes, sans
contrôler l'identité des centroïdes.

**A6 — Résultats GCI et domaine non persistés, critère ε_q affiché mais
inutilisé (U, L).**
- `_on_mesh_done` et `_on_dc_done` ne font qu'écrire dans le log : aucun
  fichier de résultats.
- Le groupe « 2 · Convergence criterion (RMSE thresholds) » (`_q_eps`) est
  sauvegardé dans le profil mais n'est lu que par `_e_max` et `_plot`, deux
  fonctions jamais appelées (`_hist` n'est jamais rempli) : l'onglet
  *Convergence* reste vide.
- Les tolérances effectivement appliquées sont les « Sizing tolerances
  (relative) » (0.02 par défaut).
- Le bouton « Compute initial domain » et les « max cap » ne servent qu'à
  l'aperçu : l'étude part de `_dims_from_cfg()` (`:906`).

**A7 — Métriques E_q hétérogènes et non conformes à la comparaison sans
interpolation.**
- Convergence de domaine : norme relative `‖a−b‖/‖b‖` de moyennes temporelles
  masquées EVF, échantillonnées au **plus proche voisin**. Vérifié (X) : si la
  grille ZOI est alignée sur les nœuds, **100 % des points (289/289) sont
  équidistants de 4 centroïdes**. Le choix de la cellule dépend alors du
  départage de `cKDTree` : il est identique dans le cas testé, mais rien ne le
  garantit.
- GCI : **interpolation barycentrique** (Delaunay), puis réduction à **une
  moyenne spatiale** par grandeur. Des erreurs locales de signes opposés
  peuvent se compenser.
- `domain_opt.py` : correspondance exacte par indices de cellule
  (`grid_keys`, `align_to_reference`) et écart moyen absolu. C'est
  précisément la comparaison sans interpolation recherchée, mais ce moteur
  **n'est pas raccordé**.

**A8 — Aucune vérification que la ZOI est incluse dans la ROI extraite (X).**
Le bundle ne contient que les éléments de la ROI (`cel_results.py`, filtre
`_in_bbox`). Vérifié : un point ZOI situé à 0.2 mm est échantillonné sur
l'élément de bord de la ROI, à 0.153 mm de distance, sans avertissement.

**A9 — Plafond de domaine figé (L).** `domain_sizing.py:168` fixe
`_DIAGONAL_OVER_ELEM_MAX = 90.6`. Cette constante n'est pas recalculée à
partir du matériau ni de la coupure du filtre, contrairement à
`ModelConfig.mass_scaling_bounds`. Sa provenance n'est pas documentée dans le
code : **valeur non sourcée**. Elle arrête la convergence de domaine
(`stopped_by="diagonal"`).

**A10 — Aucune séparation χ / L dans l'onglet Sensitivity (L).** La catégorie
« Numérique » reste sélectionnable : seul `elem_size` est exclu
(`sensitivity_tab.py:60`). `step.mass_scaling_factor`
(`param_registry.py:253`) peut donc être perturbé comme un paramètre
physique. Aucune règle n'empêche de confondre variation physique et erreur
numérique.

---

## 3. Matrice

Statuts : **présent**, **partiel**, **absent**, **non vérifiable**.

### 3.1 Incertitudes DIC et tolérances

| Tâche | Capacité attendue | Statut | Accès UI | Moteur | Preuves | Vérif. | Limite / blocage | Correction minimale |
|---|---|---|---|---|---|---|---|---|
| DIC | ROI fixe, maillage, corrélation, sous-pixel, rejets, échelle, fps | présent | Experimental Data › DIC (Select/Validate ROI, Parameters…, résolution, fps) | `dic.DicParams`, `dic_global.DicGlobalParams` | `dic.py:44`, `dic_global.py:70` ; ZNCC min, bord de recherche, texture, saturation, couverture | L T U | `fps = 0` donne `dt = 1` en silence (`dic.py:599`) | refuser fps ≤ 0 |
| DIC | Cas à déplacement nul ou connu (0.25–2 px, x et y) | partiel | chargement d'une séquence de dossier possible ; aucune consigne ni comparaison | générateurs `fshift`, `speckle` dans les **tests** | `tests/test_dic_local_validation.py`, `test_dic_global_audit.py:61-92` | T | uniquement en tests unitaires, aucun outil utilisateur | onglet *Noise* : générer ou charger, puis comparer à la consigne |
| DIC | Tests affine et bande : atténuation, élargissement | partiel | — | tests `test_affine_field`, `test_shear_band_slope`, `test_E_shear_band…` | idem | T | pas de mesure exportée | réutiliser les générateurs des tests |
| DIC | Biais périodique sous-pixel | partiel | — | `test_peak_locking`, `test_C_subpixel_translation` | idem | T | pas de balayage ni de courbe | balayage 0–1 px, courbe du biais |
| DIC | Biais, dispersion, RMSE, statistiques spatiales, % rejet, cartes | absent | carte ZNCC et masque `valid` seulement | — | `_QUALITY_FIELDS` (viewer) | L | ABS | statistiques sur les cas à consigne |
| DIC | Conversion incrément → vitesse, unités | présent (mesure) / absent (erreurs) | automatique | `compute_dic_fields` | `dic.py:492-501` : V = U·mm/px·fps, t au milieu de la paire, y inversé | L T | les erreurs ne sont pas converties | multiplier σ_u par mm/px·fps |
| DIC | Biais systématique vs dispersion aléatoire | absent | — | — | — | L | ABS | découlera de la ligne précédente |
| DIC | Incertitude analytique (Hild & Roux) | NR | case « Compute uncertainty » désactivée | `estimate_sigma_f`, `compute_uncertainty` | `dic_tab.py:521`, `dic_global.py:766/885` | L U | aucune source de σ_f (onglet *Noise* vide) | σ_f depuis la paire à vide de la session |
| ε_q | Définir, enregistrer, transmettre ε_q | partiel | deux jeux de valeurs : « RMSE thresholds » (absolus, **non utilisés**) et « Sizing tolerances (relative) » (utilisés, 0.02) | `_tolerances()` vers GCI et domaine | `optimization_tab.py:146/257`, `OptimizationCfg` | L U | provenance non tracée ; seules des tolérances relatives sont transmises | un seul tableau ε_q avec champ de provenance |
| ε_q | ε_q = max(k_num σ_num, α σ_exp) | absent | — | — | aucune occurrence | L | ABS ; formulation en place : tolérance relative fixe par grandeur | après les outils DIC |
| ε_q | Ne pas assimiler les fluctuations temporelles au bruit | partiel | fenêtre 30–100 % codée en dur | `window_mask` | `domain_convergence.py:190/306`, `mesh_gci.py:308` | L | aucune σ estimée, donc pas de confusion automatique, mais pas d'estimation non plus | exposer et tracer la fenêtre |
| ε_q | Incertitudes expérimentales T et efforts | absent | onglets *IRT*, *Forces*, *Noise*, *Calib. thermal* vides | `load_forces` (import + tracé seulement) | `experimental_data_tab.py:59-66` | U L | DATA + ABS | — (données à acquérir) |

### 3.2 Sensibilité physique à L₀** fixé

| Tâche | Capacité attendue | Statut | Accès UI | Moteur | Preuves | Vérif. | Limite / blocage | Correction minimale |
|---|---|---|---|---|---|---|---|---|
| Plan | Sélection des paramètres, nominal, bornes, pas absolu ou relatif | présent | tableau Vary/Ref/Min/Max/Delta/Delta% | `param_registry`, `jacobian_plan.build_plan` | `sensitivity_tab.py:120-130` | L T U | vitesse de coupe à facteur m/min figé (TODO §1) ; A10 | exclure la catégorie « Numérique » |
| Plan | Nominal / +δ / −δ | présent | FD scheme = central | `build_plan` | `jacobian_plan.py:75-113` | L T X | — | — |
| Plan | Autres paramètres fixes | présent | — | `plan_to_configs` (deepcopy) ; le worker reçoit une copie | `jacobian_plan.py:116`, `sensitivity_tab.py:844` | L T | le profil a pu être modifié avant (A1) | corriger A1 |
| Jobs | Orchestration, état, échecs | présent | Run / Cancel / progression `.sta` | `run_worker`, `runner_core.run_plan` | `runDone`, `failures`, `n_attempted` | L T | Windows `taskkill` NV | — |
| Jobs | Reprise des calculs manquants | absent | — | — | `run_worker.py:343` supprime le bundle existant et relance | L | tout est recalculé | sauter un run si `meta.json.model_config` est identique |
| Extraction | Vx, Vy, T, EVF, efforts associés aux points et temps | partiel | cases EVF / V / TEMP | `cel_results.extract_results` | V1, V2, V, TEMP, EVF et RF1/RF2 écrits ; UI limitée à `V` | L U ; NV (ODB) | A4 | ajouter V1 et V2 aux cases |
| Indicateurs | Dérivées signées, cartes RMS | présent (cartes) | onglet Maps (Signed / Aggregate) + CSV | `jacobian_field_maps`, `map_export.write_maps` | `map_export.py:1-22` | L T X | A5 | masque et fenêtre optionnels |
| Indicateurs | Indicateurs globaux | CALC | colonnes `[field]`, `Δ% (rel)` | `jacobian_field_analysis` | A2 | X | +δ seul, SSD non normalisée | RMS centrale normalisée |
| Indicateurs | Normalisation | partiel | case Norm (**décochée par défaut**) | `analyze` : élasticité dQ/dx·x₀/Q₀ | `jacobian_plan.py:139-164` | L X | aucune normalisation pour les champs ; le Chart mélange les unités | élasticité ou δ·dF/dx pour les champs |
| Sélection | 2–3 directions critiques | absent | classement par QoI seulement | `jacobian_ranking` | — | L | ABS | tableau croisé normalisé |
| Définitions | Colonnes `sensitivity`, `dQdx`, `normalized`, `[field]`, `Δ% (rel)` | voir §5 | CSV « Save results… » | `export_results` | `_COLUMNS` (`:24`) | L X | pas d'unités, de δ ni de schéma dans le CSV | ajouter ces colonnes |
| Définitions | Efforts moyens | CALC | QoI `Fx_mean` / `Fy_mean` | `qoi.qoi_Fx_mean` | A3 | L | moyenne de \|RF\|, sans fenêtre | ajouter une moyenne signée fenêtrée en N/mm |
| Traitement | Fenêtre temporelle, NaN, masque EVF | partiel | — | `nansum` / `nanmean` | A5 | L | toutes frames, pas de masque, `_align` tronque | réutiliser `window_mask` et le masque EVF |
| Export | Conservation des champs signés, rechargement | partiel | bundles et cartes sur disque ; CSV manuel | `config.json` (plan complet) | `sensitivity_tab.py:860-904` | L | matrice Y par run non exportée ; aucun rechargement d'étude | exporter Y ; recharger une étude |

### 3.3 Robustesse du dimensionnement et Φ_L

| Tâche | Capacité attendue | Statut | Accès UI | Moteur | Preuves | Vérif. | Limite / blocage | Correction minimale |
|---|---|---|---|---|---|---|---|---|
| États | Définir χ⁻, χ₀, χ⁺ | absent | — | — | — | L | ABS | liste d'états = surcharges χ d'un profil |
| États | Réutiliser des résultats identiques | absent | — | — | `optimization_tab.py:629`, `run_worker.py:343` suppriment puis relancent | L | ABS | comparer `model_config` de `meta.json` |
| États | Calcul avec L₀** conservé | partiel | onglet Job (manuel) | `run_simul` | — | L | A1 peut altérer L₀** | corriger A1 |
| Contrôle | Configurations de contrôle | partiel | GCI (3–6 maillages, domaine fixe) ; domaine (+4 éléments) | `mesh_gci`, `domain_convergence` | `optimization_tab.py:900-1066` | L T ; NV (Abaqus) | outil, mass scaling et pipeline complet non raccordés | — |
| Contrôle | Comparer deux L au même χ | partiel | uniquement à l'intérieur des études | voir A7 | — | L X | pas d'outil générique ; métriques hétérogènes | outil « comparer 2 bundles » avec contrôle d'égalité de χ |
| Φ_L | E_q, E_q/ε_q, Φ_L, grandeur critique | partiel | log seulement | `_binding_change` = max(Δ_q/tol_q) interne | `domain_convergence.py` ; `_e_max` et `_plot` morts | L U | pas de Φ_L exporté ; grandeur critique non affichée | calculer, afficher et écrire Φ_L et la grandeur critique |
| Décision | Conserver ou relancer le dimensionnement | absent | — | — | — | L | ABS | règle Φ_L ≤ 1 |
| Export | Tableau χ → Φ_L → L** | absent | — | — | A6 | L | ABS | CSV dans le dossier d'étude |
| Principe | Physique (Δχ) vs numérique (ΔL, même χ) | absent | séparation par onglets seulement | — | A10, A1 | L X | aucune vérification de χ identique | contrôle d'égalité de χ avant tout E_q |
| Cohérence | ROI, coordonnées, temps, unités lors des changements de L | partiel | ZOI (onglet Model), ROI (Geometry) | `roi_grid`, `nearest_samples`, `bilinear_field` | A7, A8 | X | égalités multiples, ZOI hors ROI, interpolation, force /h dans le GCI seulement | contrôle ZOI ⊂ ROI ; clés exactes à maillage fixe |
| Correspondance | Sans interpolation | NR | — | `domain_opt.grid_keys`, `align_to_reference` | `domain_opt.py:320/356` | L T | non raccordé | l'utiliser en convergence de domaine |
| Garde-fous | Réussite du job | présent | — | `_check_job_succeeded` (`.sta`), code retour, bundle | `cel_model.py:855` | L ; NV | — | — |
| Garde-fous | Contrôle énergétique | NR | — | `mean_ke_ie_ratio` ; ALLKE/ALLIE extraits | `domain_opt.py:377`, `cel_results.py` | L T | utilisé seulement par `mass_scaling` (non raccordé) | rapporter le ratio par run |
| Garde-fous | Transport EVF, copeau aux frontières | absent | — | — | extraction limitée à la ROI | L | ABS ; nécessite une extraction côté Abaqus | EVF sur une bande de bord (Abaqus, à valider) |

### 3.4 Variabilité près du seuil et hystérésis

| Tâche | Capacité attendue | Statut | Accès UI | Moteur | Preuves | Vérif. | Limite / blocage | Correction minimale |
|---|---|---|---|---|---|---|---|---|
| Seuil | Repérer Φ_L ≈ 1 | absent | — | — | pas de Φ_L persistant | L | dépend de §3.3 | — |
| Seuil | Protocole de répétabilité, σ_Φ | absent | — | — | — | L | un calcul déterministe répété n'estime pas l'incertitude numérique | protocole à définir (perturbation de L ou du maillage) |
| Décision | Hystérésis, zone incertaine, anti-oscillation | absent | — | — | — | L | ABS | après Φ_L |
| Trace | Journal des décisions, admissibilité finale | absent | — | — | — | L | ABS | après Φ_L |

### 3.5 Boucle identification–dimensionnement

| Tâche | Capacité attendue | Statut | Accès UI | Moteur | Preuves | Vérif. | Limite / blocage | Correction minimale |
|---|---|---|---|---|---|---|---|---|
| Boucle | χᵏ → Lᵏ → χ̂ᵏ⁺¹ → relaxation → χᵏ⁺¹ | absent (seulement envisagée) | « Inverse identification — coming later » | aucun solveur (`scipy.optimize` absent) | `main.py:127` ; TODO §3.3 | L U | ABS | hors périmètre de cette semaine |
| Données | Cibles réellement disponibles, pondérations | absent | — | champ DIC exportable ; `field_metrics` réutilisable | TODO §3.2 (alignement simulation ↔ essai ouvert) | L | DATA : pas de T ni d'efforts expérimentaux traités | — |
| Suivi | Paramètres, résidus, Φ_L, L, coût, oscillations, arrêt, reprise | absent | — | — | — | L | ABS | — |

### 3.6 Traçabilité, exports et consolidation

| Tâche | Capacité attendue | Statut | Accès UI | Moteur | Preuves | Vérif. | Limite / blocage | Correction minimale |
|---|---|---|---|---|---|---|---|---|
| Trace | `.inp` exécuté conservé | présent | automatique | `cleanup_working_directory` garde `.inp`, `.odb`, `.sta`, `.npz`, `.dat`, `.msg` | `cel_model.py:908-922` | L ; NV | — | — |
| Trace | Paramètres physiques et numériques | présent | — | `meta.json["model_config"]` (unités internes) ; `config.json` (unités d'affichage + `unit_system`) | `cel_results.py`, `sensitivity_tab.py:860` | L T | A1 | — |
| Trace | Mass scaling appliqué et traitement thermique | présent | Step | ρ·ms, Cp/ms (ρ·Cp conservé) | `cel_model.py:341-367` | L ; NV (`.inp`) | — | vérifier le `.inp` (§7) |
| Trace | CL, vitesses, températures, contact | présent | — | `model_config` (`bcs`, `interaction`) | `to_params_dict` | L X | — | — |
| Trace | ROI, maillage d'extraction, fenêtre, masques | partiel | — | ROI dans `meta.json` ; ZOI, grille et tolérances dans `config.json` | `optimization_tab.py:916-924` (domaine), `:998-1006` (GCI) | L | fenêtre (0.3, 1.0) non enregistrée ; sensibilité : ni fenêtre ni masque | enregistrer fenêtre et masque |
| Trace | Formules, normalisations, tolérances | partiel | — | `config.json` (`normalize`, `delta`, `scheme`) ; `maps_index.csv` | — | L | le CSV de résultats ne contient ni métrique ni unités | colonnes `unit`, `metric`, `delta`, `scheme` |
| Trace | État des jobs, décisions | partiel | log, statut | `RunResult.failures` (en mémoire) | — | L | non persisté ; aucune décision | `runs_status.csv` |
| Cohérence | Profil / campagne / entrées exécutées | partiel | — | instantané d'unités du plan (testé) | `test_sensitivity_fixes.py` | T | A1 ; vitesse m/min figée | corriger A1 |
| Reprise | Export, rechargement sans recalcul | partiel | Results › Load results… (bundle par bundle) | `ResultsBundle.load` | `results_tab.py:199` | L T | aucune étude rechargeable | recharger `config.json` + bundles |

---

## 4. Moteurs non raccordés à l'interface (existants et testés)

| Module | Rôle | Tests | Accès UI |
|---|---|---|---|
| `sensitivity/mass_scaling.py` (+ `make_mass_scaling_sample_fn`) | identification du facteur avec garde ALLKE/ALLIE | `test_mass_scaling.py` | aucun |
| `sensitivity/domain_opt.py` + `domain_opt_worker.py` | domaine par doublement et bissection, clés exactes, écart moyen absolu | `test_domain_opt.py` | aucun (cité dans l'en-tête de `optimization_tab.py:9`, non importé) |
| `sensitivity/mesh_opt.py` | raffinement de Cauchy, `verify_stability` | `test_mesh_opt.py` | aucun |
| `sensitivity/mesh_pipeline.py` + `_worker.py` | pipeline en 6 étapes (pièce, outil, domaine, vérifications) + mass scaling | `test_mesh_pipeline.py` | aucun |
| `dic_global.estimate_sigma_f` / `compute_uncertainty` | incertitude DIC analytique | `test_M_predicted_scatter_matches_empirical` | case désactivée |
| `sensitivity/morris_plan.py` | criblage global | tests `lot2b` | raccordé (méthode Morris) |

Note : la docstring de `mesh_opt.py` annonce une RMSE, mais `errors_between`
calcule un **écart moyen absolu** (`domain_opt.roi_error`, alias `roi_rmse`).
Le libellé « RMSE thresholds » de l'interface reprend cette confusion.

---

## 5. Définition exacte des colonnes de sensibilité (vérifiée, annexe A)

| Colonne | Contenu calculé | Unité | Schéma utilisé |
|---|---|---|---|
| `sensitivity` (QoI scalaire) | `dQdx` si Norm décochée ; sinon élasticité `dQdx·x₀/Q₀` | unité QoI / unité affichée du paramètre, ou sans dimension | celui du plan (central : (Q₊−Q₋)/2δ) |
| `dQdx` | dérivée brute, toujours présente pour les QoI scalaires ; vide pour les champs | unité QoI / unité du paramètre | idem |
| `normalized` | booléen de la case Norm (défaut : faux) ; vide pour les champs | — | — |
| `x0`, `Q0` | valeur nominale (unités d'affichage du plan) et QoI du run de base | — | — |
| `<var> [field]` | `Σ_{éléments,frames}(F₊−F₀)² / δ²`, toujours ≥ 0 | (unité du champ)² / (unité du paramètre)², proportionnel à N_él·N_frames | **+δ seul**, même en central |
| `<var> Δ% (rel)` | `100·‖F₊−F₀‖/‖F₀‖` (L2 relative sur éléments × frames) | % pour le pas δ choisi (non divisé par δ) | **+δ seul** |
| Carte signée (Maps, CSV) | `(F₊−F₋)/(2δ)` par élément et par frame ; agrégats : moyenne signée, RMS | unité du champ / unité du paramètre (`map_export.FIELD_UNITS` : V mm/s, TEMP °C) | central |

Traitement de V : composantes V1 et V2 moyennées sur les nœuds de chaque
élément, puis norme. Les colonnes et cartes « V » sont donc des **dérivées de
la norme**, et non des normes de dérivée. Les tableaux (`[field]`, `Δ%`)
et les cartes n'utilisent **pas le même schéma** : pour une réponse non
linéaire, ils peuvent se contredire.

---

## 6. Plan de correction

### 6.1 Fonctionnalités manquantes qui bloquent les calculs de cette semaine

Hypothèse sur le contenu de la semaine, à confirmer : mass scaling, puis
convergence du maillage et du domaine, puis premiers tests de robustesse
autour de χ₀.

1. Protection du profil actif contre les études (A1) : prérequis de tout le
   reste.
2. Persistance des résultats GCI et domaine, avec Φ_L et grandeur critique
   (A6).
3. Raccordement du mass scaling. À défaut, fixer le facteur manuellement et
   documenter la borne affichée dans *Step*.
4. Tableau ε_q unique, avec provenance marquée « provisoire », tant que les
   outils DIC ne fournissent pas σ.

### 6.2 Ordre minimal des corrections (architecture existante)

1. **A1** : `copy.deepcopy(self.cfg)` dans `_on_run_mesh_gci` et
   `_on_run_domain_convergence`, puis écrire le L retenu dans `config.json`,
   sans toucher au profil.
2. **Persistance** : écrire `results.json` et `results.csv` dans le dossier
   d'étude (scalaires par run, E_q, E_q/ε_q, `stopped_by`, fenêtre, seuil EVF,
   grandeur critique, Φ = max E_q/ε_q). Brancher `_plot` sur ces données.
3. **Tolérances** : supprimer ou raccorder `_q_eps`. Un seul jeu ε_q (absolu
   ou relatif, déclaré) avec un champ `source` ∈ {provisoire, DIC, GCI}.
4. **Sensibilité (A2, A4, A5)** : indicateur global central
   `RMS(F₊−F₋)/(2δ)` sur éléments × frames, forme normalisée ×x₀ ;
   cases V1 et V2 ; options fenêtre (`window_mask`) et masque EVF
   (règle de `reduce_zoi`) ; unités, δ, schéma et métrique dans le CSV ;
   export de Y par run. Conserver les colonnes actuelles, renommées
   `… SSD(+δ)`, pour la comparabilité avec la campagne existante.
5. **Efforts (A3)** : QoI `Fc_mean_win` (moyenne signée fenêtrée,
   `_history_window_mean`, divisée par `elem_size` en N/mm) à côté de
   `Fx_mean`.
6. **Correspondance (A7, A8)** : en convergence de domaine (maillage fixe),
   remplacer le plus proche voisin par `grid_keys` / `align_to_reference` ;
   vérifier que la ZOI est incluse dans la ROI avant le lancement.
7. **Comparaison de deux L au même χ** : outil qui charge deux bundles,
   refuse si le sous-ensemble physique de `model_config` diffère, puis
   calcule E_q et Φ_L. Base des tests de robustesse χ⁻/χ₀/χ⁺ avec
   réutilisation par égalité de `model_config`.
8. **Garde-fous** : ratio ALLKE/ALLIE par run dans les résultats d'étude ;
   contrôle EVF de bord (côté Abaqus, à valider sur un vrai run).
9. **DIC** (bloque ε_q expérimental) : onglet *Noise* fondé sur les
   générateurs des tests (déplacement nul, translations 0.25–2 px en x et y,
   affine, bande) ; statistiques biais / σ / RMSE / % rejet et cartes ;
   `estimate_sigma_f` depuis la paire à vide ; activation de
   `compute_uncertainty`.
10. Ensuite seulement : ε_q = max(k σ_num, α σ_exp), σ_Φ et hystérésis,
    boucle d'identification.

### 6.3 Améliorations utiles mais non bloquantes

- `_log_ui` est appelé depuis le thread de travail
  (`optimization_tab.py:724`) : accès à un widget Qt hors du thread GUI,
  risque de plantage sur les longues études. Passer par un signal.
- La convergence de domaine relance le domaine de base à chaque itération
  alors qu'il vient d'être calculé comme candidat : un run gaspillé par
  itération. Ajouter un cache par dimensions.
- `fps ≤ 0` dans la DIC ; libellé « RMSE » erroné ; « Compute initial domain »
  et « max cap » sans effet sur l'étude.
- A9 : documenter ou dériver la constante 90.6 depuis
  `mass_scaling_bounds`.
- A10 : exclure la catégorie « Numérique » de l'onglet Sensitivity, ou
  l'étiqueter.

---

## 7. Éléments non vérifiables ici et protocole sur votre poste

| Élément | Comment le vérifier |
|---|---|
| Mass scaling effectivement écrit | Job › *Write .inp only* avec ms = 10 ; dans le `.inp`, `*Density` = ρ·10 et `*Specific Heat` = Cp/10 pour les deux matériaux |
| V1, V2, RF1, RF2 et ALLKE/ALLIE présents dans le bundle | `np.load("<job>.results.npz").files` doit contenir `EULER…__fields__V1`, `…V2`, `history__RF1_RP`, `history__ALLKE` |
| Nombre d'échantillons d'historique | `len(history__time)` ≈ `n_frames + 1` (conséquence de `ho_n_intervals = n_frames`) |
| Série filtrée ou brute | log `.gui.log` : ligne « Filtered series preferred: field=… history=… » |
| Signe de RF1 et RF2 | tracer `history__RF1_RP` : si le signe est négatif, `Fx_mean` (valeur absolue) et la moyenne signée diffèrent |
| Cancel Windows (`taskkill /T`) | `docs/abaqus_validation_checklist.md` |
| Campagne de 13 runs | ouvrir `config.json` : `scheme`, `varied_parameters[].delta`, `normalize`, `field_vars` ; relire `maps_index.csv` (`map_unit`, `delta`) |

---

## 8. Checklist de validation après correction

- [ ] Après un GCI et une convergence de domaine, `cfg.elem_size` et
      `cfg.euler_geometry` sont inchangés (test unitaire sur le schéma de
      `v_mutation`).
- [ ] Le dossier d'étude contient `results.json` et `results.csv` avec E_q,
      ε_q, E_q/ε_q, Φ_L, grandeur critique, fenêtre et seuil EVF, et ces
      fichiers se rechargent.
- [ ] Réponse asymétrique synthétique (+1 / −5) : l'indicateur global central
      donne 100, comme la carte ; le résultat est invariant au nombre
      d'éléments et de frames.
- [ ] Rotation de V à norme constante : V1 et V2 non nuls, |V| nul, tous deux
      disponibles dans l'interface.
- [ ] Effort : sur un signal synthétique avec transitoire, `Fc_mean_win`
      exclut le transitoire et vaut F/`elem_size`.
- [ ] Une ZOI hors ROI est refusée avant lancement.
- [ ] À maillage fixe, la convergence de domaine compare les mêmes cellules
      (clés identiques) entre deux domaines.
- [ ] Outil de comparaison de deux L : refus si χ diffère ; E_q, Φ_L et
      grandeur critique exportés.
- [ ] Un run dont `model_config` est identique est réutilisé, pas relancé.
- [ ] CSV de sensibilité : colonnes `unit`, `metric`, `delta`, `scheme`,
      `window`, `evf_mask` présentes ; Y par run exporté.
- [ ] DIC : cas à déplacement nul et translations 0.25–2 px en x et y donnent
      biais, σ, RMSE et % de rejet, avec cartes et conversion en mm/s.
- [ ] Suite de tests complète verte (référence avant correction :
      714 passed, 4 skipped).
- [ ] Sur le poste Abaqus : points du §7 cochés.

---

## Annexe A — scripts de vérification exécutés

Environnement : venv hors dépôt, `pip install -r requirements.txt`,
`PYTHONPATH=.`, `QT_QPA_PLATFORM=offscreen`.

`v_mutation` (A1) :

```python
from gui.core.model_config import ModelConfig
from gui.core.domain_sizing import DomainDims
from gui.sensitivity.domain_convergence import run_domain_convergence
from gui.sensitivity.mesh_gci import run_mesh_gci
cfg = ModelConfig(); e = float(cfg.elem_size)
dims = DomainDims(h_wp=12*e, h_void=12*e, l_wp=12*e, l_void=12*e)
try: run_domain_convergence(lambda c: None, cfg, (-5*e,5*e,-5*e,5*e), dims, e, e,
                            tolerances={"TEMP": .02}, margin_elems=0)
except RuntimeError: pass
print(cfg.euler_geometry.h_wp)          # 0.2 -> 0.06
cfg2 = ModelConfig()
try: run_mesh_gci(lambda c: None, cfg2, (-5*e,5*e,-5*e,5*e), dims, e, finest_elem_size=e/2)
except RuntimeError: pass
print(cfg2.elem_size)                   # 0.005 -> 0.0025
```

`v_sens` (A2, A4, §5) : plan central sur μ (x₀ = 0.3, δ = 0.03), champs
constants F₀ = 100, F₊ = 101, F₋ = 95 ; résultats :
`[field]` = 5.556e4 (10 éléments × 5 frames) et 8.889e5 (40 × 20) ;
`Δ%` = 1 ; carte centrale = 100 ; rotation de V : d|V| = 0, |dV| = 141.4.

`v_nn` (A7, A8) : maillage structuré e = 0.005, ZOI alignée sur les nœuds :
289/289 points équidistants de plusieurs centroïdes ; point (0.2, 0) hors ROI
échantillonné sur le centroïde (0.0475, 0.0025), soit à 0.153 mm.

`v_ui` (U) : arborescence des onglets et commandes relevés sur `MainWindow`
(voir §3).
