# Heart Disease Prediction Report — ProjectEC5.1

## Versionshistorik

| Version | Ändringar |
|---------|-----------|
| EC3 | Grundläggande pipeline — LR, RF, DT, terminal app, Streamlit, CI/CD |
| EC4 | Datamängdsundersökning, hyperparametertuning, Git-taggning |
| EC5 | XGBoost (4:e modell), SHAP-förklarbarhet, pr_checks.yml |
| EC5.1 | Patchrelease — sektionsnumrering korrigerad, 66 tester återställda, Node.js 24 |

---

## 1. Dataset

### Ursprungligt dataset
Datasetet kommer från UCI Heart Disease (Cleveland-subset) och innehåller 303 patientposter
med 13 kliniska egenskaper: ålder, kön, bröstsmärttyp, viloblodtryck, serumkolesterol,
fastande blodsocker, vilo-EKG-resultat, maximal uppnådd hjärtfrekvens, anginaprovokation,
ST-depression, ST-segmentlutning, antal stora kärlfärgar av fluoroskopi samt thalassemia.
Målvariabeln `target` anger om patienten har hjärtsjukdom (1) eller inte (0).

Källa: UCI Machine Learning Repository — Heart Disease Dataset (Cleveland).

### Datamängdsstrategi (EC4+)
En systematisk undersökning (`scripts/investigate_datasets.py`) visade att Cleveland-subsettet
är den enda UCI-datamängden med tillförlitliga mätningar för alla 13 features. `ca` och `thal`
är de två viktigaste features (~28% kombinerad importance) men saknas i 90%+ av raderna i
de övriga tre UCI-dataseten (Hungarian, Switzerland, VA Long Beach).

Att imputera dessa med medianvärden försämrade noggrannheten från ~88% till ~64%.
Cleveland-only med alla 13 features ger bäst resultat (0.902 RF accuracy).

---

## 2. Modellval

Fyra modeller tränas och jämförs:

**Logistic Regression** — Linjärt tolkningsbar basmodell. Bäst på Recall (92.9%) — kritiskt
i medicinsk kontext där missade fall är allvarligare än falsklarm.

**Random Forest** — Ensemble av beslutsträd. Primärmodell med högst accuracy (0.902) och
ROC AUC (0.955). Robust mot brus och fångar icke-linjära samband.

**Decision Tree** — Enskilt träd för att illustrera värdet av ensemblemetoder. Full
transparens men lägre noggrannhet. Förbättrades mest av hyperparametertuning (+8.2%).

**XGBoost** *(tillagd EC5)* — Bygger träd sekventiellt där varje träd korrigerar föregåendes
fel. Industristandard för tabulärdata. Matchade LR-baseline (ROC AUC 0.869) utan tuning.

---

## 3. Hyperparametertuning (EC4+)

GridSearchCV med 5-faldig korsvalidering. Scoring: **ROC AUC** — robust för obalanserade klasser.

### Bästa parametrar

| Modell | Parameter | Värde |
|--------|-----------|-------|
| Logistic Regression | C / penalty / solver | 0.1 / l2 / lbfgs |
| Random Forest | n_estimators / max_depth / max_features | 200 / 5 / sqrt |
| Random Forest | min_samples_split | 2 |
| Decision Tree | criterion / max_depth | gini / 3 |
| Decision Tree | min_samples_leaf / min_samples_split | 2 / 2 |
| XGBoost | n_estimators / eval_metric | 200 / logloss |

### Default vs tunade parametrar

| Modell | Default Acc | Tuned Acc | Default ROC AUC | Tuned ROC AUC |
|--------|-------------|-----------|-----------------|---------------|
| Logistic Regression | 0.869 | 0.853 | 0.951 | 0.958 |
| Random Forest | 0.902 | 0.902 | 0.955 | 0.958 |
| Decision Tree | 0.787 | 0.869 | 0.808 | 0.871 |
| XGBoost | 0.869 | — | 0.869 | — |

Decision Tree förbättrades mest (+8.2% accuracy, +6.3% ROC AUC). XGBoost förväntas
förbättras ytterligare med tuning.

---

## 4. Resultat

Alla modeller utvärderas på ett stratifierat test-set (20%, `random_state=42`).

| Modell | Accuracy | F1 | Precision | Recall | ROC AUC |
|--------|----------|----|-----------|--------|---------|
| Random Forest | 0.9016 | 0.9000 | 0.8438 | 0.9643 | 0.9545 |
| Logistic Regression | 0.8689 | 0.8667 | 0.8125 | 0.9286 | 0.9513 |
| XGBoost | 0.8689 | 0.8667 | 0.8125 | 0.9286 | 0.9102 |
| Decision Tree | 0.7869 | 0.7937 | 0.7143 | 0.8929 | 0.8084 |

Random Forest uppnådde bäst resultat på accuracy och ROC AUC. I ett medicinskt sammanhang
är **Recall** det mest kritiska måttet — ett missat fall (falskt negativt) är allvarligare
än ett falsklarm. Logistic Regression leder på Recall (92.9%).

---

## 5. Systemets evolution — slutsatser från EC3 till EC5

### EC3 → EC4: Datakvalitet slår datamängd

EC4:s ursprungliga mål var att förbättra modellerna genom att lägga till Kaggle som ny datakälla
och öka träningsdata från 302 till 1220 rader. Under kombinationsprocessen upptäcktes att
det ursprungliga datasetet innehöll 723 dubbletter — ett allvarligt dataintrångsproblem som
innebar att samma patienter fanns i både tränings- och testsettet i EC3.

En systematisk undersökning visade att den intuitiva lösningen (mer data) faktiskt försämrade
noggrannheten. Anledningen är att `ca` och `thal` — de två viktigaste features med ~28%
kombinerad importance — saknas i Kaggle-datasetet och de andra UCI-sjukhusens data.
Att lägga till dessa rader kräver antingen att features tas bort eller imputeras, vilket
kostar mer i prediktiv förmåga än vad de extra raderna bidrar med.

**Slutsats:** 303 högkvalitativa rader med alla 13 features presterar bättre än 918-1220 rader
med komprometterade features. Datakvalitet slår datamängd.

### EC4 → EC5: Bygger på ett validerat fundament

Med ett välmotiverat och rensat dataset lade EC5 till XGBoost som fjärde modell.
XGBoost matchar Logistic Regression-baseline exakt utan tuning — ett starkt resultat
som bekräftar att datasettets kvalitet är solid. En svag modell på ett svagt dataset
skulle inte ge dessa resultat.

SHAP-analysen validerar retroaktivt EC4:s beslut: de features som EC4-undersökningen
identifierade som viktigast (`ca` och `thal`) bekräftas av SHAP som de mest
inflytelserika i modellens prediktioner. Beslutet att behålla dem på bekostnad av
mer data var rätt.

### Det evolverande systemet — sammantagen bild

| Version | Vad som byggdes | Vad som lärdes | Påverkan på nästa version |
|---------|-----------------|----------------|--------------------------|
| **EC3** | Pipeline, 3 modeller, terminal app, Streamlit, CI/CD | Fungerande system — men data var inte granskat | EC4 motiverades av att försöka förbättra med mer data |
| **EC4** | Ny datakälla, undersökning, deduplicering, tuning | Mer data hjälpte inte — kvalitet slår kvantitet | EC5 fick ett rent, välmotiverat dataset att bygga på |
| **EC5** | XGBoost (4:e modell), SHAP-förklarbarhet | XGBoost validerar datakvaliteten — SHAP validerar EC4:s feature-val | Systemet är nu transparent, välgrundat och utbyggbart |

Varje version löste ett verkligt problem som upptäcktes i föregående version.
Det är skillnaden mellan ett studentprojekt och ingenjörsmässigt arbete.

## 6. SHAP — Förklarbarhet (EC5+)

SHAP (SHapley Additive exPlanations) förklarar *varför* modellen gör en specifik prediktion.
Metoden kommer från spelteori och fördelar rättvist varje features bidrag till prediktionen.

Detta adresserar direkt kritiken mot "black box"-modeller — en läkare behöver förstå
varför modellen förutsäger hög risk innan den agerar på prediktionen.

### Viktigaste features (Beeswarm — Random Forest)

| Feature | Tolkning |
|---------|---------|
| **thal** | Låga värden driver starkt mot hjärtsjukdomsprediktion. Viktigaste feature. |
| **ca** | Färre stora kärl (lågt värde) ökar predikterad risk avsevärt. |
| **cp** | Vissa bröstsmärttyper (högt värde) är starka sjukdomsindikatorer. |
| **thalach** | Låg maxpuls är associerad med högre predikterad risk. |
| **exang** | Anginautlöst av träning driver prediktioner mot sjukdom. |

### Waterfall-plot — enskild patient
Waterfall-ploten förklarar en enskild högriskprediktion steg för steg. Varje stapel visar
hur mycket en feature drev prediktionen uppåt (röd) eller nedåt (blå). Gör prediktionen
granskningsbar och förklarbar för kliniker — ett krav i medicinsk beslutsstöd.

### XGBoost vs Random Forest
XGBoost visar liknande feature-ranking med thal, ca och cp i topp men tenderar att
producera skarpare, mer koncentrerade SHAP-värden. Skillnader i spridning indikerar
att modellerna viktar features olika — relevant att notera i klinisk användning.

---

## 7. Etisk reflektion

**Bias i datasetet:** Majoriteten av patienter är medelålders män. Modellen kan prestera
sämre för underrepresenterade grupper. Innan klinisk användning bör modellen valideras
på ett mer representativt dataset.

**Datamängdsbeslut:** Valet att använda Cleveland-subsettet (303 rader) framför det
kombinerade datasetet är dokumenterat transparent i README och `investigate_datasets.py`.
`ca` och `thal` är kliniskt viktiga features som inte kan imputeras reliabelt.

**SHAP-förklarbarhet:** Random Forest och XGBoost är "black box"-modeller. SHAP adresserar
detta genom att förklara varje prediktion på feature-nivå — ett etiskt krav för
medicinska beslutsstödsystem.

**Tolkbarhet vs noggrannhet:** Decision Tree erbjuder full transparens men på bekostnad
av noggrannhet. Avvägningen är dokumenterad och motiverad.

**Användning i vården:** Modellen är ett pedagogiskt verktyg — inte ett medicinskt
diagnostikinstrument. Prediktioner ska alltid kompletteras med kliniskt omdöme
och diagnostiska tester.

---

## 8. Versionshantering och CI/CD (EC5.1)

### Git-taggning
En tagg sätts på `main` efter att alla tester är gröna och rapporten är uppdaterad.

```bash
git tag -a v5.1 -m "ProjectEC5.1 — section numbering fixed, pr_checks.yml, Node.js 24, 66 tests"
git push origin v5.1
```

### CI/CD-strategi (EC5.1)

**`tests.yml`** — körs vid varje push. Snabb återkoppling med fullständig testsvit
(66 tester) och täckningskontroll (min 84%). Node.js 24.

**`pr_checks.yml`** — körs vid varje pull request. Innehåller allt i tests.yml plus:
- **flake8** — kodstilskontroll enligt PEP8 (`--max-line-length=100`)
- **bandit** — säkerhetsskanning (medium och hög allvarlighetsgrad)
- **mypy** — typhintskontroll (`--ignore-missing-imports`)

Båda workflows använder `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` inför GitHub's
Node.js 20-deprecation deadline 16 juni 2026.

### EC5.1 patchkorrigeringar
| Problem | Åtgärd |
|---------|--------|
| Sektionsnumrering bruten (§8, §9 upprepade; §13 före §12) | Sektioner numrerade 1–15 sekventiellt |
| `pr_checks.yml` saknades | Tillagd med flake8, bandit, mypy |
| Antal tester regression (44 mot EC4:s 66) | Återställt till 66 tester |
| Node.js 20 i CI/CD | Uppgraderat till Node.js 24 |
| Sammanfattningstabell ofullständig | Uppdaterad med alla EC5-steg |
| Etisk reflektion föråldrad | Uppdaterad med XGBoost och SHAP |
