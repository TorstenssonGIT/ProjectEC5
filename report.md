# Heart Disease Prediction Report

## 1. Dataset

### Original dataset
Datasetet kommer från UCI Heart Disease (Cleveland-subset) och innehåller 303 patientposter
med 13 kliniska egenskaper: ålder, kön, bröstsmärttyp, viloblodtryck, serumkolesterol,
fastande blodsocker, vilo-EKG-resultat, maximal uppnådd hjärtfrekvens, anginaprovokation,
ST-depression, ST-segmentlutning, antal stora kärlfärgar av fluoroskopi samt thalassemia.
Målvariabeln `target` anger om patienten har hjärtsjukdom (1) eller inte (0).

Källa: UCI Machine Learning Repository — Heart Disease Dataset (Cleveland).

### Utökat dataset (ProjectEC5)
För att öka modellernas generaliserbarhet kombinerades det ursprungliga datasetet med
Kaggle Heart Failure Prediction Dataset (fedesoriano, 2021), som kombinerar fem
hjärtsjukdomsdataset från Cleveland, Ungern, Schweiz, Long Beach VA och Stalog.
Det kombinerade datasetet innehåller 918 unika poster efter borttagning av dubbletter.

**Kombinationsmetodik:**

Det kombinerade datasetet (`heart_combined.csv`) skapas av `src/data_preparation.py`.
Originalfilerna (`heart.csv` och `heart_kaggle.csv`) modifieras aldrig.

1. Kaggle-datasetet använder textetiketter för kategoriska variabler som konverteras
   till numeriska värden för att matcha UCI-kodningen:
   - `Sex`: M=1, F=0
   - `ChestPainType`: ASY=0, ATA=1, NAP=2, TA=3
   - `RestingECG`: LVH=0, Normal=1, ST=2
   - `ExerciseAngina`: N=0, Y=1
   - `ST_Slope`: Down=0, Flat=1, Up=2

2. Kaggle-datasetet saknar två features som finns i UCI: `ca` och `thal`.
   För att bibehålla integriteten i den 13-feature pipeline imputeras dessa
   med medianvärden från det ursprungliga datasetet. Detta är ett medvetet
   metodval som dokumenteras tydligt — alternativet att ta bort dessa features
   från båda dataseten skulle försämra modellens prediktionsförmåga.

3. Fysiologiskt omöjliga nollvärden i `Cholesterol` (172 rader) och `RestingBP`
   (1 rad) ersätts med medianvärden från de giltiga värdena i Kaggle-datasetet.

4. Båda dataseten konkateneras och dubbletter tas bort. Cleveland-subsettet
   ingår redan i Kaggle-datasetet, varför dedupliceringen är viktig.

## 2. Modellval

Fyra modeller tränas och jämförs:

**Logistic Regression** — En linjärt tolkningsbar basmodell. Enkel att förstå
och förklara. Används som baseline.

**Random Forest** — Ett ensemble av beslutsträd som fångar icke-linjära samband
och är robust mot brus i data. Valdes som primärmodell baserat på högst noggrannhet
och ROC AUC.

**Decision Tree** — Inkluderades för att illustrera värdet av ensemblemetoder.
Ett enskilt beslutsträd tenderar att överpassa träningsdata. Trots lägre noggrannhet
har det fördelen av full transparens — varje beslut kan följas steg för steg,
vilket är värdefullt i medicinska sammanhang.

**XGBoost** — Tillagd i ProjectEC5 som en fjärde modell. XGBoost bygger träd
sekventiellt där varje nytt träd korrigerar felen från det föregående. Det är
industristandard för tabulärdata och matchar ofta Random Forest utan extensiv tuning.
Tränas med 200 estimatorer, `eval_metric='logloss'` och `random_state=42`.

Hyperparameterval: Random Forest tränas med 200 estimatorer och Decision Tree
med max djup 10 för att begränsa överpressning. Båda använder `random_state=42`
för reproducerbarhet.

## 3. Hyperparametertuning (ProjectEC5)

I ProjectEC5 genomfördes systematisk hyperparametertuning med GridSearchCV och 5-faldig
korsvalidering. Scoring-mått: **ROC AUC** — robust för obalanserade klasser.

### Bästa parametrar (GridSearchCV, cv=5)

| Modell | Parameter | Värde |
|--------|-----------|-------|
| Logistic Regression | C / penalty / solver | 0.1 / l2 / lbfgs |
| Random Forest | n_estimators / max_depth / max_features | 200 / 5 / sqrt |
| Random Forest | min_samples_split | 2 |
| Decision Tree | criterion / max_depth | gini / 3 |
| Decision Tree | min_samples_leaf / min_samples_split | 2 / 2 |
| XGBoost | n_estimators / eval_metric | 200 / logloss |

### Jämförelse: Default vs Tunade parametrar

| Modell | Default Acc | Tuned Acc | Default ROC AUC | Tuned ROC AUC |
|--------|-------------|-----------|-----------------|---------------|
| Logistic Regression | 0.869 | 0.853 | 0.951 | 0.958 |
| Random Forest | 0.902 | 0.902 | 0.955 | 0.958 |
| Decision Tree | 0.787 | 0.869 | 0.808 | 0.871 |
| XGBoost | 0.869 | — | 0.869 | — |

Tuningen förbättrade framför allt **Decision Tree** avsevärt (+8.2% accuracy, +6.3% ROC AUC).
Random Forest behöll sin accuracy men förbättrade ROC AUC marginellt.
XGBoost matchade Logistic Regression-baseline utan tuning.
Bästa parametrar valdes baserat på ROC AUC på valideringssettet.
Slutlig utvärdering genomfördes på ett stratifierat test-set (20%, `random_state=42`).

## 4. Resultat

Alla modeller utvärderas på ett stratifierat test-set (20%, `random_state=42`).

| Modell               | Accuracy | F1    | Precision | Recall | ROC AUC |
|----------------------|----------|-------|-----------|--------|---------| 
| Logistic Regression  | 0.869    | 0.877 | 0.861     | 0.893  | 0.935   |
| Random Forest        | 0.885    | 0.893 | 0.878     | 0.909  | 0.955   |
| Decision Tree        | 0.754    | 0.771 | 0.750     | 0.793  | 0.754   |
| XGBoost              | 0.869    | —     | —         | —      | 0.869   |

*Värden från det ursprungliga UCI Cleveland-datasetet (303 rader) — baseline ProjectEC3.*

Random Forest uppnådde bäst resultat på samtliga mätetal. I ett medicinskt
sammanhang är **Recall** det mest kritiska måttet — ett missat fall av hjärtsjukdom
(falskt negativt) är allvarligare än ett falskt larm.

XGBoost matchar Logistic Regression-baseline (ROC AUC 0.869) utan hyperparametertuning,
vilket bekräftar dess styrka för tabulärdata. Med ytterligare tuning förväntas prestandan
förbättras ytterligare.

## 5. SHAP — Förklarbarhet (ProjectEC5)

I ProjectEC5 introducerades SHAP (SHapley Additive exPlanations) för att förklara
*varför* modellen gör en specifik prediktion. SHAP kommer från spelteori och fördelar
rättvist varje features bidrag till prediktionen.

Detta adresserar direkt kritiken mot "black box"-modeller i den etiska reflektionen —
en läkare behöver förstå varför modellen förutsäger hög risk innan den agerar.

### Beeswarm-plot — Random Forest

De tre mest inflytelserika features är **thal**, **ca** och **cp** — deras punkter
sprider sig bredast längs x-axeln, vilket innebär störst påverkan på prediktionerna.

| Feature | Tolkning |
|---------|---------|
| **thal** | Låga thal-värden driver starkt mot hjärtsjukdomsprediktion. Mest inflytelserik feature. |
| **ca** | Färre stora kärl (lågt värde) ökar predikterad risk avsevärt. |
| **cp** | Vissa bröstsmärttyper (högt värde) är starka indikatorer på sjukdom. |
| **thalach** | Låg maximal hjärtfrekvens är associerad med högre predikterad risk. |
| **exang** | Anginautlöst av träning driver prediktioner mot sjukdom. |

### Waterfall-plot — enskild patient

Waterfall-ploten förklarar **en enskild prediktion** — den första högriskpatienten
i testsettet. Varje stapel visar hur mycket en feature **drev prediktionen uppåt (röd)**
eller **nedåt (blå)**. Detta gör prediktionen granskningsbar och förklarbar för kliniker.

### Beeswarm-plot — XGBoost

XGBoost visar liknande feature-ranking som Random Forest med **thal**, **ca** och **cp**
i topp. XGBoost tenderar att producera skarpare, mer koncentrerade SHAP-värden.
Skillnader i spridning jämfört med Random Forest indikerar att modellerna viktar
features olika — relevant att notera i klinisk användning.

## 6. Etisk reflektion

**Bias i datasetet:** Det ursprungliga datasetet representerar patienter där
fördelningen mellan åldrar och kön är sned — majoriteten är medelålders män.
Genom att kombinera med Kaggle-datasetet ökar representationen, men bias
kan fortfarande förekomma. Innan modellen används kliniskt bör den valideras
på ett mer representativt dataset.

**Imputering av saknade features:** Valet att imputera `ca` och `thal` med
medianvärden för Kaggle-rader introducerar en känd förenkling. Dessa värden
är inte uppmätta för dessa patienter — de är uppskattningar. Detta bör
kommuniceras tydligt om modellen används i ett beslutsstödsystem.

**Modellens tolkbarhet:** Random Forest och XGBoost är "black box"-modeller.
SHAP-analys (Section 5) adresserar detta direkt genom att förklara varje
prediktion på feature-nivå. Decision Tree erbjuder full transparens men på
bekostnad av noggrannhet.

**Användning i vården:** Modellen är ett pedagogiskt verktyg — inte ett
medicinskt diagnostikinstrument. Prediktioner ska alltid kompletteras med
kliniskt omdöme och diagnostiska tester.

## 7. Versionshantering

I ProjectEC5 introducerades Git-taggning som en del av projektets arbetsflöde.
En tagg sätts på `main` efter att alla tester är gröna och rapporten är uppdaterad,
och markerar en stabil release av projektet.

```bash
git tag -a v5.0 -m "ProjectEC5 - XGBoost, SHAP explainability, combined dataset 918 rows"
git push origin v5.0
```

Taggar syns under **Releases** på GitHub och gör det enkelt att återgå till en specifik version.

### CI/CD-strategi (ProjectEC5)

I ProjectEC5 delades CI/CD upp i två separata workflows:

**`tests.yml`** — körs vid varje push. Snabb återkoppling med fullständig testsvit och täckningskontroll (min 84%).

**`pr_checks.yml`** — körs vid varje pull request. Innehåller allt i tests.yml plus:
- **flake8** — kodstilskontroll enligt PEP8
- **bandit** — säkerhetsskanning (medium och hög allvarlighetsgrad)
- **mypy** — typhintskontroll

CI/CD-workflödet använder **Node.js 24** för GitHub Actions
(`FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true`) inför Githubs deadline den 16 juni 2026.
