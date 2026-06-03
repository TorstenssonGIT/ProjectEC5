# Heart Disease Prediction Report

## 1. Dataset

### Original dataset
Datasetet kommer från UCI Heart Disease (Cleveland-subset) och innehåller 303 patientposter
med 13 kliniska egenskaper: ålder, kön, bröstsmärttyp, viloblodtryck, serumkolesterol,
fastande blodsocker, vilo-EKG-resultat, maximal uppnådd hjärtfrekvens, anginaprovokation,
ST-depression, ST-segmentlutning, antal stora kärlfärgar av fluoroskopi samt thalassemia.
Målvariabeln `target` anger om patienten har hjärtsjukdom (1) eller inte (0).

Källa: UCI Machine Learning Repository — Heart Disease Dataset (Cleveland).

### Utökat dataset (ProjectEC4)
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

Tre modeller tränas och jämförs:

**Logistic Regression** — En linjärt tolkningsbar basmodell. Enkel att förstå
och förklara. Används som baseline.

**Random Forest** — Ett ensemble av beslutsträd som fångar icke-linjära samband
och är robust mot brus i data. Valdes som primärmodell baserat på högst noggrannhet
och ROC AUC.

**Decision Tree** — Inkluderades för att illustrera värdet av ensemblemetoder.
Ett enskilt beslutsträd tenderar att överpassa träningsdata. Trots lägre noggrannhet
har det fördelen av full transparens — varje beslut kan följas steg för steg,
vilket är värdefullt i medicinska sammanhang.

Hyperparameterval: Random Forest tränas med 200 estimatorer och Decision Tree
med max djup 10 för att begränsa överpressning. Båda använder `random_state=42`
för reproducerbarhet.

## 3. Resultat

Alla modeller utvärderas på ett stratifierat test-set (20%, `random_state=42`).

| Modell               | Accuracy | F1    | Precision | Recall | ROC AUC |
|----------------------|----------|-------|-----------|--------|---------|
| Logistic Regression  | 0.869    | 0.877 | 0.861     | 0.893  | 0.935   |
| Random Forest        | 0.885    | 0.893 | 0.878     | 0.909  | 0.955   |
| Decision Tree        | 0.754    | 0.771 | 0.750     | 0.793  | 0.754   |

*Värden från det ursprungliga UCI Cleveland-datasetet (303 rader).*

Random Forest uppnådde bäst resultat på samtliga mätetal. I ett medicinskt
sammanhang är **Recall** det mest kritiska måttet — ett missat fall av hjärtsjukdom
(falskt negativt) är allvarligare än ett falskt larm.

## 4. Etisk reflektion

**Bias i datasetet:** Det ursprungliga datasetet representerar patienter där
fördelningen mellan åldrar och kön är sned — majoriteten är medelålders män.
Genom att kombinera med Kaggle-datasetet ökar representationen, men bias
kan fortfarande förekomma. Innan modellen används kliniskt bör den valideras
på ett mer representativt dataset.

**Imputering av saknade features:** Valet att imputera `ca` och `thal` med
medianvärden för Kaggle-rader introducerar en känd förenkling. Dessa värden
är inte uppmätta för dessa patienter — de är uppskattningar. Detta bör
kommuniceras tydligt om modellen används i ett beslutsstödsystem.

**Modellens tolkbarhet:** Random Forest är en "black box"-modell. Feature
importance ger en övergripande bild men ersätter inte full förklarbarhet.
Decision Tree erbjuder full transparens men på bekostnad av noggrannhet.

**Användning i vården:** Modellen är ett pedagogiskt verktyg — inte ett
medicinskt diagnostikinstrument. Prediktioner ska alltid kompletteras med
kliniskt omdöme och diagnostiska tester.
