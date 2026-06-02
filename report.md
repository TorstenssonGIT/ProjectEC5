# Heart Disease Prediction Report

## Dataset
Datasetet kommer från UCI Heart Disease och innehåller 13 kliniska egenskaper såsom ålder, kön, blodtryck, kolesterol, maxpuls och thalassemia. Målvariabeln `target` anger om patienten har hjärtsjukdom (1) eller inte (0).

## Modellval
Två modeller tränades: Logistic Regression och Random Forest. Random Forest valdes som bästa modell baserat på högre noggrannhet och ROC AUC. Random Forest fångar icke-linjära samband och är robust för blandade numeriska variabler.

## Resultat
Random Forest levererar stabil prediktionsprestanda för datasetet. Utvärderingsmetoder inkluderar:
- Accuracy
- F1-score
- Precision
- Recall
- ROC AUC
- Confusion matrix

## Etiskt resonemang
Modelldatabasen representerar patienter där fördelningen mellan åldrar och kön kan vara sned. Prediktionsmodeller för hjärtsjukdom bör därför inte användas som ensam beslutsgrund i vården. Det är viktigt att modellen kompletteras med kliniskt omdöme och diagnostiska tester. Felaktiga prediktioner kan leda till onödig oro, överbehandling eller att en patient missas.

Dessutom kan modellen förstärka befintliga biaser om datasetet inte inkluderar tillräckligt många underrepresenterade grupper. Därför är det etiskt viktigt att tydligt kommunicera modellens begränsningar och använda den som stöd, inte ersättning, till medicinsk expertis.
