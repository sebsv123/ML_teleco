# Predicción de Churn en Telecomunicaciones
### Redes Neuronales con selección de variables — MLP Classifier

---

## Descripción del problema

Este proyecto aborda la predicción de abandono de clientes (*churn*) en una empresa de telecomunicaciones. El objetivo es identificar, antes de que ocurra, qué clientes tienen mayor probabilidad de darse de baja, permitiendo al equipo de retención actuar de forma proactiva.

El problema se enmarca como una clasificación binaria con **asimetría de costes**: no detectar un cliente que va a abandonar (falso negativo) tiene un coste de negocio mayor que alertar innecesariamente sobre un cliente que no iba a irse (falso positivo). Esta premisa guía todas las decisiones metodológicas del proyecto.

---

## Metodología

Se desarrollaron y compararon **dos redes neuronales MLP** con distinto enfoque de selección de variables y optimización:

### Modelo 1 — Selección por RFE
- Selección de variables mediante *Recursive Feature Elimination* con regresión logística
- Optimización orientada a **Accuracy**
- Búsqueda de hiperparámetros con `RandomizedSearchCV` (50 iteraciones)
- Arquitectura final: `(50,)` neuronas, activación `tanh`, solver `adam`

### Modelo 2 — Selección por Árbol de Decisión
- Selección de variables mediante importancia Gini de un `DecisionTreeClassifier`
- Optimización orientada a **Recall** (maximizar detección de abandonos)
- Búsqueda refinada en dos etapas con análisis de robustez mediante boxplots CV
- Arquitectura final: `(70,)` neuronas, activación `logistic`, solver `adam`, `alpha=1e-05`

Ambos métodos de selección convergieron de forma independiente en las **mismas 5 variables**, lo que constituye una validación sólida de su relevancia predictiva:

| Variable | Interpretación |
|---|---|
| `min_diurnas` | Patrón de uso principal del cliente |
| `llamadas_atencion` | Indicador de insatisfacción |
| `coste_nocturnas` | Uso diferencial por franja horaria |
| `plan_internacional` | Perfil de cliente con necesidades específicas |
| `estado` | Dimensión geográfica (Target Encoding) |

---

## Resultados

| Métrica | Modelo RFE | Modelo Árbol |
|---|---|---|
| Accuracy (test) | 84% | 84% |
| Recall clase 1 | 0.81 | **0.88** |
| Precisión clase 1 | 0.84 | 0.81 |
| F1 clase 1 | 0.83 | **0.84** |
| Falsos negativos | 170 | **108** |
| Falsos positivos | 128 | 191 |
| AUC-ROC | 0.904 | **0.911** |
| Threshold óptimo | 0.59 | 0.37 |

El **Modelo 2 es el recomendado para producción**: captura 62 clientes adicionales en riesgo real de abandono, con mayor AUC-ROC y un threshold elegido mediante análisis explícito del trade-off entre Recall, Recall macro y F1.

---

## Stack tecnológico

```
Python 3.x
pandas · numpy · matplotlib · seaborn · missingno
scikit-learn (MLPClassifier, RFE, DecisionTreeClassifier, RandomizedSearchCV,
              StandardScaler, OneHotEncoder, OrdinalEncoder, IterativeImputer)
```

---

## Estructura del repositorio

```
ML_teleco/
├── Entrega_ML_Sebastian_Sifontes.ipynb   # Notebook principal con análisis completo
└── .gitignore
```

---

## Cómo ejecutar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/sebsv123/ML_teleco.git
   ```
2. Instala las dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn missingno
   ```
3. Abre el notebook en Jupyter y ejecuta las celdas en orden. El dataset es necesario para reproducir los resultados — al ser de uso académico no está incluido en el repositorio.

---

## Autor

**Sebastián Sifontes**  
[GitHub](https://github.com/sebsv123)
