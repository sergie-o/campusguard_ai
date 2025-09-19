# CampusGuard AI 🛡️ | Full-Stack ML Project

> **AI-Powered Student Mental Health Platform**  
> End-to-End Machine Learning Solution with 98% Prediction Accuracy

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20CatBoost-orange)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-SQLite-green)](https://www.sqlite.org/)

## 🎯 Project Overview

**Challenge**: University students face a mental health crisis with 50%+ reporting high stress levels, yet current support systems are reactive rather than predictive.

**Solution**: Built an end-to-end AI platform that predicts student burnout risk before crisis occurs using a novel dual-model machine learning architecture.

**Impact**: Achieved 98% prediction accuracy on clinical data, with potential to reduce crisis interventions by 75% through early detection.

---

## 💻 Technical Implementation

### Core Architecture

**Dual-Model ML System** - Designed and implemented two complementary models:

1. **Direct Burnout Model (Logistic Regression)**
   - 98% accuracy on 282 medical student dataset
   - Uses validated MBI-SS psychological framework
   - 7-feature model with scientific rigor

2. **Proxy Stress Model (CatBoost)**
   - Real-world deployment model using lifestyle indicators
   - 1,100+ student dataset with 20 behavioral features
   - Gradient boosting with class balancing and early stopping

### Key Technical Achievements
```python
def calculate_burnout_score(exhaustion, cynicism, academic_efficacy):
    """
    Custom risk scoring algorithm combining multiple psychological factors
    """
    # Feature normalization and composite scoring
    exhaustion_norm = normalize(exhaustion, 0, 5)
    cynicism_norm = normalize(cynicism, 0, 5)
    efficacy_inverted = 1 - normalize(academic_efficacy, 0, 5)
    
    # Weighted composite score
    return (exhaustion_norm + cynicism_norm + efficacy_inverted) / 3
