# CampusGuard AI 🛡️ | Full-Stack ML Project  

> **ML-Driven Student Wellbeing Guardian**  
> End-to-End AI Solution for Predicting Burnout & Stress  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)  
[![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20CatBoost%20%7C%20Prophet-orange)](https://scikit-learn.org/)  
[![Database](https://img.shields.io/badge/Database-SQLite-green)](https://www.sqlite.org/)  

---

## 📖 The Story  

> University life isn’t just about passing exams — it’s about balancing studies, mental health, and finances. Too often, students only realize they’re stressed, burned out, or financially stretched **after it’s already too late**.  

**CampusGuard AI** changes that.  
It acts as a *digital guardian*, predicting burnout and stress early, tracking wellbeing, and guiding students with data-driven recommendations.  

<p align="center">
  <img src="https://github.com/sergie-o/campusguard_ai/blob/main/feature_postimage_linkedin.png" alt="Banner" width="900"/>
</p>  


---

## ❌ The Problem → ✅ The Solution  

![CampusGuard AI Flow](sandbox:/mnt/data/69CECD9B-2ABF-4FEE-A4DA-0E368152B274.png?_chatgptios_conversationID=68d0ffe7-85b0-8329-8554-a59162ae2d89&_chatgptios_messageID=55ac749c-5989-4d9f-8e86-58dad0c19044)  

- **Problem**: Student wellbeing is at risk due to **burnout, stress, financial pressure, and study-life imbalance**.  
- **Solution**: CampusGuard AI integrates **Machine Learning + GenAI** to predict risks, assist with solutions, and improve wellbeing.  

---

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
    return (exhaustion_norm + cynicism_norm + efficacy_inverted) / 3 / 3
