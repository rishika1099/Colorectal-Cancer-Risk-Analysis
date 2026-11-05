# 🚬 CRC Risk Prediction Dashboard

An interactive Shiny web application for predicting Colorectal Cancer (CRC) risk using logistic regression with real-time analytics, personalized visualizations, and what-if scenario testing.

---

## 📊 Overview

This dashboard provides healthcare professionals, researchers, and individuals with a comprehensive tool to assess CRC risk based on demographic, lifestyle, and dietary factors. The application features six interactive visualizations, input validation, and evidence-based healthy ranges from USDA/NIH guidelines.

---

## ✨ Key Features

### **Real-Time Risk Prediction**
- Logistic regression model trained on a 1,000-participant dataset
- Binary classification (Low Risk / High Risk) with probability scores
- Instant predictions with comprehensive validation

### **Six Interactive Visualizations**

#### **1. Risk Gauge**
- Color-coded speedometer (🎉 Green: 0-30%, ⚠️ Yellow: 30-70%, 🌡️ Red: 70-100%)
- Visual risk assessment with dynamic emoji indicators
- Reference threshold at 50%

#### **2. Nutrient Balance Assessment**
- Bar chart showing percentage deviation from healthy ranges
- Color-coded severity (Green: in range, Yellow: <20% off, Red: >20% off)
- Sex-specific ranges for Vitamin A, Vitamin C, and Iron

#### **3. Risk Prediction Simulator**
- Interactive what-if testing for individual features
- Real-time risk recalculation with impact assessment
- Dynamic feedback with emojis (🎉 Favorable, ⚠️ Minimal, 🌡️ Unfavorable)

#### **4. Risk Contribution Breakdown**
- Donut chart showing factor-by-factor contribution percentages
- Identifies which factors drive YOUR specific risk
- Includes all demographic, lifestyle, and dietary variables

#### **5. Risk Trajectory**
- Line chart tracking risk changes across multiple predictions
- Trend analysis (Improving, Stable, Increasing)
- Dynamic y-axis scaling for clear visualization

#### **6. You vs Low-Risk Profile**
- Normalized comparison to median values from 845 low-risk participants
- Side-by-side bar chart for easy comparison
- Highlights areas for improvement

---

## 🎯 Input Variables

### **Demographics**
- **Age**: 18-90 years
- **Gender**: Male / Female
- **Ethnicity**: African American, Asian, Caucasian, Hispanic

### **Lifestyle**
- **Lifestyle**: Active, Moderate Exercise, Sedentary, Smoker
- **BMI**: 15-45 (healthy range: 18.5-24.9)
- **Family History**: Yes / No (CRC in family)
- **Pre-existing Conditions**: Diabetes, Hypertension, Obesity, None

### **Daily Dietary Intake**
- **Carbohydrates**: 50-600 g/day (healthy: 225-325)
- **Proteins**: 20-250 g/day (healthy: 50-100)
- **Fats**: 20-200 g/day (healthy: 44-77)
- **Vitamin A**: 1,000-15,000 IU/day (healthy: 2,000-3,000)
- **Vitamin C**: 10-500 mg/day (healthy: 75-200)
- **Iron**: 5-60 mg/day (healthy: 8-18)

---

## 🚀 Quick Start

### **Prerequisites**

```r
# Required R packages
install.packages(c("shiny", "readr", "dplyr", "plotly"))
```

### **Installation**

1. **Download the files**
```
project/
├── app.R              # Main application
└── crc_dataset.csv    # Dataset (1,000 participants)
```

2. **Run the application**
```r
# In R console or RStudio
shiny::runApp("app.R")

# Or click "Run App" button in RStudio
```

3. **Access the dashboard**
- Opens automatically in your default browser
- URL: `http://127.0.0.1:####` (port number varies)

---

## 💡 How to Use

### **Step 1: Enter Participant Details**

Fill in all fields in the left panel. **Grey placeholders show healthy ranges**:

```
Age: 18-90
BMI: 18.5-24.9
Carbohydrates (g): 225-325
Proteins (g): 50-100
Fats (g): 44-77
Vitamin A (IU): 2000-3000
Vitamin C (mg): 75-200
Iron (mg): 8-18
```

Select options from dropdowns:
- Gender, Lifestyle, Ethnicity
- Family History, Pre-existing Conditions

### **Step 2: Click "Predict Risk"**

The system will:
- ✅ Validate all inputs (shows error if values are unrealistic)
- ✅ Calculate risk probability using logistic regression
- ✅ Update all six visualizations instantly

### **Step 3: Interpret Results**

**Prediction Result Box:**
```
🎉 Predicted CRC risk probability: 0.0342 (3.42%)
Class (cutoff 0.5): Low risk
```

**Risk Levels:**
- **0-30% (🎉 Green)**: Low risk - Continue healthy habits
- **30-70% (⚠️ Yellow)**: Moderate risk - Consider lifestyle changes  
- **70-100% (🌡️ Red)**: High risk - Consult healthcare provider

### **Step 4: Test What-If Scenarios**

Use the Risk Prediction Simulator:
1. Select a feature (e.g., BMI, Carbohydrates)
2. Adjust the slider to test different values
3. See immediate impact on risk

Example output:
```
Current Risk: 3.4%
Predicted Risk: 2.8%
Change: -0.6% (-17.6%)
Impact: 🎉 Lower risk (Favorable)
```

### **Step 5: Track Progress**

Make multiple predictions to:
- See risk trend over time in Risk Trajectory chart
- Monitor effectiveness of interventions
- Set and track health goals

---

## 📋 Example Test Cases

### **Profile 1: Healthy Young Adult (Low Risk ~3-5%)**
```
Age: 28               Carbohydrates: 280
Gender: Female        Proteins: 65
BMI: 22.5            Fats: 55
Lifestyle: Active     Vitamin A: 2400
Ethnicity: Caucasian  Vitamin C: 90
Family History: No    Iron: 16
Pre-existing: None
```

### **Profile 2: Moderate Risk (~20-30%)**
```
Age: 48               Carbohydrates: 320
Gender: Male          Proteins: 85
BMI: 27.5            Fats: 75
Lifestyle: Sedentary  Vitamin A: 3200
Ethnicity: Asian      Vitamin C: 65
Family History: No    Iron: 10
Pre-existing: None
```

### **Profile 3: High Risk (~65-80%)**
```
Age: 62               Carbohydrates: 380
Gender: Male          Proteins: 95
BMI: 31              Fats: 110
Lifestyle: Smoker     Vitamin A: 7500
Ethnicity: African American  Vitamin C: 45
Family History: Yes   Iron: 9
Pre-existing: Diabetes
```

---

## 🔬 Model Details

### **Algorithm**
Logistic Regression (GLM with binomial family)

```r
CRC_Risk ~ Age + BMI + Carbohydrates + Proteins + Fats + 
           Vitamin A + Vitamin C + Iron + Gender + Lifestyle + 
           Ethnicity + Family_History_CRC + Pre-existing Conditions
```

### **Dataset**
- **Total participants**: 1,000
- **Low-risk**: 845 (84.5%)
- **High-risk**: 155 (15.5%)
- **Features**: 15 variables (8 numeric, 7 categorical)
- **Missing data**: 1.59% (only in Pre-existing Conditions)

### **Prediction**
- **Threshold**: 0.5 (50%)
- Probability ≥ 0.5 → High Risk
- Probability < 0.5 → Low Risk

---

## 📚 Evidence-Based Healthy Ranges

Based on **USDA Dietary Guidelines (2020-2025)** and **NIH Recommended Dietary Allowances (RDA)**:

| Nutrient | Healthy Range | Source |
|----------|---------------|--------|
| **Age** | 18-90 years | Model training range |
| **BMI** | 18.5-24.9 | CDC/WHO guidelines |
| **Carbohydrates** | 225-325 g/day | USDA (45-65% of 2000 kcal) |
| **Proteins** | 50-100 g/day | RDA (varies by activity) |
| **Fats** | 44-77 g/day | USDA (20-35% of calories) |
| **Vitamin A** | 2,000-3,000 IU/day | NIH RDA (UL: 10,000 IU) |
| **Vitamin C** | 75-200 mg/day | NIH RDA + immunity support |
| **Iron** | 8-18 mg/day | NIH RDA (sex-specific) |

**Sex-Specific Ranges:**
- Vitamin A: Men 3,000 IU, Women 2,300 IU
- Vitamin C: Men 90 mg, Women 75 mg
- Iron: Men 8 mg, Women 18 mg

---

## ⚠️ Input Validation

The app validates all inputs before prediction:

### **Validation Rules**
```
Age: 18-90 years
BMI: 15-45
Carbohydrates: 50-600 g/day
Proteins: 20-250 g/day
Fats: 20-200 g/day
Vitamin A: 1,000-15,000 IU/day
Vitamin C: 10-500 mg/day
Iron: 5-60 mg/day
```

**If validation fails**, you'll see:
```
⚠️ Invalid Input Values

Please enter realistic values:
• Age must be between 18-90 years
• Proteins must be between 20-250 g/day

Tip: These ranges reflect normal adult dietary intake.
```

---

## 📖 Use Cases

### **Clinical Settings**
- Patient risk screening
- Doctor-patient consultations
- Preventive health assessments
- Diet and lifestyle counseling

### **Research**
- Population health studies
- Risk factor analysis
- Intervention effectiveness testing
- Health education programs

### **Personal Health**
- Self-assessment and monitoring
- Goal setting and tracking
- Understanding risk factors
- Testing lifestyle modifications

---

## ⚠️ Important Disclaimers

### **Medical Disclaimer**
This tool is for **educational and informational purposes only**. It is **NOT a substitute for professional medical advice, diagnosis, or treatment**.

**Always consult qualified healthcare providers for:**
- Medical conditions and concerns
- Risk assessment and screening
- Treatment decisions
- Diet and lifestyle recommendations

### **Limitations**
- Model trained on specific dataset (may not generalize to all populations)
- Does not account for all CRC risk factors (e.g., genetics, environment)
- Binary classification simplifies complex health conditions
- Healthy ranges are general guidelines (individual needs vary)

### **Data Privacy**
- All processing occurs locally in your browser
- No data is stored or transmitted externally
- Session data cleared when browser closes
- Safe for sensitive health information

---

## 🐛 Troubleshooting

### **Issue: App won't start**
```
Error in library(shiny) : there is no package called 'shiny'
```
**Solution:**
```r
install.packages(c("shiny", "readr", "dplyr", "plotly"))
```

### **Issue: Data file not found**
```
Error: 'crc_dataset.csv' does not exist
```
**Solution:**
- Ensure `crc_dataset.csv` is in the same directory as `app.R`
- Check file spelling and capitalization

### **Issue: Column name errors**
```
Error: could not find column "Carbohydrates (g)"
```
**Solution:**
- Make sure you're using `read_csv()` from readr (not base R `read.csv()`)
- OR use: `read.csv("crc_dataset.csv", check.names = FALSE)`

### **Issue: Prediction always shows 0%**
**Cause:** Unrealistic input values (e.g., 122,134g protein)
**Solution:** 
- Use realistic daily totals (see Example Test Cases above)
- Follow placeholder ranges shown in grey text
- Validation will now catch these errors

### **Issue: Plotly warnings in console**
```
Warning: No trace type specified
```
**Solution:** These are harmless and don't affect functionality. Already suppressed in current version.

---

## 📁 File Structure

```
project/
├── app.R                           # Main Shiny application (this file)
├── crc_dataset.csv                 # Dataset (1,000 participants, 15 features)
└── README.md                       # This documentation
```

**Required files:**
- `app.R` - Contains UI and server logic
- `crc_dataset.csv` - Training data for model

---

