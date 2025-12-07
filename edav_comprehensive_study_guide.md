# EDAV COMPREHENSIVE EXAM STUDY GUIDE

## TABLE OF CONTENTS
1. Univariate Continuous Data
2. Categorical Data Visualization
3. Multivariate Data
4. Missing Data Analysis
5. Linear Regression
6. Logistic Regression
7. Decision Trees
8. Interpretable Machine Learning (IML)
9. Time Series & Dates
10. Spatial Data
11. Color Theory
12. ggplot2 Grammar of Graphics

---

# 1. UNIVARIATE CONTINUOUS DATA

## 1.1 HISTOGRAMS

### Definition & Purpose
- Display distribution of single continuous variable
- Show frequency or density of observations within bins
- Identify: shape, center, spread, outliers, modality

### Types of Histograms

#### Frequency Histogram
- Y-axis = count of observations in each bin
- Bar height = number of observations
- Sum of all bar heights = total number of observations

#### Density Histogram  
- Y-axis = density (proportion per unit width)
- Area of each bar = relative frequency = binwidth × density
- **Sum of all bar AREAS = 1** (not heights!)
- Formula: Area = binwidth × density = relative frequency

### Key Histogram Concepts

**Binwidth Selection:**
- Too narrow → too much noise, overfitting
- Too wide → lose detail, underfitting
- No single "correct" binwidth
- Sturges' rule: k = log₂(n) + 1 bins
- Try multiple binwidths to explore data

**Equal vs Unequal Binwidths:**
- Equal binwidths: easier to interpret visually
- Unequal binwidths: useful for skewed data, must use density histogram
- With unequal bins, NEVER use frequency histogram (misleading)

**What to Look For:**
- Shape: symmetric, skewed (left/right), uniform
- Modality: unimodal, bimodal, multimodal
- Outliers: values far from main distribution
- Gaps: breaks in distribution
- Center: where data concentrates
- Spread: range and variability

### ggplot2 Implementation
```r
# Frequency histogram
ggplot(data, aes(x = variable)) + 
  geom_histogram(binwidth = 5)

# Density histogram
ggplot(data, aes(x = variable, y = after_stat(density))) + 
  geom_histogram(binwidth = 5)

# Custom breaks
ggplot(data, aes(x = variable)) + 
  geom_histogram(breaks = seq(0, 100, by = 10))
```

---

## 1.2 BOXPLOTS

### Structure & Components
- **Box**: IQR (Interquartile Range) = Q3 - Q1
- **Line in box**: median (Q2)
- **Whiskers extend to**: 
  - Lower: Q1 - 1.5×IQR or minimum value
  - Upper: Q3 + 1.5×IQR or maximum value
- **Outliers**: points beyond whiskers

### Boxplot Statistics
```r
boxplot.stats(data)$stats  # Returns: min, Q1, median, Q3, max
quantile(data)  # 0%, 25%, 50%, 75%, 100%
```

**Note:** ggplot2 and base R calculate whiskers slightly differently

### Uses
- Compare distributions across groups
- Identify outliers
- Quick summary of center and spread
- Less detail than histogram (no shape information)

### Limitations
- Doesn't show distribution shape (bimodal looks same as unimodal)
- Hides sample size
- Multiple modes not visible

### Comparison with Histograms
- Boxplot: better for comparisons, outlier detection
- Histogram: better for seeing full distribution shape

---

## 1.3 DENSITY CURVES

### Purpose
- Smooth estimate of probability density
- Shows overall shape without binning artifacts
- Useful for comparing distributions

### Types

#### Kernel Density Estimate (KDE)
- Smooth, continuous curve
- Bandwidth parameter controls smoothness (like binwidth in histogram)
- Area under curve = 1

```r
ggplot(data, aes(x = variable)) + 
  geom_density(adjust = 1)  # adjust controls bandwidth
```

#### Violin Plots
- Rotated, mirrored density plot
- Shows full distribution shape
- Better than boxplots for multimodal data
- Combines density information with summary statistics

```r
ggplot(data, aes(x = group, y = value)) + 
  geom_violin()
```

---

## 1.4 RIDGELINE PLOTS (Joy Plots)

### Purpose
- Compare distributions across many categories
- Vertically stacked density curves
- Shows both individual distributions and patterns across groups

### Uses
- Time series of distributions
- Comparing multiple groups
- When you have too many groups for faceted histograms

### Implementation
```r
library(ggridges)
ggplot(data, aes(x = value, y = category)) + 
  geom_density_ridges()
```

### Comparison with Boxplots
- Ridgeline: shows full distribution shape
- Boxplot: more compact, better for many groups
- Ridgeline shows bimodality, skewness better

---

## 1.5 QQ PLOTS (Quantile-Quantile Plots)

### Purpose
- Check if data follows theoretical distribution (usually normal)
- Compare two distributions

### Interpretation
- **Straight diagonal line**: data matches theoretical distribution
- **Curved upward at ends**: heavy tails (outliers)
- **Curved downward at ends**: light tails
- **S-shaped**: skewed distribution
- **Points above line at right**: right-skewed
- **Points below line at right**: left-skewed

### ggplot2 Implementation
```r
ggplot(data, aes(sample = variable)) + 
  stat_qq() + 
  stat_qq_line()
```

---

# 2. CATEGORICAL DATA VISUALIZATION

## 2.1 BAR CHARTS

### Types

#### Frequency Bar Chart
- Height = count of observations
- geom_bar() with default stat = "count"
```r
ggplot(data, aes(x = category)) + 
  geom_bar()
```

#### Value Bar Chart
- Height = pre-calculated value
- geom_bar(stat = "identity") or geom_col()
```r
ggplot(data, aes(x = category, y = value)) + 
  geom_col()
```

### Best Practices
- Order bars by frequency (descending) unless natural order exists
- Use horizontal bars for long category names
- Start y-axis at zero (never truncate)
- Add value labels if exact values matter

### Ordering Bars
```r
# By frequency
ggplot(data, aes(x = fct_infreq(category))) + geom_bar()

# By another variable
ggplot(data, aes(x = fct_reorder(category, value))) + geom_bar()

# Reverse order
ggplot(data, aes(x = fct_rev(fct_infreq(category)))) + geom_bar()
```

---

## 2.2 CLEVELAND DOT PLOTS

### Purpose
- Alternative to bar charts
- Better for many categories
- Easier to read precise values
- Less "chart junk" than bars

### Uses
- Ranked data
- When precise values matter
- When you have many categories (>10)

### Implementation
```r
ggplot(data, aes(x = value, y = fct_reorder(category, value))) + 
  geom_point(size = 3)
```

### Advantages over Bar Charts
- More data-ink ratio (Tufte principle)
- Easier to compare values
- Works better with many categories
- Don't need to start at zero

---

## 2.3 MOSAIC PLOTS

### Definition
- Filled rectangular plot showing associations between categorical variables
- Area of each rectangle ∝ frequency count
- No white space between rectangles
- Consistent number of rows and columns

### Purpose
- Visualize contingency tables
- Show associations between 2+ categorical variables
- Compare proportions across groups

### Types

#### Standard Mosaic Plot
- Variable widths AND heights
- Both dimensions convey information
- Width typically = marginal proportion of first variable
- Height = conditional proportion given first variable

#### Same Bin Width Mosaic Plot
- All columns same width (or all rows same height)
- Equivalent to stacked bar chart
- Easier to compare proportions within groups

### Interpretation
- **Equal width across groups**: no association with first variable
- **Varying heights**: association exists
- **Area**: joint probability
- **Width**: marginal probability of column variable
- **Height**: conditional probability given column

### Creating Mosaic Plots
```r
# Using vcd package
library(vcd)
mosaic(~ Var1 + Var2, data = data)

# Using ggplot2 (stacked bar approximation)
ggplot(data, aes(x = Var1, fill = Var2)) + 
  geom_bar(position = "fill")
```

### Three Variables
- Use faceting or nested rectangles
- Read carefully - complex to interpret
- Variables: x-axis, fill, facets

---

## 2.4 OTHER CATEGORICAL PLOTS

### Grouped Bar Charts
- Bars side-by-side within categories
- Compare values across two categorical variables
- Easier to compare within subgroups than stacked

```r
ggplot(data, aes(x = Cat1, fill = Cat2)) + 
  geom_bar(position = "dodge")
```

### Stacked Bar Charts
- Bars stacked on top of each other
- Show composition and total
- Hard to compare middle segments

```r
ggplot(data, aes(x = category, fill = subcategory)) + 
  geom_bar(position = "stack")
```

### Proportional Stacked Bar Charts
- All bars same height (100%)
- Compare proportions across groups
- Lose information about totals

```r
ggplot(data, aes(x = category, fill = subcategory)) + 
  geom_bar(position = "fill")
```

---

## 2.5 ALLUVIAL DIAGRAMS (SANKEY DIAGRAMS)

### Purpose
- Show flow/connections between categorical variables
- Visualize changes over time or across categories
- Display proportions and transitions

### Uses
- Survey responses across multiple questions
- Changes in categorical status over time
- Flow of people/items through categories

### Implementation
```r
library(ggalluvial)
ggplot(data, aes(axis1 = Cat1, axis2 = Cat2, axis3 = Cat3, y = Freq)) +
  geom_alluvium(aes(fill = Cat1)) +
  geom_stratum() +
  geom_text(stat = "stratum", aes(label = after_stat(stratum)))
```

### Reading Alluvial Diagrams
- Width of stream = frequency/count
- Follow flows to see transitions
- Color typically tracks one variable across axes

---

# 3. MULTIVARIATE DATA

## 3.1 TWO CONTINUOUS VARIABLES

### Scatterplots

#### Purpose
- Show relationship between two continuous variables
- Identify: correlation, patterns, outliers, clusters

#### What to Look For
- **Direction**: positive, negative, none
- **Strength**: strong, moderate, weak
- **Form**: linear, curved, none
- **Outliers**: points far from pattern
- **Clusters**: distinct groups
- **Heteroscedasticity**: non-constant variance
- **Conditional relationships**: relationship changes with x

#### Correlation vs Causation
- Correlation ≠ causation
- Always put dependent variable on y-axis (if known)
- Scatterplots show associations, not causes

#### Enhancements
```r
# Basic scatterplot
ggplot(data, aes(x = var1, y = var2)) + 
  geom_point()

# Add smooth line
ggplot(data, aes(x = var1, y = var2)) + 
  geom_point() + 
  geom_smooth(method = "lm")  # or method = "loess"

# Transparency for overplotting
ggplot(data, aes(x = var1, y = var2)) + 
  geom_point(alpha = 0.3)

# 2D density
ggplot(data, aes(x = var1, y = var2)) + 
  geom_density_2d()
```

### LOESS Smoothers

#### Purpose
- Local polynomial regression
- Show non-linear trends
- Flexible, doesn't assume global function form

#### Interpretation
- **Overfitted**: follows individual points too closely, wiggly
- **Underfitted**: too smooth, misses important patterns
- **Just right**: captures general trend without noise

#### Parameters
- `span`: controls smoothness (larger = smoother)
- Default usually works well
- If too wiggly (overfitted): increase span
- If too smooth (underfitted): decrease span

---

## 3.2 CONTINUOUS + CATEGORICAL

### Faceted Plots
- Multiple small plots, one per category
- Same scales for easy comparison

```r
ggplot(data, aes(x = continuous)) + 
  geom_histogram() + 
  facet_wrap(~ category)

# Grid layout
ggplot(data, aes(x = continuous)) + 
  geom_histogram() + 
  facet_grid(rows = vars(cat1), cols = vars(cat2))
```

### Side-by-Side Boxplots
- Compare distributions across categories
- Excellent for many groups

```r
ggplot(data, aes(x = category, y = continuous)) + 
  geom_boxplot()
```

### Overlapping Density Curves
- Show distributions on same plot
- Use transparency and different colors

```r
ggplot(data, aes(x = continuous, fill = category)) + 
  geom_density(alpha = 0.5)
```

### Ridgeline Plots
- Many categories
- Shows full distribution shapes
- Better than overlapping densities for many groups

---

## 3.3 MULTIPLE CONTINUOUS VARIABLES

### Scatterplot Matrix (SPLOM)
- All pairwise scatterplots
- Diagonal often shows distributions

```r
# Base R
plot(data[, c("var1", "var2", "var3")])

# GGally package
library(GGally)
ggpairs(data)
```

### Parallel Coordinates Plot
- Each variable is a vertical axis
- Each observation is a line connecting points
- Good for seeing overall patterns and clusters

```r
library(GGally)
ggparallel(data, columns = 1:5)
```

### 3D Scatterplots
- Third variable as color, size, or shape
- Or actual 3D plot (but 2D usually better)

```r
ggplot(data, aes(x = var1, y = var2, color = var3)) + 
  geom_point()
```

---

## 3.4 BIPLOTS

### Purpose
- Visualize multivariate data (PCA results)
- Show observations AND variables simultaneously
- Reduce dimensionality from many variables to 2D

### Components
- **Points**: observations (rows)
- **Arrows**: variables (columns)
- **Arrow direction**: how variable increases
- **Arrow length**: how much variable contributes to PCs
- **Arrow angle**: correlation between variables
  - Small angle (near parallel): positive correlation
  - ~90°: uncorrelated
  - ~180° (opposite): negative correlation

### Interpretation
- Points close together: similar observations
- Point in direction of arrow: high on that variable
- Long arrows: variable explains more variance

### Implementation
```r
pca_result <- prcomp(data, scale. = TRUE)
biplot(pca_result)

# Or ggplot2 via ggfortify
library(ggfortify)
autoplot(pca_result, loadings = TRUE, loadings.label = TRUE)
```

---

# 4. MISSING DATA ANALYSIS

## 4.1 TYPES OF MISSING DATA

### Missing Completely at Random (MCAR)
- Missingness unrelated to any variable
- Rare in practice
- Complete case analysis is unbiased (but inefficient)

### Missing at Random (MAR)
- Missingness related to observed variables, not unobserved
- Can be adjusted for in analysis
- Most common assumption

### Missing Not at Random (MNAR)
- Missingness related to unobserved values
- Most problematic
- Example: high earners don't report income

---

## 4.2 VISUALIZATION METHODS

### Missing Value Heatmap (mi package)

```r
library(mi)
x <- missing_data.frame(data)
image(x)
```

#### What to Look For
- **Row patterns**: Do certain rows have lots of missing values?
- **Column patterns**: Which variables have most missing?
- **Associations**: Is missing in one variable related to missing in another?
- **Value associations**: Is missing in one variable related to VALUES in another?

### Using geom_tile()

```r
# Create missing indicator
tidy_data <- data %>%
  rownames_to_column("id") %>%
  pivot_longer(cols = -id) %>%
  mutate(missing = ifelse(is.na(value), "yes", "no"))

# Plot
ggplot(tidy_data, aes(x = name, y = fct_rev(id), fill = missing)) +
  geom_tile(color = "white") +
  scale_fill_manual(values = c("yes" = "black", "no" = "grey80"))
```

### Aggregated Missing Patterns

```r
library(redav)
plot_missing(data)  # Shows unique patterns and their frequencies
```

#### Interpretation
- Each row = one unique missing pattern
- Width = how many rows have this pattern
- Can set percent = TRUE or FALSE
- Helps identify structural missingness

---

## 4.3 KEY QUESTIONS FOR MISSING DATA

1. **What percentage of data is missing overall?**
2. **Which variables have the most missing values?**
3. **Are there missing patterns?** (certain combinations always missing together)
4. **Are certain observations mostly missing?** (should they be removed?)
5. **Is missingness associated with other variables?**
   - Do complete/incomplete cases differ on observed variables?
   - Does having X missing predict having Y missing?
6. **Why are values missing?** (MCAR, MAR, MNAR?)

---

## 4.4 HANDLING MISSING DATA

### Complete Case Analysis (Listwise Deletion)
- Remove any row with any missing value
- Simple but loses data
- Unbiased only if MCAR
- Standard in many software packages

### Pairwise Deletion
- Use all available data for each analysis
- Sample size varies across analyses
- Can give inconsistent results

### Imputation
- Fill in missing values with estimates
- Single imputation: one value per missing
- Multiple imputation: multiple datasets with different imputed values
- More sophisticated, preserves data

### Model-Based Methods
- Use likelihood-based methods that handle missing data
- Mixed models, maximum likelihood
- Assumes MAR

---

# 5. LINEAR REGRESSION

## 5.1 SIMPLE LINEAR REGRESSION

### Model
y = β₀ + β₁x + ε

Where:
- y = response (dependent variable)
- x = predictor (independent variable)
- β₀ = intercept
- β₁ = slope
- ε = error term

### Assumptions
1. **Linearity**: relationship between x and y is linear
2. **Independence**: errors are independent
3. **Homoscedasticity**: constant variance of errors
4. **Normality**: errors normally distributed

**Most important: Linearity**  
- If violated, predictions will be wrong
- Can proceed even if other assumptions violated (with caution)

---

## 5.2 LEAST SQUARES ESTIMATION

### Goal
Minimize sum of squared residuals:

Minimize: Σ(yᵢ - ŷᵢ)²

Where:
- yᵢ = observed value
- ŷᵢ = predicted value = β₀ + β₁xᵢ
- Residual eᵢ = yᵢ - ŷᵢ

### Solution
- β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
- β₀ = ȳ - β₁x̄

---

## 5.3 MODEL EVALUATION

### R-squared (R²)

**Definition:**
R² = SSR/SST = 1 - SSE/SST

Where:
- SST (Total Sum of Squares) = Σ(yᵢ - ȳ)² → total variance in y
- SSR (Regression Sum of Squares) = Σ(ŷᵢ - ȳ)² → variance explained by model
- SSE (Error Sum of Squares) = Σ(yᵢ - ŷᵢ)² → unexplained variance

**Interpretation:**
- R² = proportion of variance in y explained by the model
- Range: 0 to 1
- R² = 0.7 means 70% of variance explained
- Higher is better, but doesn't prove causation
- Can be high even if model is wrong!

**For Simple Linear Regression:**
- R² = r² (square of correlation coefficient)

---

### Adjusted R²

**Formula:**
Adjusted R² = 1 - (1 - R²)(n - 1)/(n - p - 1)

Where:
- n = sample size
- p = number of predictors

**Purpose:**
- Penalizes adding predictors
- Accounts for model complexity
- Use when comparing models with different numbers of predictors

---

### Residual Standard Error (RSE)

**Formula:**
RSE = √[Σ(yᵢ - ŷᵢ)² / (n - 2)]

- Estimate of standard deviation of errors
- Average prediction error
- Same units as y
- Lower is better

---

## 5.4 RESIDUAL ANALYSIS

### Residual Plots

#### Residuals vs Fitted Values Plot

**What to Plot:**
- X-axis: fitted values (ŷ)
- Y-axis: residuals (y - ŷ)

**What to Look For:**
1. **Random scatter around 0**: assumptions met ✓
2. **Pattern/curve**: non-linearity → transform variables or try non-linear model
3. **Funnel shape**: heteroscedasticity (non-constant variance)
4. **Outliers**: points far from 0

**Why Use Fitted Values Instead of x?**
- Works for multiple regression too
- Single plot for all predictors

```r
model <- lm(y ~ x, data = data)
plot(fitted(model), residuals(model))
abline(h = 0, col = "red")

# Or with ggplot2
ggplot(data.frame(fitted = fitted(model), resid = residuals(model)),
       aes(x = fitted, y = resid)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red")
```

#### QQ Plot of Residuals
- Check normality assumption
- Should be straight diagonal line
- Less critical than linearity

#### Scale-Location Plot
- Check homoscedasticity
- Plot √|standardized residuals| vs fitted values
- Should be roughly horizontal

---

### Standardized Residuals

**Why Standardize?**
- Raw residuals have non-constant variance even if errors do
- Residuals near mean(x) tend to be smaller
- Standardization corrects for this

**Formula:**
e*ᵢ = (yᵢ - ŷᵢ) / [s√(1 - hᵢᵢ)]

Where:
- s = √[Σ(yᵢ - ŷᵢ)² / (n - 2)]
- hᵢᵢ = leverage = influence of observation i

**For simple linear regression:**
hᵢᵢ = 1/n + (xᵢ - x̄)² / Σ(xᵢ - x̄)²

**Outlier Detection:**
- |e*ᵢ| > 2: potential outlier
- |e*ᵢ| > 3: definite outlier

---

### Studentized Residuals
- Leave observation out when calculating standard error
- Even better for outlier detection
- Available in R as `rstudent(model)`

---

## 5.5 MULTIPLE LINEAR REGRESSION

### Model
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε

### Interpretation of Coefficients
- βⱼ = change in y for 1-unit increase in xⱼ, **holding all other variables constant**
- This is crucial - it's a partial effect
- Different from simple regression coefficient

---

### ANOVA for Linear Models

**Purpose:**
- Decompose variance explained by each predictor
- Understand contribution of each variable

**Sequential (Type I) ANOVA:**
```r
anova(lm(y ~ x1 + x2))
```

**Output:**
```
          Df  Sum Sq  Mean Sq  F value  Pr(>F)
x1         1  1000.0  1000.0   20.00    0.001
x2         1   200.0   200.0    4.00    0.05
Residuals 27  1350.0    50.0
```

**Interpretation:**
- Sum Sq: variance explained by that variable
- Order matters! x1 is entered first
- "How much does x2 add after x1?"

**Proportion of Variance:**
```r
round(a$`Sum Sq` / sum(a$`Sum Sq`), 2)
```
- Shows % of variance each variable explains

---

### Multicollinearity

**Problem:**
- Predictors are correlated with each other
- Makes coefficients unstable
- Hard to separate effects

**Detection:**
- Correlation matrix of predictors
- VIF (Variance Inflation Factor) > 10 is problematic

**Solutions:**
- Remove one of correlated predictors
- Combine correlated predictors
- Use regularization (ridge, lasso)

---

### Added Variable Plots

**Purpose:**
- Visualize effect of one predictor after accounting for others
- Plot: residuals of y ~ other x's vs residuals of xⱼ ~ other x's

**Interpretation:**
- Slope = coefficient for xⱼ in full model
- Shows partial relationship
- Identifies influential points for that variable

```r
library(car)
avPlots(model)
```

---

# 6. LOGISTIC REGRESSION

## 6.1 PURPOSE & USE CASES

### When to Use
- **Binary response variable** (0/1, yes/no, success/failure)
- Model: P(Y = 1 | X)
- Examples:
  - Pass/fail
  - Diseased/healthy
  - Click/no click
  - Customer churn

### Why Not Linear Regression?
- Predictions can be < 0 or > 1 (impossible for probabilities)
- Residuals not normally distributed
- Heteroscedasticity (variance changes with x)
- Non-linear relationship between x and probability

---

## 6.2 LOGISTIC FUNCTION

### Model
p(x) = e^(β₀ + β₁x) / (1 + e^(β₀ + β₁x))

Alternative form:
logit(p) = log(p/(1-p)) = β₀ + β₁x

Where:
- p(x) = P(Y = 1 | X = x)
- logit = log-odds (log of odds ratio)
- Range of p: (0, 1)
- Range of logit: (-∞, +∞)

### Shape
- S-shaped (sigmoid) curve
- Asymptotes at 0 and 1
- Steepness determined by β₁
- Inflection point at p = 0.5

---

## 6.3 INTERPRETATION

### Coefficients

**Positive β₁:**
- As x increases, probability of Y = 1 increases
- Curve rises from left to right

**Negative β₁:**
- As x increases, probability of Y = 1 decreases
- Curve falls from left to right

**Magnitude of β₁:**
- Larger |β₁| → steeper curve → stronger effect
- β₁ = 0 → no effect → flat line at p = intercept probability

### Odds Ratio
- Odds = p/(1-p)
- Odds ratio for 1-unit increase in x: e^β₁
- β₁ = 0.5 → OR = e^0.5 = 1.65 → 65% increase in odds

---

## 6.4 MODEL FITTING

### Maximum Likelihood Estimation

**Likelihood Function:**
L(β) = Π[p(xᵢ)]^yᵢ × [1 - p(xᵢ)]^(1-yᵢ)

**Goal:**
- Find β values that maximize likelihood
- Equivalently, maximize log-likelihood
- No closed-form solution → iterative algorithms

**In R:**
```r
model <- glm(y ~ x, data = data, family = "binomial")
summary(model)
```

---

## 6.5 MODEL EVALUATION

### Deviance

**Null Deviance (D₀):**
- -2 × log-likelihood of null (intercept-only) model
- Baseline model: predicts overall proportion for everyone

**Residual Deviance (D):**
- -2 × log-likelihood of fitted model
- Lower is better (better fit)

### McFadden's Pseudo-R²

**Formula:**
R²_McFadden = 1 - D/D₀

Where:
- D = deviance of fitted model
- D₀ = null deviance

**Interpretation:**
- Analogous to R² in linear regression
- NOT the same thing! Lower values typical
- 0.2 - 0.4 suggests strong fit
- 0 = null model (no improvement)
- 1 = perfect model (saturated)

**For null model:**
- R²_McFadden = 0 (by definition)

**For saturated model:**
- R²_McFadden = 1 (D = 0)

**Calculation in R:**
```r
1 - (model$deviance / model$null.deviance)
```

---

### Classification Accuracy

#### Confusion Matrix

Predicted →   | Positive | Negative |
Actual ↓      | (Ŷ = 1)  | (Ŷ = 0)  |
------------- | -------- | -------- |
Positive (Y=1)| TP       | FN       |
Negative (Y=0)| FP       | TN       |

Where:
- TP = True Positive
- TN = True Negative
- FP = False Positive (Type I error)
- FN = False Negative (Type II error)

#### Metrics

**Accuracy:**
Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Error Rate:**
Error Rate = (FP + FN) / (TP + TN + FP + FN) = 1 - Accuracy

**Sensitivity (Recall, True Positive Rate):**
Sensitivity = TP / (TP + FN)
- Proportion of actual positives correctly identified

**Specificity (True Negative Rate):**
Specificity = TN / (TN + FP)
- Proportion of actual negatives correctly identified

**Precision (Positive Predictive Value):**
Precision = TP / (TP + FP)
- Proportion of predicted positives that are correct

**F1 Score:**
F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall

---

#### Decision Threshold

**Default: 0.5**
- Predict Y = 1 if p(x) ≥ 0.5
- Predict Y = 0 if p(x) < 0.5

**Adjusting Threshold:**
- Lower threshold → more predicted positives → higher sensitivity, lower specificity
- Higher threshold → fewer predicted positives → lower sensitivity, higher specificity
- Choose based on costs of FP vs FN

**Example:**
- Medical screening: want high sensitivity (catch all sick people) → lower threshold
- Spam filter: want high specificity (don't block good emails) → higher threshold

---

### ROC Curve (Receiver Operating Characteristic)

**Definition:**
- Plot of True Positive Rate vs False Positive Rate
- For all possible thresholds
- Shows trade-off between sensitivity and specificity

**Components:**
- X-axis: False Positive Rate (1 - Specificity) = FP / (FP + TN)
- Y-axis: True Positive Rate (Sensitivity) = TP / (TP + FN)
- Each point = one threshold

**Interpretation:**
- **Diagonal line**: random classifier (no discrimination)
- **Above diagonal**: better than random
- **Upper left corner**: perfect classifier (TPR = 1, FPR = 0)
- **Curve closer to upper left**: better model

---

### AUC (Area Under ROC Curve)

**Definition:**
- Area under the ROC curve
- Single number summary of model discrimination

**Interpretation:**
- AUC = 1: perfect classifier
- AUC = 0.5: random classifier (no better than chance)
- AUC < 0.5: worse than random (something wrong!)

**Rules of Thumb:**
- AUC ≥ 0.9: excellent
- 0.8 ≤ AUC < 0.9: good
- 0.7 ≤ AUC < 0.8: acceptable
- 0.6 ≤ AUC < 0.7: poor
- AUC < 0.6: fail

**Meaning:**
- Probability that model ranks a random positive higher than a random negative

**In R:**
```r
library(pROC)
roc_obj <- roc(actual_y, predicted_prob)
auc(roc_obj)
plot(roc_obj)
```

---

# 7. DECISION TREES

## 7.1 BASICS

### Concept
- Recursive partitioning of feature space
- Tree structure: internal nodes = tests, leaves = predictions
- Non-parametric (no assumptions about distribution)
- Can handle non-linear relationships
- Interpretable

### Types
1. **Regression Trees**: continuous response
2. **Classification Trees**: categorical response

---

## 7.2 REGRESSION TREES

### How They Work

**Recursive Binary Splitting:**
1. Start with all data in one region
2. Find best split: which predictor Xⱼ and which value s minimizes RSS
3. Split data into two regions
4. Repeat for each region
5. Stop when stopping criterion met

**Finding Best Split:**
Minimize: Σ(yᵢ - ŷ_R₁)² + Σ(yᵢ - ŷ_R₂)²

Where:
- R₁ = {X | Xⱼ < s}
- R₂ = {X | Xⱼ ≥ s}
- ŷ_R = mean of y in region R

**Prediction:**
- For new observation, find which leaf it falls into
- Prediction = mean of training observations in that leaf

---

### Stopping Criteria

**Default in rpart:**
1. **cp (complexity parameter)**: minimum improvement in R²
   - Default: 0.01 (1% improvement required)
   - Split only if R² increases by at least cp
   
2. **minsplit**: minimum observations to attempt split
   - Default: 20
   
3. **minbucket**: minimum observations in leaf
   - Default: minsplit/3
   - Ensures leaves have enough data

---

### Pruning

**Why Prune?**
- Large trees overfit
- Small trees underfit
- Find optimal tree size

**Cost Complexity Pruning:**
Minimize: Σ(yᵢ - ŷᵢ)² + α|T|

Where:
- |T| = number of terminal nodes (leaves)
- α = tuning parameter (like cp)

**CP Table:**
```r
printcp(tree_model)
```

Columns:
- CP: complexity parameter
- nsplit: number of splits
- rel error: relative error (1 = null model)
- xerror: cross-validated error
- xstd: standard deviation of xerror

**Choosing CP:**
- Use cross-validation (xerror)
- Often: cp that minimizes xerror
- Or: largest cp within 1 std of minimum xerror (simpler model)

**Pruning:**
```r
pruned_tree <- prune(tree_model, cp = 0.05)
```

---

### Variable Importance (Regression)

**Definition:**
- Variable importance = total reduction in RSS from splits on that variable
- Sum across all splits using that variable

**Calculation:**
For each split on variable Xⱼ:
- Calculate RSS before split
- Calculate RSS after split (in both children)
- Improvement = RSS_before - (RSS_left + RSS_right)
- Sum all improvements for Xⱼ

**In R:**
```r
tree_model$variable.importance
```

- Higher value = more important variable
- Based on actual data, not just final tree

---

## 7.3 CLASSIFICATION TREES

### Key Difference
- Can't use RSS for splitting
- Use Gini impurity or entropy instead
- Predict class (not probability)

---

### Gini Impurity

**Formula:**
G = Σ p̂ₘₖ(1 - p̂ₘₖ)

Where:
- p̂ₘₖ = proportion of class k in region m
- Sum over all K classes

**Interpretation:**
- Measure of node impurity
- G = 0: pure node (all one class)
- G maximum when classes equally distributed
- For binary: max at p = 0.5

**Goal:**
- Find splits that maximize reduction in Gini impurity
- Want pure (or purer) nodes

**Example:**
- Node has 64 of class A, 17 of class B (total 81)
- p̂_A = 64/81 = 0.79, p̂_B = 17/81 = 0.21
- G = 0.79(1-0.79) + 0.21(1-0.21) = 0.166 + 0.166 = 0.33

---

### Entropy (Information Gain)

**Formula:**
H = -Σ p̂ₘₖ log₂(p̂ₘₖ)

**Interpretation:**
- Another measure of impurity
- H = 0: pure node
- Higher H = more mixed

**Information Gain:**
- Reduction in entropy from split
- Similar to Gini reduction

**rpart uses Gini by default**

---

### Variable Importance (Classification)

**Definition:**
- Variable importance = (reduction in Gini impurity) × (node count)
- Sum across all splits

**Example:**
- Split reduces Gini by 0.0835
- Node has 81 observations
- Contribution = 0.0835 × 81 = 6.76

**In R:**
```r
tree_model$variable.importance
```

---

### Classification CP Table

**Key Points:**
- Based on **misclassification rates** (not Gini)
- Still use for pruning
- Interpretation same as regression

---

### Making Predictions

**Predict Class:**
```r
predict(tree_model, newdata, type = "class")
```
- Returns predicted class

**Predict Probability:**
```r
predict(tree_model, newdata, type = "prob")
```
- Returns probability for each class
- Based on training proportions in leaf

---

## 7.4 ADVANTAGES & DISADVANTAGES

### Advantages
- Easy to interpret
- No assumptions about distributions
- Handle non-linear relationships
- Automatic variable selection
- Handle missing values
- Handle categorical predictors without dummy coding

### Disadvantages
- High variance (small changes in data → different tree)
- Not as accurate as other methods
- Biased toward variables with many levels
- Greedy algorithm (locally optimal, not globally)
- Can overfit easily

### Improvements
- Bagging
- Random forests
- Boosting

---

# 8. INTERPRETABLE MACHINE LEARNING (IML)

## 8.1 MOTIVATION

### The Problem
- Complex models (neural nets, ensembles) are black boxes
- High accuracy but no interpretability
- Need to understand: why this prediction?
- Required for trust, debugging, regulations

### Goals of IML
1. **Transparency**: understand how model works
2. **Accountability**: explain decisions
3. **Fairness**: detect and prevent bias
4. **Debugging**: find model errors
5. **Knowledge discovery**: learn from data

---

## 8.2 TYPES OF INTERPRETABILITY

### Model-Specific
- Built into model structure
- Linear regression, logistic regression, decision trees
- Coefficients have direct interpretation

### Model-Agnostic
- Works for any model
- Treats model as black box
- PDP, LIME, Shapley values

### Local vs Global
- **Local**: explain single prediction
- **Global**: understand model overall

---

## 8.3 PARTIAL DEPENDENCE PLOTS (PDP)

### Purpose
- Show marginal effect of feature on prediction
- Averaged over all other features
- Global method (not for single prediction)

### How It Works
1. Choose feature Xⱼ and grid of values
2. For each grid value:
   - Replace Xⱼ with that value for all observations
   - Make predictions
   - Average predictions
3. Plot average prediction vs Xⱼ

### Formula
PDP: f̂ₓⱼ(xⱼ) = E_X₋ⱼ[f̂(xⱼ, X₋ⱼ)]

Where:
- X₋ⱼ = all features except Xⱼ
- Average over distribution of X₋ⱼ

---

### One-Variable PDP

**Interpretation:**
- X-axis: feature value
- Y-axis: average predicted outcome
- Shows how prediction changes with feature
- Marginal effect (averaging out all other features)

**In R:**
```r
library(pdp)
partial(model, pred.var = "feature1", plot = TRUE)
```

---

### Two-Variable PDP

**Interpretation:**
- Heatmap or contour plot
- Shows interaction effects
- How prediction changes with both features simultaneously

**Convex Hull:**
- `chull = TRUE`: only show combinations that exist in data
- Avoid extrapolation to impossible combinations
- More reliable predictions

**In R:**
```r
partial(model, pred.var = c("feature1", "feature2"), 
        plot = TRUE, chull = TRUE)
```

---

### Limitations of PDP
1. **Assumes independence**: treats features as independent
   - Problematic if features correlated
   - Shows impossible combinations (without chull)
2. **Only shows averages**: hides heterogeneous effects
   - Different subgroups may have different relationships
3. **Causal interpretation**: PDP is not causal
   - Shows association, not causation

---

## 8.4 INDIVIDUAL CONDITIONAL EXPECTATION (ICE) PLOTS

### Purpose
- Disaggregate PDP
- Show effect for each observation
- Reveal heterogeneity

### How It Works
- Same as PDP but don't average
- One line per observation
- Shows how prediction changes for that individual as feature changes

### Interpretation
- **Parallel lines**: homogeneous effect (PDP is good summary)
- **Non-parallel lines**: heterogeneous effects (PDP hides important info)
- **Crossing lines**: different subgroups have different relationships

**In R:**
```r
partial(model, pred.var = "feature", ice = TRUE, plot = TRUE)
```

---

### Centered ICE (c-ICE)

**Purpose:**
- Make comparisons easier
- All lines start at same point

**How:**
- Subtract each line's starting value
- Shows change from baseline
- Easier to see patterns

**In R:**
```r
partial(model, pred.var = "feature", ice = TRUE, 
        center = TRUE, plot = TRUE)
```

---

## 8.5 LIME (Local Interpretable Model-Agnostic Explanations)

### Purpose
- Explain individual predictions
- Local method (one prediction at a time)
- Works for any model

### How It Works

**Steps:**
1. **Choose observation to explain** (x')
2. **Perturb x'**: create fake data around x'
3. **Get predictions**: use black box model on fake data
4. **Weight samples**: by distance/similarity to x'
5. **Fit simple model**: (linear, tree) on weighted fake data
6. **Explain**: use simple model to explain prediction

### Key Ideas
- Complex model may be non-linear globally
- But locally (near x') may be approximately linear
- Fit simple model in neighborhood of x'
- Simple model is interpretable

---

### Interpretation

**Output:**
- Which features support prediction?
- Which features contradict prediction?
- Feature weights (importance for this prediction)

**Example:**
```
Case: passenger 526
Prediction: Survived
Probability: 1.00
Explanation Fit: 0.4

Features supporting:
- gender = female  (weight = 0.4)
- class = 1st      (weight = 0.2)

Features contradicting:
- age = 37-55      (weight = -0.1)
```

---

### Limitations
1. **Instability**: different runs give different explanations (randomness in perturbation)
2. **Choice of neighborhood**: how large? how to weight?
3. **Choice of simple model**: linear? tree? interactions?
4. **Sampling**: fake data may not be realistic
5. **Explanation != global behavior**: only local

---

## 8.6 SHAPLEY VALUES

### Purpose
- Fair attribution of prediction to features
- Based on game theory
- Global or local method

### Game Theory Background
- **Players**: features
- **Payout**: prediction
- **Question**: how much does each player contribute?
- **Shapley value**: fair distribution of payout

---

### How It Works

**Idea:**
- Calculate contribution of feature by comparing predictions with and without it
- Average over all possible subsets of other features
- Accounts for interactions

**Formula:**
φⱼ = Σ [|S|!(|F| - |S| - 1)! / |F|!] × [f(S ∪ {j}) - f(S)]

Where:
- φⱼ = Shapley value for feature j
- S = subset of features (not including j)
- F = all features
- f(S) = expected prediction using features in S

**Properties:**
- **Efficiency**: Σφⱼ = f(x) - E[f(x)] (prediction - average)
- **Symmetry**: if features contribute equally, equal Shapley values
- **Dummy**: if feature doesn't matter, φⱼ = 0
- **Additivity**: for combined models

---

### Interpretation

**Local (Single Prediction):**
- Positive φⱼ: feature increases prediction
- Negative φⱼ: feature decreases prediction
- Magnitude: size of contribution
- Sum: prediction = baseline + Σφⱼ

**Global (Average Absolute):**
- Average |φⱼ| across all observations
- Shows overall feature importance
- Higher = more important feature

---

### SHAP (SHapley Additive exPlanations)

**Popular Implementation:**
- Specific method for computing Shapley values
- Efficient approximations (kernel SHAP, tree SHAP)
- Works for many model types
- Visualization tools

**In R:**
```r
library(fastshap)
shap_values <- explain(model, X = X_train, nsim = 500)
```

---

### Advantages
- **Fair**: only method with axiomatic fairness properties
- **Consistent**: adding helpful feature always increases its Shapley value
- **Local accuracy**: sum of contributions = prediction difference

### Limitations
1. **Computational cost**: exponentially many subsets
   - Need approximations for large feature sets
2. **Interpretation**: assumes features can be "removed"
   - What does it mean to remove correlated feature?
3. **Inclusion of unrealistic data**: samples feature combinations that don't occur

---

## 8.7 VARIABLE IMPORTANCE

### Global Measures

#### Model-Specific

**Linear Regression:**
- Coefficient magnitude (if standardized)
- t-statistic
- Contribution to R²

**Logistic Regression:**
- Coefficient magnitude
- Wald statistic

**Decision Trees:**
- Total RSS/Gini reduction
- `tree$variable.importance`

**Random Forests:**
- **Mean Decrease Accuracy**: how much accuracy drops when variable permuted
- **Mean Decrease Gini**: total Gini reduction from splits on variable
- Built-in: `importance(rf_model)`

---

#### Model-Agnostic

**Permutation Importance:**
1. Train model
2. Record baseline performance
3. For each feature:
   - Permute (shuffle) that feature
   - Predict with permuted data
   - Record performance drop
4. Importance = performance drop

**Advantages:**
- Works for any model
- Clear interpretation
- Accounts for interactions

**In R:**
```r
library(vip)
vip(model, method = "permute")
```

---

**Partial Dependence:**
- If PDP is flat → feature not important
- If PDP varies a lot → feature is important
- Can quantify with variance of PDP

---

## 8.8 DECISION RULES (RIPPER)

### Purpose
- Extract interpretable rules from data
- Alternative to decision trees
- More compact than trees

### How RIPPER Works

**Growing Phase:**
1. Start with empty rule
2. Add conditions that maximize p/(p+n)
   - p = positive cases covered
   - n = negative cases covered
3. Stop when adding conditions doesn't improve enough

**Pruning Phase:**
- Remove conditions to prevent overfitting
- Use validation set

**Optimization:**
- Refine rules
- Delete rules that don't improve

---

### Rule Format
```
IF condition1 AND condition2 AND ... THEN class
```

**Example:**
```
IF odor = foul THEN poisonous
IF gill_size = narrow AND gill_color = buff THEN poisonous
OTHERWISE edible
```

---

### Comparing Conditions

**Question:** Which condition should RIPPER add?
- Condition A: 40 positive out of 90 total → 25 positive out of 60
- Condition B: 40 positive out of 90 total → 20 positive out of 50

**Answer:** Maximize p/(p+n)
- Condition A: 25/60 = 0.417
- Condition B: 20/50 = 0.400
- Choose Condition A

**NOT:**
- ~~Maximum reduction in total cases~~
- ~~Maximum information gain~~
- ~~Minimize positive cases remaining~~
- ~~Maximize class balance~~

---

### Advantages
- Very interpretable (IF-THEN rules)
- Compact (fewer rules than tree branches)
- Handles unbalanced classes well

### Disadvantages
- May not be as accurate as complex models
- Sequential (order matters)

---

# 9. TIME SERIES & DATES

## 9.1 TIME SERIES VISUALIZATION

### Line Plots

**Purpose:**
- Show change over time
- Time on x-axis (always!)
- Response on y-axis

**What to Look For:**
1. **Trend**: long-term increase/decrease
2. **Seasonality**: regular, periodic fluctuations
3. **Cycles**: irregular fluctuations (business cycles)
4. **Noise**: random variation
5. **Outliers**: unusual values
6. **Sudden changes**: structural breaks
7. **Missing values**: gaps in data

```r
ggplot(data, aes(x = date, y = value)) +
  geom_line()
```

---

### Multiple Time Series

**Strategies:**
1. **Faceting**: separate panels
   - Easy to see individual series
   - Hard to compare directly

2. **Overlapping lines**: same plot
   - Easy to compare
   - Can get messy with many series
   - Use transparency

3. **Small multiples with highlight**:
   - Show all faded in each panel
   - Highlight one per panel

```r
# Faceting
ggplot(data, aes(x = date, y = value)) +
  geom_line() +
  facet_wrap(~ series)

# Overlapping
ggplot(data, aes(x = date, y = value, color = series)) +
  geom_line(alpha = 0.7)
```

---

## 9.2 DATE HANDLING IN R

### Date Classes

**Date:**
- Date only (no time)
- Example: 2024-01-15

**POSIXct:**
- Date and time
- Stored as seconds since 1970-01-01
- Example: 2024-01-15 14:30:00

**POSIXlt:**
- Date and time as list
- Rarely used

---

### Converting to Dates

```r
# Character to Date
as.Date("2024-01-15")
as.Date("01/15/2024", format = "%m/%d/%Y")

# Character to POSIXct
as.POSIXct("2024-01-15 14:30:00")

# Using lubridate
library(lubridate)
ymd("2024-01-15")
mdy("01/15/2024")
ymd_hms("2024-01-15 14:30:00")
```

---

### Date Format Codes

Common codes:
- %Y: 4-digit year (2024)
- %y: 2-digit year (24)
- %m: month number (01-12)
- %b: abbreviated month (Jan)
- %B: full month (January)
- %d: day of month (01-31)
- %H: hour (00-23)
- %M: minute (00-59)
- %S: second (00-59)

---

### Extracting Date Components

```r
# Base R
date_obj <- as.Date("2024-01-15")
year(date_obj)   # requires lubridate
month(date_obj)
day(date_obj)

# lubridate
library(lubridate)
year(date_obj)
month(date_obj)
day(date_obj)
wday(date_obj)      # day of week (1-7)
wday(date_obj, label = TRUE)  # Sun, Mon, ...
```

---

### Date Arithmetic

```r
# Add/subtract days
date + 7
date - 3

# Difference between dates
date2 - date1  # returns difftime object

# Sequences
seq(from = start_date, to = end_date, by = "day")
seq(from = start_date, to = end_date, by = "month")
```

---

# 10. SPATIAL DATA

## 10.1 MAPS & SPATIAL VISUALIZATION

### Choropleth Maps

**Purpose:**
- Show values by geographic region
- Color/shading represents data value
- Regions = countries, states, counties, zip codes, etc.

**What to Look For:**
- Spatial patterns
- Clusters
- Hot spots
- Regional differences

---

### Creating Choropleths in R

**Using choroplethr package:**
```r
library(choroplethr)
data("df_pop_state")  # requires 'region' and 'value' columns
state_choropleth(df_pop_state)
```

**Using ggplot2 + maps:**
```r
library(maps)
states_map <- map_data("state")

ggplot(data, aes(map_id = region)) +
  geom_map(aes(fill = value), map = states_map) +
  expand_limits(x = states_map$long, y = states_map$lat)
```

---

### Considerations

**Normalization:**
- Show rates/proportions, not raw counts
- Account for population or area
- Example: crime rate per 1000, not total crimes

**Color Scales:**
- Sequential: one variable (light to dark)
- Diverging: data has meaningful midpoint
- Quantitative: continuous values
- Qualitative: categories

**Map Projections:**
- Different projections distort differently
- Choose appropriate for region and purpose

---

### Shapefiles

**What are they:**
- Geospatial vector data format
- Contains boundaries of regions
- Multiple files (.shp, .shx, .dbf, etc.)

**Reading in R:**
```r
library(sf)
shapefile <- st_read("path/to/file.shp")

# Plot
ggplot(shapefile) +
  geom_sf(aes(fill = variable))
```

---

### Point Maps

**Purpose:**
- Show locations of events/observations
- Each point = one observation

**Enhancements:**
- Size: represent quantity
- Color: represent category
- Transparency: handle overplotting
- Jittering: avoid exact overlaps

```r
ggplot(data, aes(x = longitude, y = latitude)) +
  geom_point(aes(size = population, color = category),
             alpha = 0.5) +
  borders("state")
```

---

## 10.2 ADVANCED SPATIAL PLOTS

### Cartograms
- Distort geographic size to represent data
- Example: states sized by population
- Hard to read but eye-catching

### Tile Grid Maps (Statebins)
- Each region = one tile
- Maintain relative positions
- Easy to compare values
- Lose geographic accuracy

```r
library(statebins)
statebins(data, value_col = "value", state_col = "state")
```

---

# 11. COLOR THEORY

## 11.1 COLOR PRINCIPLES

### Color Channels
- **Hue**: the color itself (red, blue, green)
- **Saturation**: intensity/purity of color
- **Lightness/Value**: how light or dark

---

### Types of Color Scales

#### Sequential
- **Use**: ordered data, one variable
- **Design**: light to dark, one or two hues
- **Examples**: 
  - Population density (low to high)
  - Temperature (cold to hot)
- **ColorBrewer**: Blues, Reds, Greens

```r
scale_fill_brewer(palette = "Blues")
scale_fill_gradient(low = "white", high = "darkblue")
```

---

#### Diverging
- **Use**: data with meaningful midpoint
- **Design**: two hues, meet at neutral middle
- **Examples**:
  - Change (negative to positive)
  - Temperature (cold to hot)
  - Correlation (-1 to +1)
- **ColorBrewer**: RdBu, BrBG, PiYG

```r
scale_fill_brewer(palette = "RdBu")
scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                     midpoint = 0)
```

---

#### Qualitative
- **Use**: categorical data, no order
- **Design**: distinct, distinguishable hues
- **Examples**:
  - Species
  - Countries
  - Product categories
- **ColorBrewer**: Set1, Set2, Set3, Paired

```r
scale_fill_brewer(palette = "Set2")
scale_color_manual(values = c("red", "blue", "green"))
```

---

### Perceptual Uniformity

**Problem:**
- Not all colors perceived as equally spaced
- Rainbow scale: perceptually non-uniform
- Viewers see false boundaries

**Solution:**
- Use perceptually uniform scales
- viridis family: designed for uniform perception
- ColorBrewer palettes

```r
scale_fill_viridis_c()  # continuous
scale_fill_viridis_d()  # discrete
```

---

### Colorblind-Friendly

**Guidelines:**
- Avoid red-green combinations
- Use viridis scales
- Use ColorBrewer colorblind-safe palettes
- Add patterns/shapes in addition to color
- Test with colorblind simulators

```r
# ColorBrewer colorblind safe
scale_fill_brewer(palette = "Set2")  # all colorblind safe
scale_fill_brewer(palette = "Dark2")
```

---

## 11.2 PRACTICAL COLOR ADVICE

### When to Use Color
- Encode categorical variables
- Encode continuous variables
- Highlight specific data
- Show groups

### When NOT to Use Color
- When not necessary (minimize ink!)
- Too many categories (>7 hard to distinguish)
- If grayscale would work

### Number of Colors
- **2-3**: easy to distinguish
- **4-7**: still manageable
- **8+**: very difficult, consider alternatives

---

### Best Practices
1. **Be intentional**: every color should have meaning
2. **Be consistent**: same color = same thing across plots
3. **Use white space**: don't overuse color
4. **Test**: view on different devices, print in grayscale
5. **Consider audience**: colorblindness, cultural associations
6. **Label directly**: don't rely solely on legends

---

# 12. ggplot2 GRAMMAR OF GRAPHICS

## 12.1 PHILOSOPHY

### Layered Grammar
- ggplot2 implements "Grammar of Graphics"
- Data visualization = mapping data to aesthetic properties
- Build plots by adding layers

### Core Concept
- **Data**: what to plot
- **Aesthetics**: how to map data to visual properties
- **Geometries**: what type of plot
- **Statistics**: transformations of data
- **Scales**: control mapping from data to aesthetics
- **Coordinates**: coordinate system
- **Facets**: split into subplots
- **Themes**: overall appearance

---

## 12.2 BASIC STRUCTURE

### Template
```r
ggplot(data = <DATA>) +
  <GEOM_FUNCTION>(mapping = aes(<MAPPINGS>),
                  stat = <STAT>,
                  position = <POSITION>) +
  <COORDINATE_FUNCTION> +
  <FACET_FUNCTION> +
  <SCALE_FUNCTION> +
  <THEME_FUNCTION>
```

### Minimal Example
```r
ggplot(data, aes(x = var1, y = var2)) +
  geom_point()
```

---

## 12.3 COMPONENTS

### Data
- Must be data frame
- Each row = observation
- Each column = variable
- Tidy data works best

---

### Aesthetics (aes)
- Map variables to visual properties
- Common aesthetics:
  - **x**: x-axis position
  - **y**: y-axis position
  - **color**: color of points/lines
  - **fill**: fill color of bars/areas
  - **size**: size of points
  - **shape**: shape of points
  - **alpha**: transparency
  - **linetype**: type of line

**Inside aes():**
- Maps variable to aesthetic
- `aes(x = var1, color = var2)`

**Outside aes():**
- Sets fixed value
- `geom_point(color = "blue")`

---

### Geoms
- Geometric objects to display data
- Different geoms for different plot types

**Common Geoms:**
- `geom_point()`: scatterplot
- `geom_line()`: line plot
- `geom_bar()`: bar chart
- `geom_histogram()`: histogram
- `geom_boxplot()`: boxplot
- `geom_density()`: density plot
- `geom_smooth()`: smoothed line
- `geom_tile()`: heatmap
- `geom_violin()`: violin plot

---

### Stats
- Statistical transformations
- Applied before plotting
- Usually automatic

**Examples:**
- `stat = "identity"`: use data as-is
- `stat = "count"`: count observations (default for geom_bar)
- `stat = "bin"`: bin data (default for geom_histogram)
- `stat = "smooth"`: smooth data

**Using after_stat():**
- Access computed variables
- Example: `aes(y = after_stat(density))` in histogram

---

### Position
- Adjust position of geoms
- Handle overlapping

**Options:**
- `position = "identity"`: no adjustment (default for points)
- `position = "stack"`: stack on top (default for bars)
- `position = "dodge"`: side-by-side
- `position = "fill"`: stack to 100%
- `position = "jitter"`: add random noise (avoid overplotting)

```r
geom_bar(position = "dodge")
geom_point(position = "jitter")
```

---

### Scales
- Control mapping from data to aesthetics
- Customize axes, colors, sizes, etc.

**Common Scale Functions:**
```r
scale_x_continuous()      # continuous x-axis
scale_y_log10()          # log scale y-axis
scale_color_manual()     # manual colors
scale_fill_brewer()      # ColorBrewer colors
scale_size_continuous()  # continuous sizes
```

**Components:**
- **name**: axis/legend title
- **breaks**: where to put tick marks/legend keys
- **labels**: text for ticks/keys
- **limits**: min and max
- **trans**: transformation (log, sqrt, etc.)

---

### Coordinates
- Coordinate system

**Options:**
- `coord_cartesian()`: default, Cartesian
- `coord_flip()`: flip x and y axes
- `coord_polar()`: polar coordinates
- `coord_map()`: map projections

```r
ggplot(data, aes(x = category, y = value)) +
  geom_bar(stat = "identity") +
  coord_flip()  # horizontal bars
```

---

### Facets
- Split plot into subplots
- One subplot per level of variable(s)

**Functions:**
```r
facet_wrap(~ variable)           # wrap into rectangular layout
facet_grid(rows = vars(var1))    # grid, rows only
facet_grid(cols = vars(var2))    # grid, columns only
facet_grid(rows = vars(var1), cols = vars(var2))  # both
```

**Options:**
- `scales = "free"`: independent axis scales per facet
- `scales = "free_x"`: free x-axis only
- `scales = "free_y"`: free y-axis only
- `ncol`: number of columns (facet_wrap only)

---

### Themes
- Control non-data plot appearance
- Overall look and feel

**Pre-built Themes:**
- `theme_gray()`: default
- `theme_bw()`: black and white
- `theme_minimal()`: minimal
- `theme_classic()`: classic look
- `theme_void()`: completely empty

**Customization:**
```r
theme(
  plot.title = element_text(size = 14, face = "bold"),
  axis.title = element_text(size = 12),
  axis.text = element_text(size = 10),
  legend.position = "bottom",
  panel.grid = element_blank()
)
```

---

## 12.4 BEST PRACTICES

### Building Plots Incrementally
```r
p <- ggplot(data, aes(x = var1, y = var2))
p <- p + geom_point()
p <- p + geom_smooth()
p <- p + labs(title = "My Plot")
p
```

### Labels
```r
labs(
  title = "Main Title",
  subtitle = "Subtitle",
  x = "X Axis Label",
  y = "Y Axis Label",
  caption = "Source: ...",
  color = "Legend Title"
)
```

### Saving Plots
```r
ggsave("plot.png", width = 6, height = 4, dpi = 300)
```

---

## 12.5 COMMON PATTERNS

### Overlaying Geoms
```r
ggplot(data, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm")
```

### Different Data in Different Layers
```r
ggplot(data1, aes(x = x, y = y)) +
  geom_point() +
  geom_point(data = data2, aes(x = x2, y = y2), color = "red")
```

### Reordering Categories
```r
ggplot(data, aes(x = fct_reorder(category, value), y = value)) +
  geom_col()
```

---

# 13. KEY MATHEMATICAL CONCEPTS

## 13.1 RESIDUALS & FITTED VALUES

### Definitions
- **Fitted value (ŷᵢ)**: predicted value from model
- **Residual (eᵢ)**: yᵢ - ŷᵢ
- **Residual = observed - predicted**

### Properties
- Sum of residuals = 0 (for models with intercept)
- Residuals should be random if model is good
- Patterns in residuals indicate model problems

---

## 13.2 ANOVA DECOMPOSITION

### Sum of Squares Decomposition
SST = SSR + SSE

Where:
- **SST** (Total) = Σ(yᵢ - ȳ)² → total variation
- **SSR** (Regression) = Σ(ŷᵢ - ȳ)² → explained by model
- **SSE** (Error) = Σ(yᵢ - ŷᵢ)² → unexplained (residual)

### R-squared
R² = SSR/SST = 1 - SSE/SST

- Proportion of variance explained
- 0 ≤ R² ≤ 1

---

## 13.3 LOESS FITTING DIAGNOSTICS

### Assessing LOESS Fit

**Overfitted (too wiggly):**
- Follows individual points too closely
- Many peaks and valleys
- High variance, low bias
- Solution: increase span parameter

**Underfitted (too smooth):**
- Misses important patterns
- Too straight
- High bias, low variance
- Solution: decrease span parameter

**Just right:**
- Captures general trend
- Smooths out noise
- Balances bias and variance

---

## 13.4 SHAPLEY VALUE CALCULATION

### Formula Components
For feature j:

φⱼ = Σ_S [weight × contribution]

Where:
- S = subsets not including j
- weight = |S|!(|F| - |S| - 1)! / |F|!
- contribution = f(S ∪ {j}) - f(S)

### Example Calculation
Features: {STABLE_JOB, HOMEOWNER}

For HOMEOWNER:
1. Subset ∅: contribution = f({H}) - f(∅) = 70 - 40 = 30
2. Subset {S}: contribution = f({S,H}) - f({S}) = 100 - 60 = 40

Shapley value = (1/2) × 30 + (1/2) × 40 = 35

---

# 14. EXAM TIPS & COMMON PITFALLS

## 14.1 READING PLOTS CAREFULLY

### Missing Value Heatmaps
- X's mark missing values
- Look for columns WITHOUT missing that predict missingness in others
- These are the "predictive" variables

### Residual Plots
- Random scatter = good
- Pattern = problem (non-linearity, heteroscedasticity)
- No strong conclusions possible = "none of the above"

### Classification Plots
- Count points carefully
- Consider decision boundary
- Remember: prediction vs actual

---

## 14.2 COMMON MISTAKES

### McFadden's Pseudo-R²
- For null model: ALWAYS 0 (by definition)
- Not the same as regular R²
- Values 0.2-0.4 are considered strong

### ANOVA Interpretation
- "Variable A more important than B" is NOT necessarily true
- Depends on order entered
- Correlation between predictors matters
- Can't determine from ANOVA alone if predictors are correlated

### Partial Dependence Plots
- Convex hull prevents extrapolation
- Shows combinations that DON'T exist in data
- Not about removing missing values
- Not about testing correlation

### LIME Limitations
- Gives DIFFERENT explanations on different runs (randomness)
- Can handle categorical data
- Doesn't require GLM
- Can approximate non-linear models locally

---

## 14.3 QUICK REFERENCE FORMULAS

### Linear Regression
- SST = SSR + SSE
- R² = 1 - SSE/SST
- Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1)

### Logistic Regression
- p(x) = e^(β₀+β₁x) / (1 + e^(β₀+β₁x))
- McFadden's R² = 1 - D/D₀
- Odds ratio = e^β₁

### Decision Trees (Classification)
- Gini = Σ p̂ₘₖ(1 - p̂ₘₖ)
- Variable importance = (Gini reduction) × (node count)

### RIPPER
- Chooses condition maximizing p/(p+n)
- NOT maximum reduction in cases
- NOT maximum information gain

---

## 14.4 VISUALIZATION CHECKLIST

**Before Creating Plot:**
1. What is the variable type? (continuous, categorical, ordinal)
2. How many variables? (1, 2, 3+)
3. What is the purpose? (distribution, relationship, comparison, composition)
4. Who is the audience?

**After Creating Plot:**
1. Are axes labeled clearly?
2. Is there a title?
3. Are scales appropriate? (start at 0 for bars?)
4. Is it colorblind-friendly?
5. Is there a legend (if needed)?
6. Can it be simplified?

---

# 15. FINAL SUMMARY

## Essential Plot Types

### One Variable
- **Continuous**: histogram, density, boxplot
- **Categorical**: bar chart, dot plot

### Two Variables
- **Continuous + Continuous**: scatterplot, 2D density
- **Continuous + Categorical**: side-by-side boxplots, faceted histograms, violin plots
- **Categorical + Categorical**: mosaic plot, grouped bar chart, stacked bar chart

### Multiple Variables
- **Many continuous**: scatterplot matrix, parallel coordinates, PCA biplot
- **Many categorical**: alluvial diagram, faceted mosaic plots

### Specialized
- **Time series**: line plot
- **Spatial**: choropleth map, point map
- **Missing data**: heatmap with mi package, aggregated patterns

---

## Key Concepts to Remember

### Distributions
- Shape, center, spread, outliers, modality
- Density histogram: area = 1
- Boxplot: IQR, median, whiskers to 1.5×IQR

### Relationships
- Correlation ≠ causation
- LOESS: overfitted, underfitted, just right
- Added variable plots: partial relationships

### Regression
- Assumptions: linearity, independence, homoscedasticity, normality
- R²: proportion of variance explained
- Residual plots: check assumptions

### Classification
- Logistic: S-shaped curve
- McFadden's R²: null model = 0
- ROC/AUC: discrimination ability
- Confusion matrix: TP, TN, FP, FN

### Trees
- Regression: minimize RSS
- Classification: minimize Gini impurity
- Variable importance: total reduction
- Pruning: use cross-validation

### Interpretability
- PDP: marginal effects (averaged)
- ICE: individual effects (disaggregated)
- LIME: local explanations (may vary)
- Shapley: fair attribution (game theory)

### Color
- Sequential: ordered, one variable
- Diverging: meaningful midpoint
- Qualitative: unordered categories
- Use colorblind-friendly palettes

---

## Study Strategy

1. **Understand concepts**: don't just memorize
2. **Practice calculations**: McFadden's R², Gini, etc.
3. **Read plots carefully**: look at axes, scales, legends
4. **Know when to use what**: match plot type to data type
5. **Understand limitations**: what each method can and can't do

---


