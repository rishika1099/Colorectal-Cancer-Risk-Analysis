# COMPREHENSIVE LIST OF ALL PLOT TYPES IN EDAV COURSE

**Based on all 48 course PDFs**

---

## QUICK REFERENCE TABLE

| Data Type | Plot Types |
|-----------|-----------|
| 1 Continuous | Histogram, Density, Boxplot, Violin, Ridgeline, QQ |
| 1 Categorical | Bar chart, Cleveland dot plot |
| 2 Continuous | Scatterplot, Smooth/LOESS, 2D density, Hexbin, Contour, Heatmap |
| 2 Categorical | Mosaic, Grouped bar, Stacked bar, Proportional stack, Alluvial |
| Continuous + Categorical | Boxplots (side-by-side), Faceted histograms, Ridgeline |
| Multiple Continuous | Scatterplot matrix, Parallel coordinates, Biplot |
| Time Series | Line plot, Multi-line, Geofacet |
| Spatial | Choropleth, Point map, Density map, Cartogram, Statebins |
| Missing Data | Heatmap (mi package), Tile plot, Aggregated patterns |
| Diagnostics | Residual plots, ROC curve, Scale-location |
| ML/IML | Decision tree, PDP, ICE, LIME, Shapley, Variable importance |

**Total: 60+ distinct visualization types**

---

## DETAILED BREAKDOWN

## 1. UNIVARIATE CONTINUOUS DATA (6 types)

### 1.1 Histogram (`geom_histogram()`)
**Purpose:** Display distribution
**Variants:**
- Frequency (count)
- Density (area = 1)
- Equal/unequal binwidths  
- Right-closed (a, b] vs right-open [a, b)

### 1.2 Boxplot (`geom_boxplot()`)
**Components:** Box=IQR, line=median, whiskers=1.5×IQR, dots=outliers
**Best for:** Comparing groups, outliers

### 1.3 Violin Plot (`geom_violin()`)
**Purpose:** Mirrored density showing full distribution shape
**Advantage:** Shows bimodality

### 1.4 Density Curve (`geom_density()`)
**Purpose:** Smooth probability density
**Parameter:** adjust/bandwidth

### 1.5 Ridgeline/Joy Plot (`geom_density_ridges()`, ggridges)
**Purpose:** Stacked densities for many categories
**Best for:** Time series distributions

### 1.6 QQ Plot (`stat_qq()` + `stat_qq_line()`)
**Purpose:** Check normality
**Interpretation:** Straight line = normal

---

## 2. CATEGORICAL DATA (2 types)

### 2.1 Bar Chart (`geom_bar()` / `geom_col()`)
**Variants:**
- Frequency (geom_bar)
- Value (geom_col)
- Horizontal (coord_flip)
- Ordered (fct_reorder, fct_infreq)

### 2.2 Cleveland Dot Plot (`geom_point()`)
**Advantages:** Better data-ink ratio, easier to read values
**Variants:** Single/multiple dots, faceted, panels

---

## 3. MULTIVARIATE CATEGORICAL (5 types)

### 3.1 Mosaic Plot (`vcd::mosaic()`)
**Properties:** Area ∝ frequency, no white space
**Variants:** Standard, same bin width, 3-variable

### 3.2 Grouped Bar Chart (`position="dodge"`)
**Purpose:** Bars side-by-side

### 3.3 Stacked Bar Chart (`position="stack"`)
**Shows:** Composition + totals

### 3.4 Proportional Stacked Bar (`position="fill"`)
**Properties:** All bars 100% height

### 3.5 Alluvial/Sankey (`geom_alluvium()`, ggalluvial)
**Purpose:** Show flows/transitions
**Formats:** Stratum, lodes

---

## 4. TWO CONTINUOUS VARIABLES (8 types)

### 4.1 Scatterplot (`geom_point()`)
**Look for:** Direction, strength, form, outliers, clusters
**Enhancements:** Alpha, jitter, size/color for 3rd var

### 4.2 Scatterplot with Smooth (`geom_smooth()`)
**Methods:** lm, loess, gam

### 4.3 LOESS Smoother
**Assessment:** Overfitted (wiggly), underfitted (smooth), just right
**Parameter:** span

### 4.4 2D Density (`geom_density_2d()`)
**Variants:** Contours, filled contours

### 4.5 Hexagonal Binning (`geom_hex()`, hexbin)
**Best for:** Large datasets with overplotting

### 4.6 2D Histogram (`geom_bin_2d()`)
**Shows:** Count/density in rectangular bins

### 4.7 Contour Plot (`geom_contour()`)
**Purpose:** 3D surface as 2D contours
**Requires:** x, y, z

### 4.8 Tile/Heatmap (`geom_tile()`)
**Purpose:** Value by color in grid
**Uses:** Matrices, correlations, missing data

---

## 5. MULTIPLE CONTINUOUS VARIABLES (4 types)

### 5.1 Scatterplot Matrix (`pairs()` / `ggpairs()`)
**Shows:** All pairwise scatterplots
**Diagonal:** Distributions

### 5.2 Parallel Coordinates (`ggparallel()`, GGally)
**Format:** Variables=vertical axes, observations=lines
**Best for:** Patterns, clusters

### 5.3 Biplot (`biplot()` / `autoplot()`)
**Purpose:** PCA visualization
**Shows:** Observations (points) + variables (arrows)
**Arrows:** Direction=correlation, length=contribution

### 5.4 3D Scatterplot
**Methods:** True 3D or 2D with color/size
**Note:** 2D alternatives usually better

---

## 6. CONTINUOUS + CATEGORICAL (5 types)

### 6.1 Faceted Histograms (`facet_wrap()` / `facet_grid()`)
**Purpose:** Compare distributions, small multiples

### 6.2 Side-by-Side Boxplots
**Best for:** Many groups

### 6.3 Faceted Density Plots
**Purpose:** Compare densities by category

### 6.4 Overlapping Density Curves
**Uses:** Transparency, color

### 6.5 Ridgeline Plots
**Better than:** Overlapping densities for many groups

---

## 7. TIME SERIES & TEMPORAL (4 types)

### 7.1 Line Plot (`geom_line()`)
**Requirements:** Time on x-axis
**Look for:** Trends, seasonality, cycles, gaps

### 7.2 Multi-Line Plot
**Variants:** Overlapping, faceted, highlighted

### 7.3 Line + Points (`geom_line()` + `geom_point()`)
**Purpose:** Show data frequency

### 7.4 Geofacet (`geofacet()`, geofacet package)
**Purpose:** Time series in geographic layout

---

## 8. SPATIAL / GEOGRAPHIC (6 types)

### 8.1 Choropleth Map (`geom_map()` / choroplethr)
**Purpose:** Values by region
**Packages:** choroplethr, maps, sf

### 8.2 Point/Pushpin Map (`geom_point()` on map)
**Enhancements:** Size, color, transparency

### 8.3 Density Map
**Purpose:** Spatial density heatmap

### 8.4 Faceted Choropleth
**Purpose:** Geography across time/categories

### 8.5 Cartogram
**Purpose:** Distort geography by data
**Example:** States sized by population

### 8.6 Tile Grid Map/Statebins (`statebins()`)
**Advantages:** Equal weight, easy comparison

---

## 9. MISSING DATA VISUALIZATION (3 types)

### 9.1 Missing Data Heatmap (`mi::image()`, mi package)
**Shows:** Row/column patterns, associations

### 9.2 Missing Data Tile Plot (`geom_tile()`)
**Shows:** Missing vs present by color

### 9.3 Aggregated Missing Patterns (`plot_missing()`, redav)
**Shows:** Unique pattern frequencies

---

## 10. MODEL DIAGNOSTICS & EVALUATION (4 types)

### 10.1 Residual vs Fitted Plot
**X:** Fitted values (ŷ)
**Y:** Residuals (y - ŷ)
**Look for:** Random scatter

### 10.2 Scale-Location Plot
**Y:** √|standardized residuals|
**Purpose:** Check homoscedasticity

### 10.3 ROC Curve
**X:** False Positive Rate
**Y:** True Positive Rate
**Metric:** AUC (area under curve)

### 10.4 Confusion Matrix Visualization
**Shows:** TP, TN, FP, FN

---

## 11. MACHINE LEARNING / IML (6 types)

### 11.1 Decision Tree Diagram (`rpart.plot()` / `ggparty()`)
**Shows:** Splits, nodes, predictions

### 11.2 Partial Dependence Plot (`pdp::partial()`)
**Shows:** Marginal effect
**Variants:** 1-var (line), 2-var (heatmap), convex hull

### 11.3 ICE Plot (Individual Conditional Expectation)
**Shows:** One line per observation
**Variants:** Standard, centered (c-ICE)

### 11.4 LIME Plot (`lime::plot_features()`)
**Purpose:** Explain individual predictions
**Type:** Horizontal bar chart

### 11.5 Shapley Value Plot (`fastshap` / SHAP)
**Purpose:** Fair attribution
**Variants:** Local, global, waterfall, force

### 11.6 Variable Importance Plot (`vip()`)
**Purpose:** Global feature importance
**Types:** Bar chart, scaled to 100

---

## 12. SPECIALIZED / ADVANCED (4 types)

### 12.1 Added Variable Plot (`car::avPlots()`)
**Purpose:** Partial relationship in multiple regression

### 12.2 Correlation Plot/Correlogram
**Variants:** Heatmap, network, ellipses

### 12.3 Spine Plot (`spineplot()`)
**Purpose:** Conditional distributions (categorical)

### 12.4 CD Plot (`cdplot()`)
**Purpose:** Conditional density (continuous by categorical)

---

## 13. ADDITIONAL ggplot2 GEOMS (8 types)

### 13.1 geom_jitter()
**Purpose:** Add noise to avoid overplotting

### 13.2 geom_rug()
**Purpose:** Marginal distributions along axes

### 13.3 geom_segment() / geom_curve()
**Purpose:** Line segments, curves

### 13.4 geom_text() / geom_label()
**Purpose:** Text annotations

### 13.5 geom_area()
**Purpose:** Filled area under line

### 13.6 geom_ribbon()
**Purpose:** Confidence intervals (ymin, ymax)

### 13.7 geom_errorbar() / geom_linerange()
**Purpose:** Error bars/ranges

### 13.8 geom_crossbar()
**Purpose:** Point estimate + range

---

## ORGANIZED BY COURSE TOPICS

### **Univariate Continuous (Slides 02, 04, 05):**
Histogram, Boxplot, Density, Violin, Ridgeline, QQ plot

### **Categorical (Slides 06, 07):**
Bar chart, Cleveland dot plot

### **Multivariate Categorical (Slides 13, 14, 15, 16, 17):**
Mosaic, Grouped/stacked bars, Alluvial

### **Dependency Relationships (Slide 09):**
Scatterplot, Smooth, 2D density, Hexbin, Contour

### **Multiple Continuous (Slides 12, 18, 19):**
Scatterplot matrix, Parallel coordinates, Biplot

### **Faceting & Combining (Slides 10, 11):**
Faceted plots, small multiples

### **Missing Data (Slides 08, 25):**
Heatmaps, tile plots, aggregated patterns

### **Spatial (Slides 23, 24):**
Choropleth, point maps, cartograms

### **Time Series (Slides 26, 27):**
Line plots, multi-line, geofacet

### **Linear Regression (Slides 30, 31):**
Residual plots, added variable plots

### **Logistic Regression (Slide 32):**
Logistic curves, classification plots

### **Decision Trees (Slides 35, 36, 37):**
Tree diagrams

### **ROC (Slide 45):**
ROC curves, AUC

### **IML (Slides 28, 29, 38, 39, 41, 42, 46):**
PDP, ICE, LIME, Shapley, Variable importance

### **Color & Grammar (Slides 03, 21, 22):**
Color scales, ggplot2 components

---

## EXAM PREPARATION TIPS

### **Know When to Use What:**
1. **1 continuous** → histogram, boxplot, density
2. **1 categorical** → bar chart
3. **2 continuous** → scatterplot (+ smooth if needed)
4. **2 categorical** → mosaic, stacked bars
5. **Continuous + categorical** → side-by-side boxplots, faceted histograms
6. **Many continuous** → scatterplot matrix, parallel coordinates
7. **Time series** → line plot
8. **Spatial** → choropleth
9. **Missing data** → heatmap (mi package)
10. **Model diagnostics** → residual plots, ROC
11. **ML interpretation** → PDP, Shapley, variable importance

### **Common Plot Characteristics to Remember:**
- **Histogram:** binwidth matters, density vs frequency
- **Boxplot:** IQR, 1.5×IQR whiskers
- **Mosaic:** area ∝ frequency
- **Ridgeline:** better than boxplots for seeing distribution shape
- **LOESS:** can be overfitted, underfitted, or just right
- **ROC:** closer to upper left = better
- **PDP:** marginal effects (averaged)
- **ICE:** individual effects (disaggregated)

---

## PACKAGE REFERENCE

| Package | Plot Types |
|---------|-----------|
| **ggplot2** | Core geoms (point, line, bar, histogram, density, boxplot, tile, etc.) |
| **ggridges** | Ridgeline plots |
| **vcd** | Mosaic plots |
| **ggalluvial** | Alluvial diagrams |
| **GGally** | ggpairs, parallel coordinates |
| **mi** | Missing data heatmaps |
| **redav** | Aggregated missing patterns |
| **choroplethr** | Choropleth maps |
| **statebins** | Tile grid maps |
| **geofacet** | Geographic faceting |
| **pdp** | Partial dependence plots |
| **lime** | LIME explanations |
| **fastshap** | Shapley values |
| **vip** | Variable importance |
| **rpart.plot** | Decision tree plots |
| **ggparty** | Decision tree plots |
| **car** | Added variable plots |

---

