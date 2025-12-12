# app.R
library(shiny)
library(readr)
library(dplyr)
library(plotly)

# ---------- Helpers ----------
`%||%` <- function(a, b) if (!is.null(a) && !is.na(a) && a != "") a else b

risk_band <- function(p) {
  if (is.null(p) || is.na(p)) return("unknown")
  if (p < 0.30) return("low")
  if (p < 0.70) return("mid")
  "high"
}

risk_emoji <- function(p) {
  band <- risk_band(p)
  if (band == "low") return("🎉")
  if (band == "mid") return("⚠️")
  if (band == "high") return("🌡️")
  "❓"
}

# ---------- Recommended defaults (approx for a ~2000 kcal diet) ----------
DEFAULTS <- list(
  age  = 30,
  bmi  = 23,
  carb = 275,
  fat  = 65,
  
  prot_f = 55,
  prot_m = 65,
  
  # Vitamin A: IU (approx; preformed retinol)
  vita_f = 2300,
  vita_m = 3000,
  
  vitc_f = 75,
  vitc_m = 90,
  
  iron_f = 18,
  iron_m = 8
)

HEALTH_BANDS <- list(
  age  = c(18, 90),
  bmi  = c(18.5, 24.9),
  carb = c(225, 325),
  prot = c(50, 75),
  fat  = c(44, 78),
  
  vita_max = 10000,  # IU (practical reference)
  vitc_max = 200,    # mg (app band)
  iron_max = 45      # mg (practical bound)
)

# ---------- 1. Load data and prepare ----------
crc <- read_csv("crc_dataset.csv", show_col_types = FALSE)

crc <- crc |>
  mutate(
    CRC_Risk = as.factor(CRC_Risk),
    Gender = as.factor(Gender),
    Lifestyle = as.factor(Lifestyle),
    Ethnicity = as.factor(Ethnicity),
    Family_History_CRC = as.factor(Family_History_CRC),
    `Pre-existing Conditions` = as.factor(`Pre-existing Conditions`)
  )

# Fit a logistic regression
crc_model <- glm(
  CRC_Risk ~ Age + BMI +
    `Carbohydrates (g)` + `Proteins (g)` + `Fats (g)` +
    `Vitamin A (IU)` + `Vitamin C (mg)` + `Iron (mg)` +
    Gender + Lifestyle + Ethnicity + Family_History_CRC +
    `Pre-existing Conditions`,
  data = crc,
  family = binomial
)

# ---------- 2. UI ----------
ui <- fluidPage(
  tags$head(
    # Static favicon - cigarette emoji
    tags$link(
      rel = "icon",
      href = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚬</text></svg>"
    ),
    
    tags$style(HTML("
      body { font-size: 14px; }
      .form-group { margin-bottom: 8px; }
      .control-label { font-size: 12px; font-weight: 600; margin-bottom: 3px; }
      .form-control { height: 32px; font-size: 13px; }
      select.form-control { height: 32px; padding-top: 4px; padding-bottom: 4px; }
      .sidebar-panel-custom {
        padding: 15px;
        max-height: 95vh;
        overflow-y: auto;
      }
      .main-panel-custom { padding: 15px; }
      h4 { margin-top: 5px; margin-bottom: 15px; font-size: 18px; }
      .btn-predict { margin-top: 10px; }
      .sim-title { font-size: 13px; font-weight: bold; margin-top: 0; margin-bottom: 8px; }
    "))
  ),
  
  titlePanel("CRC Risk Prediction", windowTitle = "CRC Risk Prediction"),
  
  fluidRow(
    # Left Panel
    column(
      4,
      class = "sidebar-panel-custom",
      
      h4("Enter Participant Details"),
      
      fluidRow(
        column(6,
               numericInput("Age", "Age",
                            value = NA,
                            min = HEALTH_BANDS$age[1], max = HEALTH_BANDS$age[2])
        ),
        column(6,
               numericInput("BMI", "BMI",
                            value = NA,
                            min = 10, max = 60, step = 0.1)
        )
      ),
      
      fluidRow(
        column(6,
               numericInput("Carb", "Carbohydrates (g)",
                            value = NA,
                            min = 0, max = 600, step = 1)
        ),
        column(6,
               numericInput("Prot", "Proteins (g)",
                            value = NA,
                            min = 0, max = 250, step = 1)
        )
      ),
      
      fluidRow(
        column(6,
               numericInput("Fat", "Fats (g)",
                            value = NA,
                            min = 0, max = 200, step = 1)
        ),
        column(6,
               numericInput("VitA", "Vitamin A (IU)",
                            value = NA,
                            min = 0, max = 15000, step = 50)
        )
      ),
      
      fluidRow(
        column(6,
               numericInput("VitC", "Vitamin C (mg)",
                            value = NA,
                            min = 0, max = 2000, step = 1)
        ),
        column(6,
               numericInput("Iron", "Iron (mg)",
                            value = NA,
                            min = 0, max = 60, step = 0.1)
        )
      ),
      
      fluidRow(
        column(6,
               selectInput("Gender", "Gender",
                           choices = c("Select..." = "", levels(crc$Gender)))
        ),
        column(6,
               selectInput("Lifestyle", "Lifestyle",
                           choices = c("Select..." = "", levels(crc$Lifestyle)))
        )
      ),
      
      fluidRow(
        column(6,
               selectInput("Ethnicity", "Ethnicity",
                           choices = c("Select..." = "", levels(crc$Ethnicity)))
        ),
        column(6,
               selectInput("FH", "Family History",
                           choices = c("Select..." = "", levels(crc$Family_History_CRC)))
        )
      ),
      
      selectInput("PreCond", "Pre-existing Conditions",
                  choices = c("Select..." = "", levels(crc$`Pre-existing Conditions`))),
      
      actionButton("predict_btn", "Predict Risk",
                   class = "btn btn-primary btn-block btn-predict")
    ),
    
    # Right Panel
    column(
      8,
      class = "main-panel-custom",
      
      div(style = "background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;",
          h4(style = "margin-top: 0; margin-bottom: 10px;", "Prediction Result"),
          verbatimTextOutput("prediction_text")
      ),
      
      h4(style = "margin-top: 10px; margin-bottom: 10px;", "Risk Visualizations"),
      
      fluidRow(
        style = "margin-bottom: 15px;",
        column(4, plotlyOutput("risk_gauge", height = "240px")),
        column(4, plotlyOutput("distance_healthy", height = "240px")),
        
        # First row, 3rd tile: simulator with STATIC title
        column(
          4,
          div(style = "background-color: #f8f9fa; padding: 10px; border-radius: 5px; height: 240px;",
              tags$div(class = "sim-title", "Risk Prediction Simulator"),
              selectInput("whatif_feature", "Select Feature:",
                          choices = c("BMI", "Carbohydrates (g)", "Proteins (g)", "Fats (g)",
                                      "Vitamin A (IU)", "Vitamin C (mg)", "Iron (mg)"),
                          selected = "BMI"),
              sliderInput("whatif_value", "Adjusted Value:",
                          min = 0, max = 100, value = 50, step = 1),
              verbatimTextOutput("whatif_result", placeholder = TRUE),
              tags$style(HTML("#whatif_result { font-size: 11px; padding: 8px; max-height: 60px; overflow-y: auto; }"))
          )
        )
      ),
      
      fluidRow(
        column(4, plotlyOutput("risk_contribution", height = "240px")),
        column(4, plotlyOutput("risk_trajectory", height = "240px")),
        column(4, plotlyOutput("low_risk_comparison", height = "240px"))
      )
    )
  )
)

# ---------- 3. Server ----------
server <- function(input, output, session) {
  
  prediction_results <- reactiveValues(
    prob = NULL,
    class_pred = NULL,
    new_row = NULL,
    history = list()
  )
  
  # Detect sex from Gender label
  sex_flags <- reactive({
    g <- tolower(trimws(input$Gender %||% ""))
    list(
      is_female = grepl("^f", g) || grepl("woman|female", g),
      is_male   = grepl("^m", g) || grepl("man|male", g)
    )
  })
  
  # Sex-specific recommended defaults - NOT USED ANYMORE since inputs start blank
  # Keeping the code structure in case you want to use it later
  
  # Update what-if slider ranges
  observeEvent(input$whatif_feature, {
    flags <- sex_flags()
    
    vita_default <- if (flags$is_male) DEFAULTS$vita_m else DEFAULTS$vita_f
    vitc_default <- if (flags$is_male) DEFAULTS$vitc_m else DEFAULTS$vitc_f
    iron_default <- if (flags$is_male) DEFAULTS$iron_m else DEFAULTS$iron_f
    prot_default <- if (flags$is_male) DEFAULTS$prot_m else DEFAULTS$prot_f
    
    ranges <- list(
      "BMI" = c(15, 45, DEFAULTS$bmi),
      "Carbohydrates (g)" = c(0, 600, DEFAULTS$carb),
      "Proteins (g)" = c(0, 250, prot_default),
      "Fats (g)" = c(0, 200, DEFAULTS$fat),
      "Vitamin A (IU)" = c(0, 15000, vita_default),
      "Vitamin C (mg)" = c(0, 2000, vitc_default),
      "Iron (mg)" = c(0, 60, iron_default)
    )
    
    rv <- ranges[[input$whatif_feature]]
    updateSliderInput(session, "whatif_value",
                      min = rv[1], max = rv[2], value = rv[3])
  })
  
  # What-if result text with DYNAMIC IMPACT emoji
  output$whatif_result <- renderText({
    req(prediction_results$new_row, input$whatif_value)
    
    modified_row <- prediction_results$new_row
    modified_row[[input$whatif_feature]] <- as.numeric(input$whatif_value)
    
    new_prob <- predict(crc_model, newdata = modified_row, type = "response")
    current_prob <- prediction_results$prob
    
    diff <- new_prob - current_prob
    diff_pct <- if (current_prob > 0) (diff / current_prob) * 100 else NA_real_
    
    # Dynamic emoji based on impact
    impact_emoji <- if (abs(diff) < 0.01) {
      "⚠️"
    } else if (diff < 0) {
      "🎉"
    } else {
      "🌡️"
    }
    
    impact_text <- if (abs(diff) < 0.01) {
      "Minimal impact"
    } else if (diff < 0) {
      "Lower risk (Favorable)"
    } else {
      "Higher risk (Unfavorable)"
    }
    
    paste0(
      "Current Risk: ", round(current_prob * 100, 1), "%\n",
      "Predicted Risk: ", round(new_prob * 100, 1), "%\n",
      "Change: ", ifelse(diff > 0, "+", ""), round(diff * 100, 1), "% ",
      if (!is.na(diff_pct)) paste0("(", ifelse(diff > 0, "+", ""), round(diff_pct, 1), "%)\n") else "\n",
      "Impact: ", impact_emoji, " ", impact_text
    )
  })
  
  observeEvent(input$predict_btn, {
    
    # Check for missing values
    if (is.null(input$Age) || is.na(input$Age) ||
        is.null(input$BMI) || is.na(input$BMI) ||
        is.null(input$Carb) || is.na(input$Carb) ||
        is.null(input$Prot) || is.na(input$Prot) ||
        is.null(input$Fat) || is.na(input$Fat) ||
        is.null(input$VitA) || is.na(input$VitA) ||
        is.null(input$VitC) || is.na(input$VitC) ||
        is.null(input$Iron) || is.na(input$Iron) ||
        input$Gender == "" || input$Lifestyle == "" ||
        input$Ethnicity == "" || input$FH == "" || input$PreCond == "") {
      
      showModal(modalDialog(
        title = "Missing Information",
        "Please fill in all fields before making a prediction.",
        easyClose = TRUE,
        footer = modalButton("OK")
      ))
      return()
    }
    
    # Validate realistic ranges
    validation_errors <- c()
    
    if (input$Age < 18 || input$Age > 90) {
      validation_errors <- c(validation_errors, "• Age must be between 18-90 years")
    }
    if (input$BMI < 15 || input$BMI > 45) {
      validation_errors <- c(validation_errors, "• BMI must be between 15-45")
    }
    if (input$Carb < 50 || input$Carb > 600) {
      validation_errors <- c(validation_errors, "• Carbohydrates must be between 50-600 g/day")
    }
    if (input$Prot < 20 || input$Prot > 250) {
      validation_errors <- c(validation_errors, "• Proteins must be between 20-250 g/day")
    }
    if (input$Fat < 20 || input$Fat > 200) {
      validation_errors <- c(validation_errors, "• Fats must be between 20-200 g/day")
    }
    if (input$VitA < 1000 || input$VitA > 15000) {
      validation_errors <- c(validation_errors, "• Vitamin A must be between 1,000-15,000 IU/day")
    }
    if (input$VitC < 10 || input$VitC > 500) {
      validation_errors <- c(validation_errors, "• Vitamin C must be between 10-500 mg/day")
    }
    if (input$Iron < 5 || input$Iron > 60) {
      validation_errors <- c(validation_errors, "• Iron must be between 5-60 mg/day")
    }
    
    if (length(validation_errors) > 0) {
      showModal(modalDialog(
        title = "⚠️ Invalid Input Values",
        HTML(paste0(
          "<p><strong>Please enter realistic values:</strong></p>",
          "<p style='margin-left: 15px; line-height: 1.8;'>",
          paste(validation_errors, collapse = "<br>"),
          "</p>",
          "<p style='margin-top: 15px;'><em>Tip: These ranges reflect normal adult dietary intake.</em></p>"
        )),
        easyClose = TRUE,
        footer = modalButton("OK")
      ))
      return()
    }
    
    new_row <- data.frame(
      Age = as.numeric(input$Age),
      BMI = as.numeric(input$BMI),
      `Carbohydrates (g)` = as.numeric(input$Carb),
      `Proteins (g)` = as.numeric(input$Prot),
      `Fats (g)` = as.numeric(input$Fat),
      `Vitamin A (IU)` = as.numeric(input$VitA),
      `Vitamin C (mg)` = as.numeric(input$VitC),
      `Iron (mg)` = as.numeric(input$Iron),
      Gender = factor(input$Gender, levels = levels(crc$Gender)),
      Lifestyle = factor(input$Lifestyle, levels = levels(crc$Lifestyle)),
      Ethnicity = factor(input$Ethnicity, levels = levels(crc$Ethnicity)),
      Family_History_CRC = factor(input$FH, levels = levels(crc$Family_History_CRC)),
      `Pre-existing Conditions` = factor(input$PreCond, levels = levels(crc$`Pre-existing Conditions`)),
      check.names = FALSE
    )
    
    prob <- as.numeric(predict(crc_model, newdata = new_row, type = "response"))
    class_pred <- ifelse(prob >= 0.5, "High risk", "Low risk")
    
    prediction_results$prob <- prob
    prediction_results$class_pred <- class_pred
    prediction_results$new_row <- new_row
    
    prediction_results$history[[length(prediction_results$history) + 1]] <- list(
      timestamp = Sys.time(),
      risk = prob,
      bmi = as.numeric(input$BMI),
      age = as.numeric(input$Age)
    )
    
    output$prediction_text <- renderText({
      emo <- risk_emoji(prob)
      paste0(
        emo, " Predicted CRC risk probability: ", round(prob, 4), " (", round(prob * 100, 2), "%)\n",
        "Class (cutoff 0.5): ", class_pred
      )
    })
  })
  
  # Risk Gauge
  output$risk_gauge <- renderPlotly({
    req(prediction_results$prob)
    prob <- prediction_results$prob
    emo <- risk_emoji(prob)
    
    fig <- plot_ly(
      type = "indicator",
      mode = "gauge+number+delta",
      value = prob * 100,
      title = list(text = paste0(emo, " Risk Score (%)"), font = list(size = 16)),
      delta = list(reference = 50),
      gauge = list(
        axis = list(range = list(0, 100), tickwidth = 1, tickfont = list(size = 10)),
        bar = list(color = ifelse(prob < 0.3, "green", ifelse(prob < 0.7, "orange", "red"))),
        steps = list(
          list(range = c(0, 30), color = "lightgreen"),
          list(range = c(30, 70), color = "lightyellow"),
          list(range = c(70, 100), color = "lightcoral")
        ),
        threshold = list(
          line = list(color = "red", width = 3),
          thickness = 0.75,
          value = 50
        )
      )
    ) |>
      layout(margin = list(l = 10, r = 10, t = 40, b = 10))
    
    fig
  })
  
  # Distance from Healthy Range (sex-specific mins for Vit A, Vit C, Iron)
  output$distance_healthy <- renderPlotly({
    req(prediction_results$new_row)
    
    flags <- sex_flags()
    
    vita_min <- if (flags$is_male) DEFAULTS$vita_m else DEFAULTS$vita_f
    vita_max <- HEALTH_BANDS$vita_max
    
    vitc_min <- if (flags$is_male) DEFAULTS$vitc_m else DEFAULTS$vitc_f
    vitc_max <- HEALTH_BANDS$vitc_max
    
    iron_min <- if (flags$is_male) DEFAULTS$iron_m else DEFAULTS$iron_f
    iron_max <- HEALTH_BANDS$iron_max
    
    health_data <- data.frame(
      Feature = c("Age", "BMI", "Carbs", "Protein", "Fat", "Vit A", "Vit C", "Iron"),
      User_Value = c(
        as.numeric(input$Age),
        as.numeric(input$BMI),
        as.numeric(input$Carb),
        as.numeric(input$Prot),
        as.numeric(input$Fat),
        as.numeric(input$VitA),
        as.numeric(input$VitC),
        as.numeric(input$Iron)
      ),
      Min_Healthy = c(
        HEALTH_BANDS$age[1],
        HEALTH_BANDS$bmi[1],
        HEALTH_BANDS$carb[1],
        HEALTH_BANDS$prot[1],
        HEALTH_BANDS$fat[1],
        vita_min,
        vitc_min,
        iron_min
      ),
      Max_Healthy = c(
        HEALTH_BANDS$age[2],
        HEALTH_BANDS$bmi[2],
        HEALTH_BANDS$carb[2],
        HEALTH_BANDS$prot[2],
        HEALTH_BANDS$fat[2],
        vita_max,
        vitc_max,
        iron_max
      )
    )
    
    health_data <- health_data |>
      mutate(
        Distance_Pct = case_when(
          User_Value < Min_Healthy ~ ((User_Value - Min_Healthy) / Min_Healthy) * 100,
          User_Value > Max_Healthy ~ ((User_Value - Max_Healthy) / Max_Healthy) * 100,
          TRUE ~ 0
        ),
        Status_Color = case_when(
          Distance_Pct == 0 ~ "lightgreen",
          abs(Distance_Pct) < 20 ~ "lightyellow",
          TRUE ~ "coral"
        )
      )
    
    plot_ly(
      health_data,
      x = ~Feature,
      y = ~Distance_Pct,
      type = "bar",
      marker = list(color = ~Status_Color),
      text = ~paste0(ifelse(Distance_Pct > 0, "+", ""), round(Distance_Pct, 1), "%"),
      textposition = "outside",
      textfont = list(size = 8),
      showlegend = FALSE
    ) |>
      layout(
        title = list(text = "Nutrient Balance Assessment", font = list(size = 13)),
        xaxis = list(title = "", tickangle = -45, tickfont = list(size = 8)),
        yaxis = list(
          title = "% Off",
          titlefont = list(size = 10),
          tickfont = list(size = 8),
          zeroline = TRUE,
          zerolinecolor = "green",
          zerolinewidth = 2
        ),
        margin = list(l = 40, r = 20, t = 40, b = 60)
      )
  })
  
  # Risk Contribution Breakdown (donut)
  output$risk_contribution <- renderPlotly({
    req(prediction_results$new_row, prediction_results$prob)
    
    coefs <- coef(crc_model)
    
    contributions <- data.frame(
      Factor = character(),
      Contribution = numeric(),
      stringsAsFactors = FALSE
    )
    
    numeric_features <- c("Age", "BMI", "Carbohydrates (g)", "Proteins (g)",
                          "Fats (g)", "Vitamin A (IU)", "Vitamin C (mg)", "Iron (mg)")
    feature_names <- c("Age", "BMI", "Carbs", "Protein", "Fat", "Vit A", "Vit C", "Iron")
    
    user_vals <- c(as.numeric(input$Age), as.numeric(input$BMI), as.numeric(input$Carb),
                   as.numeric(input$Prot), as.numeric(input$Fat), as.numeric(input$VitA),
                   as.numeric(input$VitC), as.numeric(input$Iron))
    
    dataset_means <- c(mean(crc$Age, na.rm = TRUE), mean(crc$BMI, na.rm = TRUE),
                       mean(crc$`Carbohydrates (g)`, na.rm = TRUE),
                       mean(crc$`Proteins (g)`, na.rm = TRUE),
                       mean(crc$`Fats (g)`, na.rm = TRUE),
                       mean(crc$`Vitamin A (IU)`, na.rm = TRUE),
                       mean(crc$`Vitamin C (mg)`, na.rm = TRUE),
                       mean(crc$`Iron (mg)`, na.rm = TRUE))
    
    for (i in seq_along(numeric_features)) {
      if (!is.na(user_vals[i]) && !is.na(dataset_means[i]) && numeric_features[i] %in% names(coefs)) {
        deviation <- user_vals[i] - dataset_means[i]
        coef_val <- as.numeric(coefs[numeric_features[i]])
        contrib <- abs(deviation * coef_val)
        
        if (!is.na(contrib) && !is.nan(contrib)) {
          contributions <- rbind(
            contributions,
            data.frame(Factor = feature_names[i], Contribution = contrib, stringsAsFactors = FALSE)
          )
        }
      }
    }
    
    coef_names <- names(coefs)
    pick_coef <- function(prefix, level_val) {
      hits <- coef_names[grepl(prefix, coef_names, fixed = TRUE) & grepl(level_val, coef_names, fixed = TRUE)]
      if (length(hits) > 0) hits[1] else NA_character_
    }
    
    if (input$Gender != "") {
      nm <- pick_coef("Gender", input$Gender)
      if (!is.na(nm)) contributions <- rbind(contributions, data.frame(Factor = "Gender", Contribution = abs(as.numeric(coefs[nm]))))
    }
    if (input$Lifestyle != "") {
      nm <- pick_coef("Lifestyle", input$Lifestyle)
      if (!is.na(nm)) contributions <- rbind(contributions, data.frame(Factor = "Lifestyle", Contribution = abs(as.numeric(coefs[nm]))))
    }
    if (input$FH != "") {
      nm <- pick_coef("Family_History_CRC", input$FH)
      if (!is.na(nm)) contributions <- rbind(contributions, data.frame(Factor = "Family History", Contribution = abs(as.numeric(coefs[nm]))))
    }
    if (input$Ethnicity != "") {
      nm <- pick_coef("Ethnicity", input$Ethnicity)
      if (!is.na(nm)) contributions <- rbind(contributions, data.frame(Factor = "Ethnicity", Contribution = abs(as.numeric(coefs[nm]))))
    }
    if (input$PreCond != "") {
      nm <- pick_coef("Pre-existing Conditions", input$PreCond)
      if (!is.na(nm)) contributions <- rbind(contributions, data.frame(Factor = "Condition", Contribution = abs(as.numeric(coefs[nm]))))
    }
    
    if (nrow(contributions) == 0 || sum(contributions$Contribution, na.rm = TRUE) == 0) {
      return(
        plot_ly() |>
          layout(
            title = list(text = "Risk Contribution %", font = list(size = 13)),
            xaxis = list(visible = FALSE),
            yaxis = list(visible = FALSE),
            annotations = list(
              list(
                text = "No significant<br>contributions found",
                x = 0.5, y = 0.5,
                xref = "paper", yref = "paper",
                showarrow = FALSE,
                font = list(size = 11)
              )
            ),
            margin = list(l = 10, r = 10, t = 35, b = 10)
          )
      )
    }
    
    contributions <- contributions |>
      filter(!is.na(Contribution) & !is.nan(Contribution) & Contribution > 0) |>
      arrange(desc(Contribution))
    
    total <- sum(contributions$Contribution, na.rm = TRUE)
    contributions$Percentage <- (contributions$Contribution / total) * 100
    
    plot_ly(
      contributions,
      labels = ~Factor,
      values = ~Percentage,
      type = "pie",
      hole = 0.4,
      textinfo = "label+percent",
      textposition = "outside",
      textfont = list(size = 8),
      marker = list(colors = c("#FF6B6B", "#FFA07A", "#FFD93D", "#6BCB77",
                               "#4D96FF", "#A28089", "#C77DFF", "#E9C46A"))
    ) |>
      layout(
        title = list(text = "Risk Contribution %", font = list(size = 13)),
        showlegend = FALSE,
        margin = list(l = 20, r = 20, t = 40, b = 20)
      )
  })
  
  # Comparison to Low-Risk Profile
  output$low_risk_comparison <- renderPlotly({
    req(prediction_results$new_row)
    
    low_risk_profile <- data.frame(
      Feature = c("Age", "BMI", "Carbs", "Protein", "Fat", "Vit A", "Vit C", "Iron"),
      Your_Value = c(
        as.numeric(input$Age),
        as.numeric(input$BMI),
        as.numeric(input$Carb),
        as.numeric(input$Prot),
        as.numeric(input$Fat),
        as.numeric(input$VitA),
        as.numeric(input$VitC),
        as.numeric(input$Iron)
      ),
      Low_Risk_Target = c(51, 26.2, 277, 86, 70, 5916, 77, 12.9)
    )
    
    low_risk_profile <- low_risk_profile |>
      mutate(
        Your_Normalized = scale(Your_Value)[, 1],
        Target_Normalized = scale(Low_Risk_Target)[, 1]
      )
    
    plot_ly(low_risk_profile, x = ~Feature) |>
      add_trace(y = ~Your_Normalized, name = "You", type = "bar",
                marker = list(color = "coral")) |>
      add_trace(y = ~Target_Normalized, name = "Low-Risk Target", type = "bar",
                marker = list(color = "lightgreen")) |>
      layout(
        title = list(text = "You vs Low-Risk Profile", font = list(size = 13)),
        xaxis = list(title = "", tickangle = -45, tickfont = list(size = 8)),
        yaxis = list(title = "Normalized", titlefont = list(size = 10), tickfont = list(size = 8)),
        barmode = "group",
        showlegend = TRUE,
        legend = list(font = list(size = 8), orientation = "h", y = -0.25),
        margin = list(l = 40, r = 10, t = 35, b = 50)
      )
  })
  
  # Risk Trajectory
  output$risk_trajectory <- renderPlotly({
    history <- prediction_results$history
    
    if (length(history) == 0) {
      return(
        plot_ly() |>
          layout(
            title = list(text = "Risk Trajectory", font = list(size = 13)),
            xaxis = list(visible = FALSE),
            yaxis = list(visible = FALSE),
            annotations = list(
              list(
                text = "Track Your Progress<br><br>Make multiple predictions<br>to see your risk trend.<br><br>Each prediction will add<br>a new point to this chart.",
                x = 0.5, y = 0.5,
                xref = "paper", yref = "paper",
                showarrow = FALSE,
                font = list(size = 10, color = "#666")
              )
            ),
            margin = list(l = 40, r = 10, t = 35, b = 40)
          )
      )
    }
    
    trajectory_data <- data.frame(
      Session = 1:length(history),
      Risk = sapply(history, function(x) x$risk * 100),
      Timestamp = sapply(history, function(x) format(x$timestamp, "%H:%M:%S"))
    )
    
    trend_text <- "First prediction"
    trend_color <- "gray"
    if (length(history) > 1) {
      trend_text <- ifelse(
        trajectory_data$Risk[length(history)] < trajectory_data$Risk[1],
        "Improving Trend",
        ifelse(
          trajectory_data$Risk[length(history)] > trajectory_data$Risk[1],
          "Increasing Trend",
          "Stable Trend"
        )
      )
      trend_color <- ifelse(
        trajectory_data$Risk[length(history)] < trajectory_data$Risk[1],
        "green",
        ifelse(
          trajectory_data$Risk[length(history)] > trajectory_data$Risk[1],
          "red",
          "orange"
        )
      )
    }
    
    y_min <- max(0, min(trajectory_data$Risk) - 10)
    y_max <- min(100, max(trajectory_data$Risk) + 10)
    
    plot_ly(
      trajectory_data,
      x = ~Session, y = ~Risk,
      type = "scatter", mode = "lines+markers",
      line = list(color = "coral", width = 3),
      marker = list(size = 10, color = "coral", line = list(color = "darkred", width = 2)),
      text = ~paste0("Prediction #", Session,
                     "<br>Risk: ", round(Risk, 1), "%",
                     "<br>Time: ", Timestamp),
      hoverinfo = "text",
      name = "Your Risk"
    ) |>
      layout(
        title = list(text = paste0("Risk Trajectory: ", trend_text),
                     font = list(size = 13, color = trend_color)),
        xaxis = list(title = "Prediction #", titlefont = list(size = 10), tickfont = list(size = 8), dtick = 1),
        yaxis = list(title = "Risk %", titlefont = list(size = 10), tickfont = list(size = 8), range = c(y_min, y_max)),
        showlegend = FALSE,
        margin = list(l = 40, r = 10, t = 35, b = 40)
      )
  })
}

shinyApp(ui = ui, server = server)