# Student Productivity & Behavior Dashboard

A visual analytics project exploring academic performance drivers in a synthetic 20,000-record student dataset. The project includes an interactive Streamlit dashboard and an exploratory analysis notebook.

Live dashboard: https://data-visualization-project-lcr96ebrjrnpmx94tysxf6.streamlit.app/

## Project Overview

This dashboard investigates how digital habits, academic effort, and wellness indicators relate to student final grades. The analysis focuses on three main questions:

- How do phone usage, social media, YouTube, and gaming relate to final grades?
- Is there a study-hour threshold where focus stops improving, especially under high stress?
- How do top-performing students differ from underperforming students behaviorally?

The dashboard is built as a five-page Streamlit application:

- `Overview`: KPI cards, distributions, correlation ranking, and heatmap.
- `Digital Habits`: screen-time patterns and their relationship with grades.
- `Academic Effort`: study hours, attendance, assignments, and exercise.
- `Wellness and Cognition`: sleep, stress, focus, productivity, and sunburst analysis.
- `About`: methodology, cleaning steps, and data preview.

## Dataset

The project uses the Student Productivity and Behavior Dataset (20K), a synthetic Kaggle dataset licensed under Apache 2.0. It contains 20,000 student records with demographic, digital behavior, academic effort, wellness, and outcome variables.

Main derived features include:

- `screen_time_hours`
- `distraction_ratio`
- `performance_segment`
- `stress_zone`
- `sleep_quality`
- `digital_load`
- `exercise_group`

## Key Findings

- Assignment completion and daily study hours are among the strongest positive predictors of final grade.
- Phone usage and social media show clear negative relationships with academic performance.
- High-stress students show a focus-score plateau around six study hours per day.
- Elite students generally combine higher study effort, better attendance, lower digital distraction, and more balanced wellness patterns.

## How To Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

The notebook `analysis.ipynb` contains the exploratory analysis and mirrors the cleaning logic used in the dashboard.

## Project Files

- `app.py`: main Streamlit dashboard.
- `analysis.ipynb`: exploratory analysis notebook.
- `student_productivity.csv`: dataset used by the dashboard and notebook.
- `requirements.txt`: Python dependencies for running the dashboard.
- `.streamlit/config.toml`: optional Streamlit theme/config file for consistent appearance.



## Team Members

Yusuf Alper İlhan  
Student ID: 150230318

Bahadır Karadağ  
Student ID: 150240326
