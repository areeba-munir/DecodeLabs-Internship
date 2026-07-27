# Tech Stack & Career Path Recommender

## Project Overview

The **Tech Stack & Career Path Recommender** is a content-based recommendation system developed as Project 3 of the DecodeLabs Artificial Intelligence Internship 2026.

The application accepts a user’s technical skills and interests, compares them with different technology career profiles, and recommends the most relevant career paths.

It uses **TF-IDF vectorisation** to convert text-based skills into numerical vectors and **cosine similarity** to measure the alignment between the user profile and each career role.

The application also identifies:

- Matched skills
- Missing skills
- Relevant tools
- Long-term learning areas
- Similarity scores
- Reasons behind each recommendation

---

## Project Objective

The goal of this project is to build a simple recommendation system that:

1. Accepts user preferences or interests
2. Converts user and career information into comparable numerical vectors
3. Calculates similarity between the user and available career roles
4. Ranks the roles according to relevance
5. Displays the strongest career recommendations

---

## Problem Statement

Technology contains many different career paths, including:

- Data Science
- Artificial Intelligence
- Data Analytics
- Software Development
- Cloud Computing
- DevOps
- Cybersecurity
- Data Engineering

Beginners often find it difficult to decide which career path aligns with their existing skills.

This project helps reduce that confusion by converting career guidance into a ranked, similarity-based recommendation process.

---

## Recommendation Type

This project uses **content-based filtering**.

Content-based filtering compares the user’s skills directly with the skills and attributes associated with each career role.

For example:

```text
User skills:
Python, SQL, Machine Learning

Possible recommendations:
1. Data Scientist
2. Machine Learning Engineer
3. Data Analyst
```

Unlike collaborative filtering, this approach does not require historical information from thousands of other users.

---

## Key Features

- Interactive Streamlit web interface
- Supports predefined skill selection
- Supports custom skill input
- Requires at least three skills or interests
- Cleans and normalises user input
- Handles common skill abbreviations and aliases
- Converts text into TF-IDF vectors
- Calculates cosine similarity
- Ranks career roles from highest to lowest similarity
- Supports Top-3 to Top-5 recommendations
- Displays similarity scores
- Shows matched user skills
- Performs personalised skill-gap analysis
- Displays relevant tools for each career
- Suggests long-term growth skills
- Provides an explanation for each recommendation
- Displays a career comparison chart
- Allows recommendation results to be downloaded as a CSV file
- Handles missing files and invalid input

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Streamlit
- Matplotlib

---

## Dataset

The project uses a custom dataset named:

```text
raw_skills.csv
```

The dataset contains:

- 24 technology career roles
- 10 career categories
- Core skills
- Relevant tools
- Role descriptions
- Future learning suggestions

### Dataset Columns

| Column | Description |
|---|---|
| `role` | Name of the technology career |
| `category` | Broader career category |
| `skills` | Main skills required for the role |
| `tools` | Technologies and tools associated with the role |
| `description` | Brief explanation of the career |
| `next_skills` | Advanced skills recommended for future growth |

### Career Areas Included

- Artificial Intelligence
- Data Science
- Data Analytics
- Data Engineering
- Software Development
- Cloud and DevOps
- Cybersecurity
- Software Quality
- Product and Business
- Data and Infrastructure

---

## Recommendation Workflow

The application follows an Input–Process–Output architecture.

```text
User skills and interests
            ↓
Input cleaning and normalisation
            ↓
TF-IDF vectorisation
            ↓
Cosine-similarity calculation
            ↓
Career-role ranking
            ↓
Top-N filtering
            ↓
Recommendations and skill-gap analysis
```

### Step 1: Input

The user selects or enters at least three skills.

Example:

```text
Python
SQL
Machine Learning
Data Analysis
```

### Step 2: Input Normalisation

The application:

- Converts skills to lowercase
- Removes unnecessary symbols
- Removes duplicate entries
- Standardises common aliases

Examples:

```text
AI       → artificial intelligence
ML       → machine learning
DL       → deep learning
NLP      → natural language processing
CV       → computer vision
JS       → JavaScript
PowerBI  → Power BI
```

### Step 3: Feature Extraction

The user profile and career-role profiles are converted into TF-IDF vectors.

TF-IDF gives more importance to meaningful and specific terms while reducing the influence of common words.

### Step 4: Similarity Calculation

Cosine similarity calculates the alignment between:

```text
User skill vector
        and
Career-role vector
```

A higher score means the career profile has stronger mathematical alignment with the selected user skills.

### Step 5: Ranking

All career roles are sorted in descending order using their cosine-similarity scores.

### Step 6: Top-N Filtering

The application returns only the strongest recommendations to avoid information overload.

The user can request between three and five recommendations.

### Step 7: Skill-Gap Analysis

The application compares the selected user skills with the skills required for each recommended career.

It then displays:

- Existing matched skills
- Important missing skills
- Relevant tools
- Long-term growth areas

---

## Project Structure

```text
Project-3-Tech-Stack-Recommender/
├── app.py
├── recommender.py
├── raw_skills.csv
├── requirements.txt
├── README.md
└── outputs/
    ├── app_home.png
    ├── recommendations.png
    └── comparison_chart.png
```

### File Descriptions

#### `app.py`

Contains the Streamlit user interface.

It handles:

- Skill selection
- Custom skill input
- Recommendation controls
- Recommendation cards
- Similarity visualisation
- CSV download

#### `recommender.py`

Contains the main recommendation logic.

It handles:

- Dataset loading
- Dataset validation
- Input cleaning
- Skill normalisation
- TF-IDF vectorisation
- Cosine-similarity calculation
- Career ranking
- Matched-skill extraction
- Skill-gap analysis
- Recommendation explanations

#### `raw_skills.csv`

Contains the career-role dataset.

#### `requirements.txt`

Contains the Python dependencies required to run the project.

#### `outputs/`

Contains screenshots of the application and recommendation results.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/areeba-munir/DecodeLabs-Internship.git
```

### 2. Open the Project 3 folder

```bash
cd DecodeLabs-Internship/Project-3-Tech-Stack-Recommender
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate.bat
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

The `requirements.txt` file contains:

```text
pandas
scikit-learn
streamlit
matplotlib
```

---

## How to Run

Run the Streamlit application:

```bash
streamlit run app.py
```

Alternatively:

```bash
python -m streamlit run app.py
```

The application will open in the browser at an address similar to:

```text
http://localhost:8501
```

---

## How to Use

1. Open the Streamlit application.
2. Select at least three skills from the available options.
3. Add custom skills if required.
4. Select the number of recommendations.
5. Click **Generate recommendations**.
6. Review the recommended career paths.
7. Examine the matched skills and skill gaps.
8. Compare the similarity scores in the chart.
9. Download the results as a CSV file if required.

---

## Example User Profile

```text
Python
SQL
Machine Learning
Data Analysis
Statistics
```

Possible recommendations:

```text
1. Data Scientist
2. Machine Learning Engineer
3. Data Analyst
```

Each recommendation contains:

- Career title
- Career category
- Similarity score
- Role description
- Matched skills
- Skill gaps
- Relevant tools
- Long-term growth skills
- Recommendation explanation

---

## Example Recommendation

```text
1. Data Scientist

Category:
Data Science

Similarity score:
33.38%

Matched skills:
Python, SQL, Machine Learning, Data Analysis

Skill gaps:
Statistics, Data Visualisation, Predictive Modelling

Relevant tools:
Pandas, NumPy, Scikit-learn, Jupyter

Long-term growth skills:
Deep Learning, Feature Engineering, MLOps

Reason:
The Data Scientist role is recommended because the user's
Python, SQL, Machine Learning, and Data Analysis skills align
with the main requirements of this career path.
```

---

## Screenshots

### Application Home

![Application Home](outputs/app_home.png)

### Career Recommendations

![Career Recommendations](outputs/recommendations.png)

### Career Similarity Comparison

![Similarity Comparison](outputs/comparison_chart.png)

---

## Input Validation

The application checks that the user provides at least three unique skills.

For insufficient input, it displays:

```text
Please select or enter at least three different skills.
```

The application also handles:

- Missing dataset files
- Empty datasets
- Missing CSV columns
- Invalid recommendation counts
- CSV parsing errors

---

## Similarity Score Interpretation

The displayed scores represent cosine similarity between the selected user skills and career-role profiles.

For example:

```text
Data Analyst                   37.80%
Data Scientist                 33.38%
Business Intelligence Analyst 22.35%
```

The highest score represents the strongest relative alignment among the available roles.

These scores are not formal employment probabilities or guaranteed career-suitability percentages.

---

## Limitations

- Recommendations depend on the quality of the custom dataset.
- The application evaluates skill similarity but does not consider work experience.
- Education, personality, salary expectations, and location are not included.
- Cosine similarity does not measure professional readiness.
- The system currently uses manually defined career profiles.
- User feedback is not yet stored for future recommendations.

---

## Future Improvements

- Expand the dataset with more technology roles
- Add experience-level selection
- Include beginner, intermediate, and advanced skill levels
- Add preferred industry and career-goal filters
- Add course and certification recommendations
- Include salary and market-demand information
- Add user ratings and feedback
- Store recommendation history
- Add collaborative filtering
- Build a hybrid recommendation system
- Integrate live job-market datasets
- Deploy the application publicly
- Add user authentication
- Generate personalised learning roadmaps

---

## Learning Outcomes

Through this project, I practised:

- Recommendation-system concepts
- Content-based filtering
- Input–Process–Output architecture
- Text cleaning and preprocessing
- Skill alias normalisation
- Feature extraction using TF-IDF
- Vector-space representation
- Cosine-similarity calculation
- Ranking and Top-N filtering
- Pandas DataFrame operations
- CSV dataset validation
- Personalised skill-gap analysis
- Recommendation explanations
- Streamlit application development
- Matplotlib visualisation
- Error handling
- Project documentation
- Git and GitHub project organisation

---

## Important Disclaimer

This application provides recommendations based on mathematical similarity between user-selected skills and predefined career profiles.

It is intended for educational and career-exploration purposes only and should not be treated as a formal career assessment or employment guarantee.

---

## Author

**Areeba Munir**

BS Computer Science  
The Islamia University of Bahawalpur

---

## Internship

**DecodeLabs Artificial Intelligence Internship — Batch 2026**

---

## Licence

This project is included in the DecodeLabs Internship repository and is available under the repository’s MIT Licence.