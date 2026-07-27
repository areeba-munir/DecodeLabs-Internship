# ---------------------------------------------------------
# DecodeLabs Artificial Intelligence Internship
# Project 3: Tech Stack and Career Path Recommender
# ---------------------------------------------------------


from pathlib import Path
from typing import Iterable
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path(__file__).with_name("raw_skills.csv")

REQUIRED_COLUMNS = {
    "role",
    "category",
    "skills",
    "tools",
    "description",
    "next_skills",
}


# Common abbreviations and alternative skill names
SKILL_ALIASES = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "js": "javascript",
    "ts": "typescript",
    "aws cloud": "aws",
    "gcp": "google cloud",
    "powerbi": "power bi",
    "sklearn": "scikit learn",
    "scikit-learn": "scikit learn",
    "gen ai": "generative ai",
    "llms": "large language models",
    "llm": "large language models",
    "api": "apis",
    "data visualisation": "data visualization",
    "visualisation": "visualization",
    "bi": "business intelligence",
    "cloud": "cloud computing",
}
SKILL_CATALOGUE = [
    "python",
    "sql",
    "excel",
    "statistics",
    "data analysis",
    "data cleaning",
    "data visualization",
    "dashboards",
    "reporting",
    "business intelligence",
    "data modelling",
    "machine learning",
    "predictive modelling",
    "deep learning",
    "feature engineering",
    "mlops",
    "algorithms",
    "model deployment",
    "apis",
    "data pipelines",
    "software engineering",
    "artificial intelligence",
    "natural language processing",
    "computer vision",
    "generative ai",
    "large language models",
    "prompt engineering",
    "vector databases",
    "rag",
    "neural networks",
    "mathematics",
    "optimization",
    "object detection",
    "image processing",
    "video analysis",
    "linux",
    "cloud computing",
    "automation",
    "containers",
    "networking",
    "scripting",
    "aws",
    "azure",
    "google cloud",
    "docker",
    "kubernetes",
    "cybersecurity",
    "threat detection",
    "incident response",
    "risk assessment",
    "ethical hacking",
    "vulnerability assessment",
    "web security",
    "databases",
    "data warehousing",
    "big data",
    "etl",
    "testing",
    "api testing",
    "debugging",
    "javascript",
    "html",
    "css",
    "react",
    "java",
    "c++",
    "backend development",
    "frontend development",
    "web development",
    "mobile development",
    "object oriented programming",
    "data structures",
    "system design",
    "cloud architecture",
    "distributed systems",
]


def clean_text(text: str) -> str:
    """
    Convert text to lowercase and remove unnecessary symbols.
    """

    text = str(text).lower().strip()

    # Keep letters, numbers, plus signs, and spaces
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)

    # Replace repeated spaces
    text = " ".join(text.split())

    return text


def normalise_skill(skill: str) -> str:
    """
    Clean a skill and convert common aliases to standard names.
    """

    cleaned_skill = clean_text(skill)

    return SKILL_ALIASES.get(
        cleaned_skill,
        cleaned_skill
    )


def prepare_user_skills(
    user_skills: str | Iterable[str]
) -> list[str]:
    """
    Convert user input into a clean list of unique skills.

    The function accepts either:
    - A comma-separated string
    - A list of skill names
    """

    if isinstance(user_skills, str):
        skills = user_skills.split(",")
    else:
        skills = list(user_skills)

    cleaned_skills = []

    for skill in skills:
        normalised = normalise_skill(skill)

        if normalised and normalised not in cleaned_skills:
            cleaned_skills.append(normalised)

    return cleaned_skills

def phrase_exists(
    phrase: str,
    text: str
) -> bool:
    """Check whether a complete skill phrase exists in text."""

    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"

    return bool(
        re.search(
            pattern,
            clean_text(text)
        )
    )


def extract_role_skills(
    role_skills_text: str
) -> list[str]:
    """Extract recognised skills from a role profile."""

    detected_skills = []

    for skill in SKILL_CATALOGUE:
        if phrase_exists(skill, role_skills_text):
            detected_skills.append(skill)

    return detected_skills


def find_missing_skills(
    user_skills: list[str],
    role_skills_text: str,
    maximum: int = 5
) -> list[str]:
    """
    Find role skills that are not present in the user's profile.
    """

    role_skills = extract_role_skills(
        role_skills_text
    )

    normalised_user_skills = {
        normalise_skill(skill)
        for skill in user_skills
    }

    missing_skills = [
        skill
        for skill in role_skills
        if skill not in normalised_user_skills
    ]

    return missing_skills[:maximum]

def load_role_data(
    file_path: Path = DATA_FILE
) -> pd.DataFrame:
    """
    Load and validate the technology career-role dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if data.empty:
        raise ValueError(
            "The career-role dataset is empty."
        )

    data = data.fillna("")

    # Combine the useful columns into one searchable profile
    data["search_text"] = (
        data[
            [
                "role",
                "category",
                "skills",
                "tools",
            ]
        ]
        .astype(str)
        .agg(" ".join, axis=1)
        .apply(clean_text)
    )

    return data

def find_matched_skills(
    user_skills: list[str],
    role_text: str
) -> list[str]:
    """Find user skills that genuinely appear in a role."""

    return [
        skill
        for skill in user_skills
        if phrase_exists(skill, role_text)
    ]


def create_explanation(
    role: str,
    matched_skills: list[str],
    category: str
) -> str:
    """
    Generate a short explanation for a recommendation.
    """

    if matched_skills:
        readable_skills = ", ".join(matched_skills)

        return (
            f"{role} is recommended because your skills in "
            f"{readable_skills} align with this {category} role."
        )

    return (
        f"{role} is recommended because your overall profile "
        f"shows similarity with this {category} career path."
    )




def recommend_roles(
    user_skills: str | Iterable[str],
    top_n: int = 3
) -> pd.DataFrame:
    """
    Recommend the most relevant technology roles.

    Steps:
    1. Clean the user's skills
    2. Convert role profiles and user input into TF-IDF vectors
    3. Calculate cosine similarity
    4. Rank roles by similarity
    5. Return the top recommendations
    """

    cleaned_skills = prepare_user_skills(
        user_skills
    )

    if len(cleaned_skills) < 3:
        raise ValueError(
            "Please provide at least three different skills "
            "or interests."
        )

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    role_data = load_role_data()

    user_profile = " ".join(cleaned_skills)

    # Add the user profile to the role documents so all text
    # is converted using the same TF-IDF vocabulary
    documents = (
        role_data["search_text"].tolist()
        + [user_profile]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    role_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]

    similarity_scores = cosine_similarity(
        user_vector,
        role_vectors
    ).flatten()

    results = role_data.copy()

    results["similarity_score"] = similarity_scores
    results["match_percentage"] = (
        results["similarity_score"] * 100
    ).round(2)

    results = results.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_n)

    matched_skills_column = []
    skill_gaps_column = []
    explanations = []

    for _, row in results.iterrows():
        matched_skills = find_matched_skills(
            cleaned_skills,
            row["search_text"]
        )

        missing_skills = find_missing_skills(
            cleaned_skills,
            row["skills"],
            maximum=5
        )

        matched_skills_column.append(
            ", ".join(matched_skills)
            if matched_skills
            else "General profile similarity"
        )

        skill_gaps_column.append(
            ", ".join(missing_skills)
            if missing_skills
            else "No major skill gaps identified"
        )

        explanations.append(
            create_explanation(
                role=row["role"],
                matched_skills=matched_skills,
                category=row["category"]
            )
        )

    results["matched_skills"] = matched_skills_column
    results["skill_gaps"] = skill_gaps_column
    results["explanation"] = explanations

    output_columns = [
        "role",
        "category",
        "match_percentage",
        "matched_skills",
        "skill_gaps",
        "tools",
        "description",
        "next_skills",
        "explanation",
    ]

    return results[output_columns].reset_index(
        drop=True
    )
    


def display_recommendations(
    recommendations: pd.DataFrame
) -> None:
    """
    Display recommendation results in the terminal.
    """

    print("\n" + "=" * 70)
    print("TOP CAREER RECOMMENDATIONS")
    print("=" * 70)

    for index, row in recommendations.iterrows():
        print(
            f"\n{index + 1}. {row['role']} "
            f"— {row['match_percentage']:.2f}% match"
        )

        print(f"Category: {row['category']}")
        print(f"Matched skills: {row['matched_skills']}")
        print(f"Relevant tools: {row['tools']}")
        print(f"Suggested next skills: {row['next_skills']}")
        print(f"Reason: {row['explanation']}")

        print("-" * 70)


def main() -> None:
    """
    Run a terminal-based test of the recommendation system.
    """

    print("=" * 70)
    print("TECH STACK AND CAREER PATH RECOMMENDER")
    print("=" * 70)

    print(
        "\nEnter at least three skills or interests, "
        "separated by commas."
    )

    print(
        "Example: Python, SQL, Machine Learning"
    )

    user_input = input("\nYour skills: ")

    try:
        recommendations = recommend_roles(
            user_input,
            top_n=3
        )

        display_recommendations(
            recommendations
        )

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError
    ) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()