# ---------------------------------------------------------
# DecodeLabs Artificial Intelligence Internship
# Project 3: Tech Stack and Career Path Recommender
# Streamlit Application
# ---------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from recommender import (
    load_role_data,
    prepare_user_skills,
    recommend_roles,
)


st.set_page_config(
    page_title="Tech Career Recommender",
    page_icon="🚀",
    layout="wide",
)


COMMON_SKILLS = [
    "Python",
    "SQL",
    "Excel",
    "Statistics",
    "Data Analysis",
    "Data Visualisation",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Natural Language Processing",
    "Computer Vision",
    "Generative AI",
    "Large Language Models",
    "JavaScript",
    "HTML",
    "CSS",
    "React",
    "Java",
    "C++",
    "Backend Development",
    "Frontend Development",
    "Web Development",
    "Mobile Development",
    "APIs",
    "Databases",
    "Linux",
    "Cloud Computing",
    "AWS",
    "Azure",
    "Docker",
    "Kubernetes",
    "DevOps",
    "Automation",
    "Cybersecurity",
    "Networking",
    "Software Testing",
    "Data Engineering",
    "Data Pipelines",
]


@st.cache_data
def get_role_data() -> pd.DataFrame:
    """Load and cache the career-role dataset."""

    return load_role_data()


def combine_user_skills(
    selected_skills: list[str],
    custom_input: str
) -> list[str]:
    """
    Combine selected and manually entered skills.
    """

    custom_skills = prepare_user_skills(
        custom_input
    )

    combined_skills = prepare_user_skills(
        selected_skills + custom_skills
    )

    return combined_skills


def display_recommendation_card(
    rank: int,
    recommendation: pd.Series
) -> None:
    """
    Display one career recommendation.
    """

    match_percentage = float(
        recommendation["match_percentage"]
    )

    with st.container(border=True):
        title_column, score_column = st.columns(
            [4, 1]
        )

        with title_column:
            st.subheader(
                f"{rank}. {recommendation['role']}"
            )

            st.caption(
                recommendation["category"]
            )

        with score_column:
            st.metric(
                label="Similarity score",
                value=f"{match_percentage:.2f}%"
            )

        st.write(
            recommendation["description"]
        )

        first_column, second_column = st.columns(2)

        with first_column:
            st.markdown("**Matched skills**")
            st.write(
                recommendation["matched_skills"]
            )

            st.markdown("**Relevant tools**")
            st.write(
                recommendation["tools"]
            )

        with second_column:
            st.markdown("**Skill gaps to work on**")
            st.write(
               recommendation.get(
                    "skill_gaps",
                    "Skill-gap information is unavailable."
                )
            )
            st.markdown("**Long-term growth skills**")
            st.write(
                    recommendation["next_skills"]
            )

            st.markdown("**Why this role?**")
            st.write(
                recommendation["explanation"]
            )


def main() -> None:
    """Run the Streamlit application."""

    st.title("🚀 Tech Stack & Career Path Recommender")

    st.write(
        "Discover technology roles that align with your "
        "current skills and interests."
    )

    st.info(
        "Select or enter at least three skills. "
        "The system uses TF-IDF and cosine similarity "
        "to generate personalised recommendations."
    )
    st.caption(
        "The recommendations are based on mathematical skill similarity "
        "and are intended as career guidance, not a formal assessment."
    )

    try:
        role_data = get_role_data()

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError
    ) as error:
        st.error(
            f"Unable to load the career dataset: {error}"
        )
        st.stop()

    with st.sidebar:
        st.header("About the system")

        st.write(
            "This is a content-based recommendation "
            "system that compares user skills with "
            "technology career profiles."
        )

        st.markdown(
            """
            **Recommendation process**

            1. Clean and normalise skills  
            2. Build TF-IDF vectors  
            3. Calculate cosine similarity  
            4. Rank career roles  
            5. Display the strongest matches
            """
        )

        st.divider()

        st.metric(
            "Career roles",
            len(role_data)
        )

        st.metric(
            "Career categories",
            role_data["category"].nunique()
        )

    st.subheader("Your skills and interests")

    selected_skills = st.multiselect(
        label="Select your skills",
        options=COMMON_SKILLS,
        placeholder="Choose at least three skills"
    )

    custom_input = st.text_input(
        label="Add other skills",
        placeholder=(
            "Example: Pandas, TensorFlow, Power BI"
        ),
        help=(
            "Separate multiple skills using commas."
        )
    )

    top_n = st.slider(
        label="Number of recommendations",
        min_value=3,
        max_value=5,
        value=3
    )

    user_skills = combine_user_skills(
        selected_skills,
        custom_input
    )

    if user_skills:
        st.write(
            f"**Selected profile:** "
            f"{', '.join(user_skills)}"
        )

    recommend_button = st.button(
        "Generate recommendations",
        type="primary",
        use_container_width=True
    )

    if recommend_button:
        if len(user_skills) < 3:
            st.warning(
                "Please select or enter at least "
                "three different skills."
            )
            return

        try:
            recommendations = recommend_roles(
                user_skills,
                top_n=top_n
            )

        except (
            FileNotFoundError,
            ValueError,
            pd.errors.ParserError
        ) as error:
            st.error(
                f"Recommendation error: {error}"
            )
            return

        st.divider()
        st.header("Your recommended career paths")

        for index, recommendation in recommendations.iterrows():
                display_recommendation_card(
                rank=index + 1,
                recommendation=recommendation
            )


        # This section must be outside the loop
        st.subheader("Recommendation comparison")

        chart_data = recommendations[
            ["role", "match_percentage"]
        ].copy()

        chart_data = chart_data.sort_values(
            by="match_percentage",
            ascending=True
        )

        figure, axis = plt.subplots(figsize=(8, 3.8))

        axis.barh(
            chart_data["role"],
            chart_data["match_percentage"]
        )

        axis.set_title("Career Role Similarity Scores")
        axis.set_xlabel("Cosine Similarity Score (%)")
        axis.set_ylabel("Career Role")
        axis.set_xlim(0, 100)

        for position, score in enumerate(
            chart_data["match_percentage"]
        ):
            axis.text(
                score + 1,
                position,
                f"{score:.2f}%",
                va="center"
            )

        plt.tight_layout()

        st.pyplot(
            figure,
            use_container_width=True
        )

        plt.close(figure)

        download_data = recommendations.to_csv(
                    index=False
        ).encode("utf-8")

        st.download_button(
            label="Download recommendation results",
            data=download_data,
            file_name="career_recommendations.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander(
            "View recommendation data"
        ):
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )


if __name__ == "__main__":
    main()