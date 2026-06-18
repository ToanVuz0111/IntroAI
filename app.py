from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from course_recommender.config import load_config
from course_recommender.recommendation import load_bundle, recommend
from course_recommender.utils import load_json


st.set_page_config(page_title="MOOCCube Course Recommender", page_icon="🎓", layout="wide")
st.title("🎓 Intelligent Course Recommendation")
st.caption("MOOCCube → BERT → learner context → Mahalanobis → Actor–Critic/PPO → English top-K")

mode = st.sidebar.selectbox("Runtime", ["Full BERT smoke", "Fast offline", "Full experiment"])
config_name = {
    "Full BERT smoke": "full_smoke.yaml",
    "Fast offline": "fast_demo.yaml",
    "Full experiment": "full_experiment.yaml",
}[mode]
config = load_config(ROOT / "configs" / config_name)
artifacts = ROOT / config["artifacts_dir"]
page = st.sidebar.radio("Navigation", [
    "Project Overview",
    "Dataset Explorer",
    "Learner Profile",
    "Course Recommendations",
    "Interactive Feedback",
    "Model Evaluation",
    "Ablation Lab",
    "Explainability",
    "Research Notes",
])
if st.sidebar.button("Reset session"):
    st.session_state.clear()
    st.rerun()

required = artifacts / "course_embeddings.npy"
if not required.exists():
    st.warning(f"Artifacts for {mode} are missing. Run `scripts/run_pipeline.py --config configs/{config_name}`.")
    st.stop()


@st.cache_data
def load_frames(path: str):
    root = Path(path)
    return (
        pd.read_csv(root / "courses.csv", dtype={"course_id": str}),
        pd.read_csv(root / "users.csv", dtype={"user_id": str}),
        pd.read_csv(root / "train.csv", dtype={"user_id": str, "course_id": str}),
        pd.read_csv(root / "val.csv", dtype={"user_id": str, "course_id": str}),
        pd.read_csv(root / "test.csv", dtype={"user_id": str, "course_id": str}),
    )


@st.cache_resource
def cached_bundle(config_path: str):
    return load_bundle(load_config(config_path))


courses, users, train, validation, test = load_frames(str(artifacts))
bundle = cached_bundle(str(ROOT / "configs" / config_name))
user_ids = bundle["user_ids"]

if page == "Project Overview":
    st.subheader("Architecture and reproduction status")
    st.code(
        "MOOCCube → cleaning/anonymization → BERT 768-D + learner 10-D → state 778-D\n"
        "→ Mahalanobis/PCA + clustering → catalog Actor–Critic → PPO/GAE → top-K translation"
    )
    cols = st.columns(4)
    cols[0].metric("Courses", f"{len(courses):,}")
    cols[1].metric("Users", f"{len(users):,}")
    cols[2].metric("Interactions", f"{len(train) + len(validation) + len(test):,}")
    cols[3].metric("Embedding", config["embedding"]["backend"].upper())
    st.success("Real local multilingual BERT is active." if config["embedding"]["backend"] == "bert" else "Hash embedding fallback is active.")
    st.info("Paper-reported metrics and experimental metrics are never mixed. Public MOOCCube lacks true quiz labels.")

elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    cols = st.columns(3)
    cols[0].metric("Missing course titles", int(courses["title"].isna().sum()))
    cols[1].metric("Derived interactions", f"{100 * train['is_derived'].astype(str).str.lower().eq('true').mean():.1f}%")
    sparsity = 1.0 - len(train) / max(len(users) * len(courses), 1)
    cols[2].metric("Train sparsity", f"{sparsity:.4%}")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(train, x="reward", nbins=30, title="Reward distribution"), use_container_width=True)
    right.plotly_chart(px.histogram(train.groupby("user_id").size(), nbins=30, title="Interactions per learner"), use_container_width=True)
    timeline = train.assign(timestamp=pd.to_datetime(train["timestamp"], errors="coerce")).dropna(subset=["timestamp"])
    if not timeline.empty:
        monthly = timeline.set_index("timestamp").resample("ME").size().reset_index(name="interactions")
        st.plotly_chart(px.line(monthly, x="timestamp", y="interactions", title="Interaction timeline"), use_container_width=True)
    st.dataframe(courses[["course_id", "title", "category", "difficulty", "total_videos"]].head(200), use_container_width=True)
    eda_path = artifacts / "eda_correlations.csv"
    if eda_path.exists():
        st.subheader("Statistical analysis")
        correlations = pd.read_csv(eda_path)
        st.plotly_chart(
            px.bar(correlations, x="feature", y="pearson_r", title="Pearson correlation with reward"),
            use_container_width=True,
        )
        if (artifacts / "eda_outliers.csv").exists():
            st.dataframe(pd.read_csv(artifacts / "eda_outliers.csv"), use_container_width=True)

elif page == "Learner Profile":
    st.subheader("Learner Profile")
    selected = st.selectbox("Learner", user_ids, key="profile_user")
    index = user_ids.index(selected)
    feature_names = load_json(artifacts / "feature_meta.json")["columns"]
    values = bundle["user_features"][index]
    profile = pd.DataFrame({"feature": feature_names, "value": values})
    st.plotly_chart(px.bar(profile, x="feature", y="value", range_y=[0, 1]), use_container_width=True)
    history = train[train["user_id"] == selected].merge(courses[["course_id", "title"]], on="course_id", how="left")
    st.dataframe(history[["timestamp", "title", "completion_rate", "quiz_score", "engagement_time", "reward"]], use_container_width=True)
    st.caption(f"Semantic profile norm: {np.linalg.norm(bundle['profiles'][index]):.4f}")

elif page == "Course Recommendations":
    st.subheader("Top-K Course Recommendations")
    selected = st.selectbox("Learner", user_ids, key="recommend_user")
    top_k = st.slider("Top-K", 5, 20, 10)
    config["recommendation"]["semantic_weight"] = st.slider("Semantic weight", 0.0, 1.0, float(config["recommendation"]["semantic_weight"]), 0.05)
    config["mahalanobis"]["logit_fusion_alpha"] = st.slider("Mahalanobis α", 0.0, 1.0, float(config["mahalanobis"]["logit_fusion_alpha"]), 0.05)
    config["mahalanobis"]["logit_fusion_beta"] = st.slider("Cluster β", 0.0, 1.0, float(config["mahalanobis"]["logit_fusion_beta"]), 0.05)
    if st.button("Generate recommendations", type="primary"):
        with st.spinner("Ranking the catalog, then translating only the final top-K…"):
            result = recommend(config, selected, top_k)
        st.session_state["last_recommendations"] = result
    result = st.session_state.get("last_recommendations")
    if result is not None:
        for row in result.itertuples(index=False):
            with st.container(border=True):
                st.markdown(f"### {row.rank}. {row.title_en}")
                st.caption(row.original_title_zh)
                st.write(row.description_en)
                st.progress(float(np.clip((row.final_score + 1) / 2, 0, 1)), text=f"Final score: {row.final_score:.4f}")
        st.download_button("Export recommendations CSV", result.to_csv(index=False).encode("utf-8-sig"), "recommendations.csv", "text/csv")

elif page == "Interactive Feedback":
    st.subheader("Interactive Feedback")
    result = st.session_state.get("last_recommendations")
    if result is None:
        st.info("Generate recommendations first.")
    else:
        title = st.selectbox("Recommended course", result["title_en"].tolist())
        completion = st.slider("Completion rate", 0.0, 1.0, 0.8)
        quiz = st.slider("Quiz score", 0.0, 1.0, 0.8)
        engagement = st.slider("Engagement", 0.0, 1.0, 0.7)
        liked = st.toggle("Like", True)
        reward = 0.4 * completion + 0.4 * quiz + 0.2 * engagement
        if not liked:
            reward *= 0.25
        st.metric("Computed reward", f"{reward:.3f}")
        st.plotly_chart(px.bar(
            pd.DataFrame({"component": ["Completion", "Quiz", "Engagement"], "contribution": [0.4 * completion, 0.4 * quiz, 0.2 * engagement]}),
            x="component", y="contribution", title="Reward decomposition",
        ), use_container_width=True)
        st.caption("This updates the displayed feedback state only; BERT/PPO are not retrained on each Streamlit rerun.")

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    evaluation = artifacts / "evaluation.csv"
    baseline = artifacts / "baseline_results.csv"
    if evaluation.exists():
        frame = pd.read_csv(evaluation)
        st.dataframe(frame, use_container_width=True)
        st.plotly_chart(px.bar(frame, x="k", y=["precision", "recall", "mrr", "ndcg"], barmode="group"), use_container_width=True)
    if baseline.exists():
        frame = pd.read_csv(baseline)
        st.plotly_chart(px.bar(frame, x="model", y="recall", color="k", barmode="group", title="Baseline comparison"), use_container_width=True)
    history = artifacts / "ppo_training_history.csv"
    if history.exists():
        frame = pd.read_csv(history)
        st.plotly_chart(px.line(frame, x="iteration", y=["mean_reward", "policy_loss", "value_loss", "entropy"]), use_container_width=True)
    efficiency = artifacts / "efficiency_summary.json"
    if efficiency.exists():
        stats = load_json(efficiency)
        columns = st.columns(4)
        columns[0].metric("Mean inference", f"{stats['mean_inference_seconds']:.4f}s")
        columns[1].metric("P95 inference", f"{stats['p95_inference_seconds']:.4f}s")
        columns[2].metric("Total parameters", f"{stats['total_parameters']:,}")
        columns[3].metric("Trainable parameters", f"{stats['trainable_parameters']:,}")
    st.caption("All values above are experimental results, not paper-reported results.")

elif page == "Ablation Lab":
    st.subheader("Ablation Lab")
    path = artifacts / "ablation_results.csv"
    if path.exists():
        frame = pd.read_csv(path)
        st.dataframe(frame, use_container_width=True)
        st.plotly_chart(px.bar(frame, x="variant", y="recall", color="k", barmode="group"), use_container_width=True)
    else:
        st.info("Run `python scripts/run_ablation.py --config configs/full_smoke.yaml`.")
    five_seed = artifacts / "ablation_five_seed_summary.csv"
    if five_seed.exists():
        st.subheader("Five-seed mean ± standard deviation")
        st.dataframe(pd.read_csv(five_seed), use_container_width=True)

elif page == "Explainability":
    st.subheader("Proxy explanation and score decomposition")
    result = st.session_state.get("last_recommendations")
    if result is None:
        st.info("Generate recommendations first.")
    else:
        selected = st.selectbox("Course", result["title_en"].tolist(), key="explain_course")
        row = result[result["title_en"] == selected].iloc[0]
        values = pd.DataFrame({
            "component": ["Actor policy", "Semantic", "Mahalanobis", "Cluster"],
            "value": [
                row["actor_score"],
                config["recommendation"]["semantic_weight"] * row["semantic_similarity"],
                config["mahalanobis"]["logit_fusion_alpha"] * row["mahalanobis_similarity"],
                config["mahalanobis"]["logit_fusion_beta"] * row["cluster_affinity"],
            ],
        })
        st.plotly_chart(px.bar(values, x="component", y="value", title="Score decomposition"), use_container_width=True)
        st.warning("This is a proxy explanation of the implemented score, not a causal explanation.")

elif page == "Research Notes":
    st.subheader("Research Notes")
    for document in ["paper_analysis.md", "assumptions.md", "reproduction_report.md"]:
        with st.expander(document, expanded=document == "paper_analysis.md"):
            st.markdown((ROOT / "docs" / document).read_text(encoding="utf-8"))
