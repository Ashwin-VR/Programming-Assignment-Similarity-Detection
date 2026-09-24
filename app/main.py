from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from similarity_investigator.feature_dataset import save_pair_features
from similarity_investigator.graphs import ast_to_dot, cfg_to_dot
from similarity_investigator.ingestion import IngestionError, ingest
from similarity_investigator.models import InputFile, PairResult, Submission
from similarity_investigator.pairwise import compare_all
from similarity_investigator.relationships import relationship_groups
from similarity_investigator.languages import language_key
from similarity_investigator.semantic import CodeBERTEncoder
from similarity_investigator.xgboost_model import XGBoostReviewModel


LANGUAGE_CODE = {"Python": "python", "C++": "cpp", "Java": "java"}
SUPPORTED_TYPES = ["py", "cpp", "cc", "cxx", "hpp", "java", "zip"]

st.set_page_config(
    page_title="Code Similarity Investigator",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --paper: #f4f0e6; --ink: #101010; --yellow: #f4d35e; --pink: #ef6f9f;
        --green: #b9df55; --blue: #83c5ff; --line: 3px; --shadow: 6px 6px 0 #101010;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1420px; padding-top: 2.2rem; padding-bottom: 4rem; }
    header[data-testid="stHeader"] { background: transparent; }
    h1, h2, h3, p, label { font-family: "IBM Plex Sans", "Arial Narrow", Arial, sans-serif; }
    h1 { font-size: clamp(2.8rem, 6vw, 6.5rem) !important; line-height: .86 !important; letter-spacing: -.06em !important; font-weight: 900 !important; text-transform: uppercase; margin-bottom: .5rem !important; }
    h2 { font-size: 1.5rem !important; text-transform: uppercase; letter-spacing: .04em; font-weight: 900 !important; }
    .eyebrow, .note, .mono { font-family: "IBM Plex Mono", "Courier New", monospace; text-transform: uppercase; }
    .eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .1em; border-left: var(--line) solid var(--ink); padding-left: .65rem; margin-bottom: .7rem; }
    .lede { max-width: 760px; font-size: 1.1rem; line-height: 1.45; margin-bottom: 1.6rem; }
    .neo-box { border: var(--line) solid var(--ink); box-shadow: var(--shadow); background: #fffdf7; padding: 1.25rem; margin-bottom: 1.4rem; }
    .stat { border: var(--line) solid var(--ink); box-shadow: 4px 4px 0 var(--ink); padding: .85rem 1rem; min-height: 110px; background: #fffdf7; }
    .stat-label { font-family: "IBM Plex Mono", "Courier New", monospace; font-size: .72rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .stat-value { font-size: 2.5rem; line-height: 1; font-weight: 900; margin-top: .35rem; }
    .feature-row { display: grid; grid-template-columns: 180px 1fr 70px; gap: .7rem; align-items: center; margin: .65rem 0; font-family: "IBM Plex Mono", "Courier New", monospace; font-size: .8rem; text-transform: uppercase; }
    .feature-track { height: 14px; border: 2px solid var(--ink); background: #fffdf7; } .feature-fill { height: 100%; background: var(--ink); }
    .evidence-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 1rem; }
    .evidence-card { border: 3px solid var(--ink); background: #fffdf7; padding: 14px; box-shadow: 4px 4px 0 var(--ink); }
    .evidence-card h3 { margin: 0 0 8px; font-family: "IBM Plex Mono", "Courier New", monospace; font-size: .8rem; text-transform: uppercase; }
    .evidence-number { font-size: 1.65rem; font-weight: 900; line-height: 1; }
    .evidence-meta { margin-top: 7px; font-family: "IBM Plex Mono", "Courier New", monospace; font-size: .7rem; text-transform: uppercase; line-height: 1.45; }
    .region-table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-family: "IBM Plex Mono", "Courier New", monospace; font-size: .75rem; }
    .region-table th, .region-table td { border: 2px solid var(--ink); padding: 7px 9px; text-align: left; }
    .region-table th { background: var(--yellow); text-transform: uppercase; }
    .note { font-size: .72rem; line-height: 1.5; }
    .status-line { font-family: "IBM Plex Mono", "Courier New", monospace; font-size: .8rem; text-transform: uppercase; margin: .3rem 0; }
    button[kind="primary"] { border: var(--line) solid var(--ink) !important; border-radius: 0 !important; box-shadow: 5px 5px 0 var(--ink) !important; background: var(--yellow) !important; color: var(--ink) !important; font-family: "IBM Plex Mono", "Courier New", monospace !important; font-weight: 900 !important; text-transform: uppercase; letter-spacing: .04em; }
    section[data-testid="stFileUploaderDropzone"] { border: var(--line) solid var(--ink); border-radius: 0; background: #fffdf7; }
    [data-testid="stFileUploaderDropzoneInstructions"] { font-family: "IBM Plex Mono", "Courier New", monospace; }
    div[data-testid="stDataFrame"] { border: var(--line) solid var(--ink); }
    [data-testid="stVerticalBlockBorderWrapper"] { border: var(--line) solid var(--ink); border-radius: 0; box-shadow: var(--shadow); background: #fffdf7; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: var(--line) solid var(--ink); }
    .stTabs [data-baseweb="tab"] { border: 2px solid var(--ink); border-bottom: 0; border-radius: 0; background: #fffdf7; font-family: "IBM Plex Mono", "Courier New", monospace; font-weight: 800; text-transform: uppercase; }
    .stTabs [aria-selected="true"] { background: var(--yellow); }
    </style>
    """, unsafe_allow_html=True,
)


def stat(label: str, value: str) -> None:
    st.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)


def feature_row(label: str, value: float | None) -> None:
    if value is None:
        st.markdown(f'<div class="feature-row"><div>{label}</div><div>NOT AVAILABLE</div><div>--</div></div>', unsafe_allow_html=True)
        return
    pct = max(0.0, min(1.0, value)) * 100
    st.markdown(f'<div class="feature-row"><div>{label}</div><div class="feature-track"><div class="feature-fill" style="width:{pct:.1f}%"></div></div><div>{value:.3f}</div></div>', unsafe_allow_html=True)


def pair_dataframe(results: list[PairResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append({
            "Student A": result.student_a,
            "Student B": result.student_b,
            "Language": result.evidence.get("language", "Unknown"),
            "Token": result.features["token_similarity"],
            "AST": result.features["ast_similarity"],
            "CFG": result.features["cfg_similarity"],
            "Semantic": result.features["semantic_similarity"],
            "Evidence score": result.model["score"],
        })
    return pd.DataFrame(rows).sort_values("Evidence score", ascending=False)


def render_similarity_space(frame: pd.DataFrame) -> None:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">03 / SIMILARITY SPACE</div>', unsafe_allow_html=True)
        st.markdown("## Every dot is a pair", unsafe_allow_html=True)
        st.markdown('<div class="note">Hover a point to inspect the student pair and detector outputs. Only same-language pairs are compared.</div>', unsafe_allow_html=True)
        if frame.empty:
            st.info("No same-language pairs are available for the current class.")
            return
        try:
            import plotly.express as px
            plot = frame.copy()
            plot["Pair"] = plot["Student A"] + " ↔ " + plot["Student B"]
            plot["Semantic"] = plot["Semantic"].fillna(0.0)
            fig = px.scatter(
                plot,
                x="AST",
                y="Token",
                hover_name="Pair",
                hover_data={"Token": ":.3f", "AST": ":.3f", "CFG": ":.3f", "Semantic": ":.3f", "Evidence score": ":.3f"},
                labels={"AST": "AST similarity", "Token": "Token similarity"},
                title="AST vs Token",
            )
            fig.update_traces(marker={"size": 11})
            fig.update_xaxes(showline=True, linewidth=2, linecolor="#101010", tickfont={"color": "#101010"}, title_font={"color": "#101010"}, gridcolor="#b7b1a6", zerolinecolor="#101010")
            fig.update_yaxes(showline=True, linewidth=2, linecolor="#101010", tickfont={"color": "#101010"}, title_font={"color": "#101010"}, gridcolor="#b7b1a6", zerolinecolor="#101010")
            fig.update_layout(height=430, margin={"l": 70, "r": 25, "t": 60, "b": 65}, paper_bgcolor="#fffdf7", plot_bgcolor="#fffdf7", font={"color": "#101010"})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except ImportError:
            st.info("Plotly is not installed. Install the project requirements to enable the interactive similarity space.")


def render_relationships(results: list[PairResult]) -> None:
    frame = pair_dataframe(results)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">01 / RELATIONSHIPS</div>', unsafe_allow_html=True)
        st.markdown("## Relationship map", unsafe_allow_html=True)
        st.markdown('<div class="note">Ordered by the available deterministic evidence score. This is a review aid, not a plagiarism verdict.</div>', unsafe_allow_html=True)
        if frame.empty:
            st.info("No same-language pairs were generated. Add at least two submissions in the same language.")
            return
        st.dataframe(frame, hide_index=True, use_container_width=True, column_config={
            "Token": st.column_config.NumberColumn(format="%.3f"),
            "AST": st.column_config.NumberColumn(format="%.3f"),
            "CFG": st.column_config.NumberColumn(format="%.3f"),
            "Semantic": st.column_config.NumberColumn(format="%.3f"),
            "Evidence score": st.column_config.NumberColumn(format="%.3f"),
        })

    with st.container(border=True):
        st.markdown('<div class="eyebrow">02 / INVESTIGATE</div>', unsafe_allow_html=True)
        labels = [f"{item.student_a}  ↔  {item.student_b}  /  {float(item.model['score']):.3f}" for item in results]
        selected = st.selectbox("Select relationship", labels, label_visibility="collapsed")
        render_pair_investigation(results[labels.index(selected)])


def render_pair_investigation(result: PairResult) -> None:
    st.markdown(f"<h2>{result.student_a}  ↔  {result.student_b}</h2>", unsafe_allow_html=True)
    st.markdown(f'<div class="note">Language: {result.evidence.get("language", "Unknown")}. {result.evidence.get("score_basis", "")}. The system exposes measured detector outputs. The reviewer decides what those measurements mean.</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        feature_row("Token similarity", float(result.features["token_similarity"]))
        feature_row("AST similarity", result.features["ast_similarity"])
        feature_row("CFG similarity", result.features["cfg_similarity"])
        feature_row("Semantic similarity", result.features["semantic_similarity"])
    with right:
        st.markdown(f'<div class="stat"><div class="stat-label">Available evidence score</div><div class="stat-value">{float(result.model["score"]):.3f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="note">matched tokens: {int(result.features["matched_token_count"])}<br>matched regions: {int(result.features["matched_region_count"])}<br>shared AST blocks: {int(result.features["shared_subtree_count"])}<br>file size ratio: {float(result.features["file_size_ratio"]):.3f}<br>function count difference: {int(result.features["function_count_difference"])}<br>class count difference: {int(result.features["class_count_difference"])}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["SOURCE", "AST", "CFG", "EVIDENCE"])
    submissions = st.session_state["submissions"]
    source_a_item = next(item for item in submissions if item.student_id == result.student_a)
    source_b_item = next(item for item in submissions if item.student_id == result.student_b)
    source_a, source_b = source_a_item.source, source_b_item.source
    language = source_a_item.language
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            st.markdown(f"**{result.student_a}**")
            st.code(source_a, language=LANGUAGE_CODE[language])
        with b:
            st.markdown(f"**{result.student_b}**")
            st.code(source_b, language=LANGUAGE_CODE[language])
    with tabs[1]:
        st.graphviz_chart(ast_to_dot(source_a, language=language_key(language)), use_container_width=True)
        st.graphviz_chart(ast_to_dot(source_b, language=language_key(language)), use_container_width=True)
    with tabs[2]:
        st.graphviz_chart(cfg_to_dot(source_a, language=language_key(language)), use_container_width=True)
        st.graphviz_chart(cfg_to_dot(source_b, language=language_key(language)), use_container_width=True)
    with tabs[3]:
        st.markdown('<div class="evidence-grid">', unsafe_allow_html=True)
        token = float(result.features["token_similarity"])
        ast = result.features["ast_similarity"]
        cfg = result.features["cfg_similarity"]
        semantic = result.features["semantic_similarity"]
        cards = [
            ("TOKEN", f"{token:.3f}", f"{int(result.features['matched_token_count'])} matched tokens / {int(result.features['matched_region_count'])} regions"),
            ("AST", "NOT AVAILABLE" if ast is None else f"{float(ast):.3f}", "Canonical structural syntax detector" if ast is not None else "Not available"),
            ("CFG", "NOT AVAILABLE" if cfg is None else f"{float(cfg):.3f}", "Lightweight control-flow detector" if cfg is not None else "Not available"),
            ("SEMANTIC", "NOT AVAILABLE" if semantic is None else f"{float(semantic):.3f}", "Local CodeBERT mean-pooled cosine similarity" if semantic is not None else "Enable local CodeBERT weights to enable this detector"),
        ]
        for title, number, meta in cards:
            st.markdown(f'<div class="evidence-card"><h3>{title}</h3><div class="evidence-number">{number}</div><div class="evidence-meta">{meta}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### Matched token regions")
        regions = result.evidence["matched_token_regions"][:20]
        if regions:
            rows = [f"<tr><td>{i:02d}</td><td>{r['left_start']} to {r['left_start'] + r['size'] - 1}</td><td>{r['right_start']} to {r['right_start'] + r['size'] - 1}</td><td>{r['size']}</td></tr>" for i, r in enumerate(regions, 1)]
            st.markdown('<table class="region-table"><thead><tr><th>Region</th><th>Student A tokens</th><th>Student B tokens</th><th>Length</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="note">No matching normalized token regions.</div>', unsafe_allow_html=True)
        st.markdown("### Detector notes")
        for item in result.evidence["explanation"]:
            st.markdown(f"- {item}")


def render_results() -> None:
    submissions: list[Submission] = st.session_state["submissions"]
    results: list[PairResult] = st.session_state["results"]
    frame = pair_dataframe(results)
    st.markdown('<div class="eyebrow">ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
    st.markdown("# FIND THE RELATIONSHIPS", unsafe_allow_html=True)
    st.markdown('<div class="lede">Start with the relationships, then inspect the evidence. Static detectors, local CodeBERT semantic similarity, and the local XGBoost review-priority model remain visible as measured evidence.</div>', unsafe_allow_html=True)
    same_language_pairs = len(results)
    columns = st.columns(4)
    for col, label, value in zip(columns, ["Submissions", "Same-language pairs", "Languages", "Mean evidence score"], [str(len(submissions)), f"{same_language_pairs:,}", str(len({item.language for item in submissions})), f"{float(frame['Evidence score'].mean()):.3f}" if not frame.empty else "--"]):
        with col:
            stat(label, value)
    render_relationships(results)
    render_similarity_space(frame)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">04 / RELATIONSHIP GROUPS</div>', unsafe_allow_html=True)
        groups = relationship_groups(results, threshold=0.70)
        if groups:
            st.dataframe(pd.DataFrame({"Group": [i + 1 for i in range(len(groups))], "Members": [", ".join(group) for group in groups]}), hide_index=True, use_container_width=True)
        else:
            st.markdown('<div class="note">No multi-submission relationship groups crossed the configured evidence threshold of 0.70.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="eyebrow">05 / FEATURE DATASET</div>', unsafe_allow_html=True)
        st.markdown("## ML FEATURE DATASET", unsafe_allow_html=True)
        st.markdown('<div class="note">The CSV contains the exact numeric feature contract used by the local XGBoost review-priority model. The model consumes measured detector outputs and never generates a plagiarism verdict.</div>', unsafe_allow_html=True)
        feature_path = st.session_state.get("feature_path")
        if feature_path:
            st.code(feature_path, language="text")
            st.download_button("DOWNLOAD PAIR FEATURE CSV", data=Path(feature_path).read_bytes(), file_name=Path(feature_path).name, mime="text/csv")


def get_ml_model() -> XGBoostReviewModel | None:
    if "xgb_model" in st.session_state:
        return st.session_state["xgb_model"]
    model = XGBoostReviewModel(model_path=ROOT / "models" / "xgboost" / "review_priority.json")
    try:
        model.load()
    except Exception:
        st.session_state["xgb_model"] = None
        return None
    st.session_state["xgb_model"] = model
    return model


def get_semantic_encoder() -> CodeBERTEncoder | None:
    if "semantic_encoder" in st.session_state:
        return st.session_state["semantic_encoder"]
    encoder = CodeBERTEncoder(cache_dir=ROOT / "models" / "huggingface", embedding_cache_dir=ROOT / "data" / "cache" / "embeddings", local_files_only=True)
    try:
        encoder._load()
    except Exception:
        st.session_state["semantic_encoder"] = None
        return None
    st.session_state["semantic_encoder"] = encoder
    return encoder


st.markdown('<div class="eyebrow">LOCAL / EVIDENCE-FIRST / MULTI-LANGUAGE</div>', unsafe_allow_html=True)
st.markdown("# CODE<br>SIMILARITY<br>INVESTIGATOR", unsafe_allow_html=True)
st.markdown('<div class="lede">Upload a class as source files or one ZIP. Process locally. Inspect measured relationships and structural evidence. Student code is never executed.</div>', unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state["results"] = []
if "submissions" not in st.session_state:
    st.session_state["submissions"] = []

upload_left, action_right = st.columns([3.2, 1], gap="large")
with upload_left:
    st.markdown('<div class="eyebrow">INPUT</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Source files or ZIP archive",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        max_upload_size=100,
        label_visibility="collapsed",
        help="Supported: Python, C++, Java, plus ZIP archives. ZIP contents are validated before analysis.",
    )
    st.markdown('<div class="note">Accepted: .py .cpp .cc .cxx .hpp .java .zip / maximum individual upload: 100 MB / source remains local</div>', unsafe_allow_html=True)
with action_right:
    st.markdown('<div class="eyebrow">OPTIONS</div>', unsafe_allow_html=True)
    use_semantic = st.checkbox("ENABLE LOCAL CODEBERT", value=True, help="Only works when microsoft/codebert-base is already available in the local model cache. Loading can take time the first time.")
    st.markdown('<div class="eyebrow">ACTION</div>', unsafe_allow_html=True)
    process = st.button("PROCESS SUBMISSIONS", type="primary", use_container_width=True, disabled=not uploaded)

if process and uploaded:
    started = time.perf_counter()
    status = st.status("Starting local analysis...", expanded=True)
    progress = st.progress(0, text="Preparing")
    try:
        inputs = [InputFile(item.name, item.getvalue()) for item in uploaded]
        status.write("Validating uploads and extracting supported source files...")
        progress.progress(10, text="Validating input")
        submissions = ingest(inputs)
        status.write(f"Loaded {len(submissions)} submissions across {len({item.language for item in submissions})} language(s).")
        progress.progress(25, text="Input ready")

        encoder = None
        if use_semantic:
            status.write("Loading local CodeBERT weights...")
            progress.progress(30, text="Loading local semantic model")
            encoder = get_semantic_encoder()
            if encoder is None:
                status.write("Local CodeBERT is not available. Continuing without semantic similarity.")
            else:
                status.write("Local CodeBERT loaded. Embeddings will be cached.")
        else:
            status.write("Semantic detector disabled for this run. Static detectors will be used.")

        def on_progress(stage: str, current: int, total: int) -> None:
            if total <= 0:
                return
            fraction = current / total
            if stage == "features":
                value = 0.30 + 0.20 * fraction
                label = f"Parsing {current}/{total} submissions"
            elif stage == "semantic":
                value = 0.50 + 0.15 * fraction
                label = f"Embedding {current}/{total} submissions"
            else:
                value = 0.65 + 0.30 * fraction
                label = f"Comparing {current}/{total} same-language pairs"
            progress.progress(int(value * 100), text=label)

        status.write("Computing normalized tokens and structural features...")
        results = compare_all(submissions, semantic_encoder=encoder, progress_callback=on_progress)
        ml_model = get_ml_model()
        if ml_model is not None:
            status.write("Applying local XGBoost review-priority model...")
            ml_model.score_results(results)
        else:
            status.write("XGBoost model artifact not found. Keeping deterministic evidence score.")
        progress.progress(96, text="Saving feature dataset")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_path = ROOT / "data" / "runs" / run_id / "pair_features.csv"
        save_pair_features(results, feature_path)
        progress.progress(100, text="Complete")
        status.update(label=f"Analysis complete in {time.perf_counter() - started:.2f}s", state="complete", expanded=False)
        st.session_state["submissions"] = submissions
        st.session_state["results"] = results
        st.session_state["feature_path"] = str(feature_path)
        st.session_state["elapsed"] = time.perf_counter() - started
        st.rerun()
    except IngestionError as exc:
        status.update(label="Input validation failed", state="error")
        st.error(str(exc))
    except SyntaxError as exc:
        status.update(label="Source parsing failed", state="error")
        st.error(f"Python parsing failed: {exc}")
    except Exception as exc:
        status.update(label="Analysis failed", state="error")
        st.exception(exc)

if st.session_state["results"]:
    render_results()
else:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">WAITING FOR A CLASS</div>', unsafe_allow_html=True)
        st.markdown("## Upload. Process. Investigate.", unsafe_allow_html=True)
        st.markdown('<div class="note">Python, C++, and Java use token, AST, lightweight CFG, local CodeBERT, and an optional local XGBoost review-priority model. Student code is never executed.</div>', unsafe_allow_html=True)
