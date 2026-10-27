
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# ============================================================
# LOGIRISK AI - DELIVERY RISK INTELLIGENCE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "delivery_risk_xgb.json"
PREPROCESSOR_PATH = BASE_DIR / "delivery_preprocessor.pkl"
CONFIG_PATH = BASE_DIR / "feature_config.json"

PAGES = [
    "Overview",
    "Risk Studio",
    "Batch Analytics",
    "Live Simulator",
    "Model Lab",
]

FEATURE_COLS = [
    "Purchase_Year",
    "Purchase_Month",
    "Purchase_DayOfWeek",
    "Purchase_Hour",
    "Purchase_Weekend",
    "customer_state",
    "Seller_State_Count",
    "Seller_Cities_Count",
    "Item_Count",
]

NUMERIC_COLS = [
    "Purchase_Year",
    "Purchase_Month",
    "Purchase_DayOfWeek",
    "Purchase_Hour",
    "Purchase_Weekend",
    "Seller_State_Count",
    "Seller_Cities_Count",
    "Item_Count",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LogiRisk AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #080d17;
    color: #edf2fc;
}

[data-testid="stSidebar"] {
    background: #0c1321;
    border-right: 1px solid #24334a;
}

.hero {
    background: linear-gradient(120deg,#101e32,#10303b);
    border: 1px solid #28465a;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 24px;
}

.hero-tag {
    color: #7ce9df;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 800;
}

.hero h1 {
    font-size: clamp(28px,4vw,42px);
    font-weight: 800;
    margin: 10px 0;
    color: #f5f8ff;
}

.hero p {
    color: #a7bad0;
    font-size: 15px;
}

.panel {
    background: #101827;
    border: 1px solid #24354d;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 12px;
}

.panel h3 {
    color: #edf4ff;
}

.muted {
    color: #9aacc4;
    font-size: 13px;
    line-height: 1.7;
}

.accent {
    color: #7ce9df;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: #101827;
    border: 1px solid #24354d;
    border-radius: 14px;
    padding: 17px;
}

div[data-testid="stMetricValue"] {
    color: #f2f7ff;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: #70e4da;
    color: #07141b;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 44px;
}

.stButton > button {
    border-radius: 9px;
    min-height: 40px;
}

div[data-testid="stForm"] {
    background: #101827;
    border: 1px solid #24354d;
    border-radius: 14px;
    padding: 22px;
}

.footer {
    color: #71839a;
    text-align: center;
    font-size: 12px;
    padding: 25px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

# Navigation state is controlled ONLY by the sidebar radio.
# No callbacks or buttons modify the widget-bound state.
if "page_selector" not in st.session_state:
    st.session_state["page_selector"] = "Overview"

if st.session_state["page_selector"] not in PAGES:
    st.session_state["page_selector"] = "Overview"

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None



# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    for file_path in [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        CONFIG_PATH,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path.name}"
            )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    expected = config.get("feature_cols")

    if expected is not None and expected != FEATURE_COLS:
        raise ValueError(
            "Feature columns mismatch. Check feature_config.json."
        )

    threshold = float(config.get("threshold", 0.50))

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold.")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    classifier = xgb.XGBClassifier()
    classifier.load_model(str(MODEL_PATH))

    try:
        encoder = (
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
        )

        states = sorted(
            str(state) for state in encoder.categories_[0]
        )

    except (KeyError, AttributeError, IndexError):
        raise ValueError(
            "Could not read state categories from preprocessor. "
            "Check your exported preprocessing pipeline."
        )

    if not states:
        raise ValueError("No customer state categories found.")

    return classifier, preprocessor, config, states


try:
    model, preprocessor, config, STATE_OPTIONS = load_model()
    MODEL_READY = True
    MODEL_ERROR = None

except Exception as exc:
    MODEL_READY = False
    MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    model = None
    preprocessor = None
    config = {}
    STATE_OPTIONS = []

THRESHOLD = float(config.get("threshold", 0.50))


# ============================================================
# HELPERS
# ============================================================

def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-tag">
                LOGIRISK AI / DELIVERY INTELLIGENCE
            </div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_features(df):
    missing = [
        col for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing:
        return f"Missing columns: {missing}"

    X = df[FEATURE_COLS].copy()

    if X.empty:
        return "No records found."

    if X.isna().any().any():
        return "Missing values found. Please clean the CSV."

    try:
        for col in NUMERIC_COLS:
            X[col] = pd.to_numeric(
                X[col], errors="raise"
            )
    except (ValueError, TypeError):
        return "Some numeric columns contain invalid values."

    values = X[NUMERIC_COLS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        return "Invalid infinite or non-finite numeric values."

    # Validate integer-valued features.
    for col in NUMERIC_COLS:
        arr = X[col].to_numpy(dtype=float)

        if not np.equal(arr, np.floor(arr)).all():
            return f"{col} must contain whole numbers."

    ranges = {
        "Purchase_Year": (2016, 2035),
        "Purchase_Month": (1, 12),
        "Purchase_DayOfWeek": (0, 6),
        "Purchase_Hour": (0, 23),
        "Seller_State_Count": (1, 100),
        "Seller_Cities_Count": (1, 1000),
        "Item_Count": (1, 100),
    }

    for col, (low, high) in ranges.items():
        if not X[col].between(low, high).all():
            return (
                f"{col} must be between {low} and {high}."
            )

    if not X["Purchase_Weekend"].isin([0, 1]).all():
        return "Purchase_Weekend must be 0 or 1."

    X["customer_state"] = X["customer_state"].astype(str)

    if not X["customer_state"].isin(STATE_OPTIONS).all():
        return (
            "Unknown customer_state. Use a supported "
            "state category from the training encoder."
        )

    return X


def predict_risk(X):
    transformed = preprocessor.transform(X[FEATURE_COLS])

    scores = model.predict_proba(transformed)[:, 1]

    scores = np.asarray(scores, dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Model returned invalid scores.")

    return np.clip(scores, 0, 1)


def make_input(
    year,
    month,
    day,
    hour,
    weekend,
    state,
    seller_states,
    seller_cities,
    item_count,
):
    return pd.DataFrame(
        [{
            "Purchase_Year": year,
            "Purchase_Month": month,
            "Purchase_DayOfWeek": day,
            "Purchase_Hour": hour,
            "Purchase_Weekend": weekend,
            "customer_state": state,
            "Seller_State_Count": seller_states,
            "Seller_Cities_Count": seller_cities,
            "Item_Count": item_count,
        }],
        columns=FEATURE_COLS,
    )


def show_result(score):
    flagged = score >= THRESHOLD

    col1, col2 = st.columns(2)

    col1.metric("Model score", f"{score:.1%}")

    col2.metric(
        "Demo classification",
        "Flagged" if flagged else "Not flagged",
    )

    st.progress(float(score))

    if flagged:
        st.warning(
            "This order is flagged by the model threshold."
        )
    else:
        st.success(
            "This order is not flagged by the model threshold."
        )

    st.caption(
        "This score is not a verified or calibrated "
        "real-world probability. It does not guarantee "
        "a delivery outcome."
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 📦 LogiRisk AI")
    st.caption("Delivery Risk Intelligence")
    st.divider()

    st.markdown("### WORKSPACE")

    selected_page = st.radio(
        "Navigate",
        PAGES,
        key="page_selector",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### MODEL STATUS")

    if MODEL_READY:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")

    st.caption("Algorithm: XGBoost")
    st.caption(f"Threshold: {THRESHOLD:.2f}")

    st.divider()
    st.caption("ACADEMIC RESEARCH PROTOTYPE")
    st.caption("Historical Olist dataset")


# ============================================================
# MODEL ERROR SCREEN
# ============================================================

if not MODEL_READY:
    page_header(
        "LogiRisk AI",
        "Delivery Risk Intelligence",
    )

    st.error("Unable to initialize the model.")

    st.code(MODEL_ERROR)

    st.info(
        "Verify the following files are present "
        "in the same directory as app.py:\n\n"
        "delivery_risk_xgb.json\n"
        "delivery_preprocessor.pkl\n"
        "feature_config.json"
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if selected_page == "Overview":

    page_header(
        "Delivery Risk Intelligence",
        "Understand order risk. Explore predictions. "
        "Analyze delivery patterns.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="accent">PROJECT OVERVIEW</div>
            <h3>Predictive analytics for e-commerce logistics</h3>
            <p class="muted">
                LogiRisk AI is an experimental ML application
                estimating late-delivery risk from historical
                order features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Start analyzing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="panel">
                <h3>Risk Studio</h3>
                <p class="muted">
                    Analyze an individual order using
                    the trained model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info("To continue, select **Risk Studio** from the sidebar.")

    with col2:
        st.markdown(
            """
            <div class="panel">
                <h3>Batch Analytics</h3>
                <p class="muted">
                    Score multiple orders and export
                    the prediction report.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info("To continue, select **Batch Analytics** from the sidebar.")

    st.divider()

    st.subheader("Model snapshot")

    a, b, c = st.columns(3)

    a.metric("Algorithm", "XGBoost")
    b.metric("Input features", len(FEATURE_COLS))
    c.metric("Demo threshold", f"{THRESHOLD:.0%}")

    st.subheader("Recorded experiment")

    a, b, c = st.columns(3)

    a.metric("ROC-AUC", "0.6187")
    b.metric("Average Precision", "0.1543")
    c.metric("Historical late rate", "8.11%")

    st.caption(
        "Recorded experiment metrics. These values "
        "are not recomputed by this application."
    )

    st.markdown(
        """
        <div class="panel">
            <h3>Prediction workflow</h3>
            <p class="muted">
                Input features → Preprocessing →
                XGBoost → Model score →
                Threshold-based classification
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RISK STUDIO
# ============================================================

elif selected_page == "Risk Studio":

    page_header(
        "Risk Studio",
        "Explore the model score for a single order.",
    )

    st.info(
        "Enter the order features expected by the "
        "trained model."
    )

    with st.form("single_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.number_input(
                "Purchase year",
                min_value=2016,
                max_value=2035,
                value=2018,
            )

            month = st.selectbox(
                "Purchase month",
                list(range(1, 13)),
            )

            day = st.selectbox(
                "Day of week (Monday = 0)",
                list(range(7)),
            )

        with col2:
            hour = st.slider(
                "Purchase hour",
                0,
                23,
                12,
            )

            weekend = st.selectbox(
                "Weekend?",
                [0, 1],
                format_func=lambda v: (
                    "Yes" if v == 1 else "No"
                ),
            )

            state = st.selectbox(
                "Customer state",
                STATE_OPTIONS,
            )

        with col3:
            seller_states = st.number_input(
                "Seller state count",
                min_value=1,
                max_value=100,
                value=1,
            )

            seller_cities = st.number_input(
                "Seller cities count",
                min_value=1,
                max_value=1000,
                value=1,
            )

            item_count = st.number_input(
                "Item count",
                min_value=1,
                max_value=100,
                value=1,
            )

        submitted = st.form_submit_button(
            "Analyze delivery risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = make_input(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(weekend),
            state,
            int(seller_states),
            int(seller_cities),
            int(item_count),
        )

        validated = validate_features(input_df)

        if isinstance(validated, str):
            st.error(validated)

        else:
            try:
                score = float(predict_risk(validated)[0])

                st.session_state.single_result = {
                    "score": score,
                    "input": input_df.copy(),
                }

            except Exception as exc:
                st.error(
                    f"Prediction failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    result = st.session_state.single_result

    if result is not None:
        st.divider()
        st.subheader("Prediction result")

        show_result(result["score"])

        st.subheader("Input features")

        st.dataframe(
            result["input"],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# BATCH ANALYTICS
# ============================================================

elif selected_page == "Batch Analytics":

    page_header(
        "Batch Analytics",
        "Analyze multiple orders and export prediction results.",
    )

    st.markdown("#### Required CSV columns")

    st.code(",".join(FEATURE_COLS))

    template = pd.DataFrame(columns=FEATURE_COLS)

    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="delivery_risk_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload order CSV",
        type=["csv"],
        key="batch_csv",
    )

    if uploaded is not None:

        try:
            batch_df = pd.read_csv(uploaded)

            st.caption(
                f"Records detected: {len(batch_df):,}"
            )

            if batch_df.empty:
                st.error("Uploaded CSV is empty.")

            elif len(batch_df) > 10000:
                st.error(
                    "Demo limit: upload up to 10,000 records."
                )

            else:
                validated = validate_features(batch_df)

                if isinstance(validated, str):
                    st.error(validated)

                else:
                    st.subheader("Preview")

                    st.dataframe(
                        validated.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button(
                        "Generate predictions",
                        type="primary",
                        key="generate_batch",
                    ):
                        try:
                            scores = predict_risk(validated)

                            results = batch_df.copy()

                            results["Risk_Score"] = scores

                            results["Risk_Flag"] = (
                                scores >= THRESHOLD
                            ).astype(int)

                            results["Risk_Label"] = np.where(
                                scores >= THRESHOLD,
                                "Flagged",
                                "Not flagged",
                            )

                            st.session_state.batch_results = (
                                results
                            )

                        except Exception as exc:
                            st.error(
                                f"Batch prediction failed: {exc}"
                            )

        except Exception as exc:
            st.error(
                f"CSV processing failed: "
                f"{type(exc).__name__}: {exc}"
            )

    results = st.session_state.batch_results

    if results is not None:

        st.divider()
        st.subheader("Batch report")

        total = len(results)

        flagged = int(results["Risk_Flag"].sum())
        not_flagged = total - flagged

        a, b, c, d = st.columns(4)

        a.metric("Orders analyzed", f"{total:,}")
        b.metric("Flagged", f"{flagged:,}")
        c.metric("Not flagged", f"{not_flagged:,}")

        d.metric(
            "Mean model score",
            f"{results['Risk_Score'].mean():.1%}",
        )

        st.caption(
            "Mean model score is not an observed "
            "late-delivery rate."
        )

        st.subheader("Classification distribution")

        distribution = pd.DataFrame({
            "Classification": [
                "Flagged",
                "Not flagged",
            ],
            "Orders": [
                flagged,
                not_flagged,
            ],
        }).set_index("Classification")

        st.bar_chart(distribution)

        st.subheader("Risk score distribution")

        bins = pd.cut(
            results["Risk_Score"],
            bins=np.linspace(0, 1, 11),
            include_lowest=True,
        )

        histogram = bins.value_counts().sort_index()

        histogram.index = histogram.index.astype(str)

        st.bar_chart(histogram)

        st.subheader("Prediction records")

        st.dataframe(
            results.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download prediction report",
            results.to_csv(index=False).encode("utf-8"),
            file_name="logirisk_predictions.csv",
            mime="text/csv",
            type="primary",
        )

        if st.button(
            "Clear batch results",
            key="clear_batch",
        ):
            st.session_state.batch_results = None
            st.rerun()


# ============================================================
# LIVE SIMULATOR — HISTORICAL-STYLE DEMO STREAM
# ============================================================

elif selected_page == "Live Simulator":

    import time
    from datetime import datetime, timedelta

    page_header(
        "Live Simulator",
        "A simulated order stream scored by the loaded model. "
        "This is a demo feed, not a live commerce integration.",
    )

    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "sim_orders" not in st.session_state:
        st.session_state.sim_orders = []
    if "sim_counter" not in st.session_state:
        st.session_state.sim_counter = 0

    left, mid, right = st.columns([1, 1, 2])
    with left:
        if st.button("▶ Start feed", type="primary", disabled=st.session_state.sim_running):
            st.session_state.sim_running = True
            st.rerun()
    with mid:
        if st.button("⏸ Pause", disabled=not st.session_state.sim_running):
            st.session_state.sim_running = False
            st.rerun()
    with right:
        if st.button("↻ Reset demo"):
            st.session_state.sim_running = False
            st.session_state.sim_orders = []
            st.session_state.sim_counter = 0
            st.rerun()

    if st.session_state.sim_running:
        st.success("Simulator running • new demo orders are scored about every 4 seconds.")
    else:
        st.info("Simulator paused. Press Start feed to begin.")

    st.caption(
        "Demo mode: generated example orders are sent through the existing model. "
        "Scores are model outputs, not calibrated real-world probabilities."
    )

    def simulator_tick():
        # Keep one short burst per refresh; no external order system is connected.
        import random
        from datetime import datetime
        state = random.choice(STATE_OPTIONS)
        now = datetime.now()
        row = make_input(
            now.year, now.month, now.weekday(), now.hour,
            int(now.weekday() >= 5), state,
            random.randint(1, 8), random.randint(1, 30),
            random.randint(1, 4),
        )
        score = float(predict_risk(row)[0])
        st.session_state.sim_counter += 1
        order = {
            "Demo_Order_ID": f"DEMO-{st.session_state.sim_counter:05d}",
            "Received_At": now.strftime("%H:%M:%S"),
            "Customer_State": state,
            "Items": int(row.iloc[0]["Item_Count"]),
            "Risk_Score": score,
            "Risk_Flag": "Flagged" if score >= THRESHOLD else "Not flagged",
        }
        st.session_state.sim_orders.insert(0, order)
        st.session_state.sim_orders = st.session_state.sim_orders[:100]

    if st.session_state.sim_running:
        @st.fragment(run_every="4s")
        def live_feed():
            if st.session_state.sim_running:
                try:
                    simulator_tick()
                except Exception as exc:
                    st.error(f"Demo order scoring failed: {exc}")
                    return
            orders = st.session_state.sim_orders
            flagged = sum(o["Risk_Flag"] == "Flagged" for o in orders)
            total = len(orders)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Orders processed", total, help="Demo orders scored in this session")
            k2.metric("Needs attention", flagged, delta="Review" if flagged else "All clear", delta_color="inverse")
            k3.metric("Not flagged", total - flagged)
            k4.metric("Flag rate", f"{flagged / total:.1%}" if total else "—")
            alert_orders = [o for o in orders if o["Risk_Flag"] == "Flagged"]
            if alert_orders:
                st.error(f"⚠️ {len(alert_orders)} demo order(s) crossed the configured {THRESHOLD:.0%} flag threshold.")
                with st.expander("Review flagged demo orders", expanded=True):
                    for item in alert_orders[:8]:
                        st.write(f"**{item['Demo_Order_ID']}** · {item['Customer_State']} · Risk score: **{item['Risk_Score']:.2%}**")
            else:
                st.success("✓ No demo orders currently exceed the configured flag threshold.")
            st.subheader("Recent demo orders")
            if orders:
                view = pd.DataFrame(orders)
                view["Risk_Score"] = view["Risk_Score"].map(lambda v: f"{v:.2%}")
                st.dataframe(view, use_container_width=True, hide_index=True)
                csv_data = pd.DataFrame(orders).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download simulator report",
                    csv_data,
                    file_name="logirisk_simulator_report.csv",
                    mime="text/csv",
                    key="sim_download",
                )
            else:
                st.info("Waiting for the first demo order.")
        live_feed()
    else:
        orders = st.session_state.sim_orders
        flagged = sum(o["Risk_Flag"] == "Flagged" for o in orders)
        total = len(orders)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Orders processed", total, help="Demo orders scored in this session")
        k2.metric("Needs attention", flagged, delta="Review" if flagged else "All clear", delta_color="inverse")
        k3.metric("Not flagged", total - flagged)
        k4.metric("Flag rate", f"{flagged / total:.1%}" if total else "—")
        alert_orders = [o for o in orders if o["Risk_Flag"] == "Flagged"]
        if alert_orders:
            st.error(f"⚠️ {len(alert_orders)} demo order(s) crossed the configured {THRESHOLD:.0%} flag threshold.")
            with st.expander("Review flagged demo orders", expanded=True):
                for item in alert_orders[:8]:
                    st.write(f"**{item['Demo_Order_ID']}** · {item['Customer_State']} · Risk score: **{item['Risk_Score']:.2%}**")
        elif orders:
            st.success("✓ No demo orders currently exceed the configured flag threshold.")
        st.subheader("Recent demo orders")
        if orders:
            view = pd.DataFrame(orders)
            view["Risk_Score"] = view["Risk_Score"].map(lambda v: f"{v:.2%}")
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button(
                "Download simulator report",
                pd.DataFrame(orders).to_csv(index=False).encode("utf-8"),
                file_name="logirisk_simulator_report.csv",
                mime="text/csv",
                key="sim_download_paused",
            )
        else:
            st.info("No simulated orders yet. Start the feed to generate the first order.")


# ============================================================
# MODEL LAB
# ============================================================

elif selected_page == "Model Lab":

    page_header(
        "Model Lab",
        "Methodology, experimental metrics and limitations.",
    )

    st.subheader("Model architecture")

    st.markdown(
        """
        <div class="panel">
            <p class="accent">01 / INPUT FEATURES</p>
            <p class="muted">
                Purchase date/time, customer state,
                seller counts and item count.
            </p>
            <hr>
            <p class="accent">02 / PREPROCESSING</p>
            <p class="muted">
                Fitted numerical and categorical
                preprocessing pipeline.
            </p>
            <hr>
            <p class="accent">03 / CLASSIFIER</p>
            <p class="muted">
                XGBoost binary classification.
            </p>
            <hr>
            <p class="accent">04 / OUTPUT</p>
            <p class="muted">
                Model score and demo classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    a.metric("ROC-AUC", "0.6187")
    b.metric("Average Precision", "0.1543")
    c.metric("Historical late rate", "8.11%")

    st.caption(
        "These are previously recorded experimental metrics."
    )

    st.subheader("Model configuration")

    st.write("Algorithm: XGBoost")
    st.write("Number of features:", len(FEATURE_COLS))
    st.write("Classification threshold:", THRESHOLD)

    with st.expander("Feature list"):
        st.code("\n".join(FEATURE_COLS))

    with st.expander("Methodology"):
        st.markdown("""
        1. Historical Olist order dataset.
        2. Target: late delivery versus estimated delivery.
        3. Chronological data split.
        4. Fitted preprocessing pipeline.
        5. XGBoost classification.
        6. Threshold-based classification.
        """)

    st.subheader("Limitations")

    st.warning("""
    - Experimental model with limited predictive discrimination.
    - Historical data may not represent future operations.
    - Scores are not verified calibrated probabilities.
    - Classification does not guarantee delivery outcomes.
    - Seller features may not be available at order time.
    - Temporal validity and leakage need review.
    - Operational deployment needs further validation.
    """)

    st.subheader("Future improvements")

    st.markdown("""
    - Temporal validation
    - Leakage review
    - Precision and recall evaluation
    - PR curve analysis
    - Business-cost-based threshold selection
    - Probability calibration evaluation
    - SHAP explanation integration
    - Drift monitoring
    - Testing on newer unseen data
    """)

    st.info(
        "Future improvements are proposed work, "
        "not currently implemented functionality."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        LOGIRISK AI · Academic Research Prototype · Olist Dataset
        <br>
        Predictions should not be used as delivery commitments.
    </div>
    """,
    unsafe_allow_html=True,
)
