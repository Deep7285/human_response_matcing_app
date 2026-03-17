import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import pdfplumber

# Page Setup 
st.set_page_config(
    page_title="Mentor–Coachee Pair Matching",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5a7a9c;
        margin-bottom: 1.5rem;
    }
    .upload-label {
        font-size: 1rem;
        font-weight: 600;
        color: #1a3a5c;
        margin-bottom: 4px;
    }
    .results-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a3a5c;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }
    .sidebar-note {
        font-size: 0.78rem;
        color: #64748b;
        font-style: italic;
        margin-top: -8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Page Header 
st.markdown('<div class="main-header">🤝 Mentor–Coachee Pair Matching</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload the mentor and coachee response datasheets to generate top 3 pairings.</div>', unsafe_allow_html=True)
st.info(" The matching algorithm gives top 3 mentor options for each coachee based on a weighted combination of criteria including " \
"specialisation, degree level, professional goals, personal interests, IIT experience, and background. Adjust the weights in the sidebar to prioritise certain criteria.\n\n"
"📋 **Note:** Coachee and mentor data must be prepared in **separate** files (CSV, Excel, or PDF).")

# Sidebar: Match critatia Weight Adjustmentment
st.sidebar.markdown("## ⚙️ Matching Criteria Weights")
st.sidebar.markdown(
    '<p class="sidebar-note">Click the ❓ icon next to each slider to learn what that criterion measures.</p>',
    unsafe_allow_html=True,
)

# Hard Skills (Degree & Specialisation)

w_spec = st.sidebar.slider(
    "Area of Specialisation Match",
    min_value=0.0, max_value=1.0, value=0.25, step=0.05,
    help=(
        "**Area of Specialisation Match**\n\n"
        "Evaluates how closely the coachee's branch of study at IIT Madras aligns with the mentor's area of professional specialisation.\n\n"
        "1. The number of branches (areas of specialisation) at IIT Madras have increased over the years.\n\n"
        "2. Applied mechanics was a part of mechanical engineering.\n\n" \
        "3. Biotechnology, Biomedical engineering and Engineering Design, are relatively new areas.\n\n"
        "4. Data science is an area of specialization for those doing dual degree programss.\n\n"
        "📌 *Adjust the weight if technical domain alignment as needed.*"
    )
)

w_deg = st.sidebar.slider(
    "Degree Match",
    min_value=0.0, max_value=1.0, value=0.15, step=0.05,
    help=(
        "**Degree Match**\n\n"
        "Matechs the academic level of the coachee's current programme with the mentor's degree in IIT Madras or from other Institute:\n\n"
        "1. The degree broadly grouped in Undergraduate, Dual Degree, Masters, MBA, PhD.\n\n"
        "2. The Degree will be matched with similar degree if satisfies otherwise will look for other degree if other critatia matched. \n\n"
        "3. All coachees who are currently pursuing an MBA will be paired with mentor coaches who either did their MBA from IIT Madras or have" 
        "done an MBA from another Indian institute after graduating from IIT Madras.\n\n"
        "4. All coachees who are currently pursuing a PhD degree will be paired with mentor coaches who have done a PhD either from IIT Madras or elsewhere.\n\n"
        "📌 *Adjusrt the weight if acadamic background similarity as needed.*"
    )
)

# Soft Skills ( Professional goals, Personal interests, IIT experience, Background & Values)

w_prof = st.sidebar.slider(
    "Professional or career Match",
    min_value=0.0, max_value=1.0, value=0.20, step=0.05,
    help=(
        "**Professional or career Match**\n\n"
        "1. Uses coachee's Career Plan to match with the mentor's Career Snapshot.\n\n"
        "2. The critaria identifies shared professional keywords such as industry, role type, and goals such as both mentioning"
         " product management , startups or 'research.\n\n"
        "📌 *Adjust the weight accordingly.*"
    )
)

w_pers = st.sidebar.slider(
    "Personal Fit Match",
    min_value=0.0, max_value=1.0, value=0.15, step=0.05,
    help=(
        "**Personal Fit Match**\n\n"
        "1. Matches the shared coachee's *Top 3 Interests* and *Main Passions* with the mentor's *Interests*.\n\n"
        "2. Captures shared hobbies, extracurriculars, and personal motivations such as music, hobbies like Reading, specific genre sports, social impact.\n\n"
        "📌 *Adjust the weight accordingly.*"
    )
)

w_iit = st.sidebar.slider(
    "IIT Experience Match",
    min_value=0.0, max_value=1.0, value=0.15, step=0.05,
    help=(
        "**IIT Experience Match**\n\n"
        "1. Analyses the  shared IIT experiences of coachee's *IIT Trajectory* and *Career Plan* alongside the mentor's *IIT Experience*.\n\n"
        "2. Both discuss campus life, clubs, academic struggles/ challenges, campus life and specific IIT Madras hostels or events.\n\n"
        "📌 *Adjust the weight accordingly.*"
    )
)

w_back = st.sidebar.slider(
    "Family Background & Inspration Match",
    min_value=0.0, max_value=1.0, value=0.10, step=0.05,
    help=(
        "**Family Background & Inspration Match**\n\n"
        "1. Matches the coachee’s home-town and growing up years prior to IIT Madras and Role Models with the mentor's *Growing Up Years*.\n\n"
        "2. The information help in matching mentor-coach who has a similar ethnic or family background.\n\n"
        "📌 *Adjust the weight accordingly.*"
    )
)

# Gender Preference
st.sidebar.markdown("### Gender Preference")

bonus_female = st.sidebar.slider(
    "Gender Preference Matches",
    min_value=0.0, max_value=0.5, value=0.10, step=0.05,
    help=(
        "**Gender Preference Matches**\n\n"
        "1. The female coachees avialability is usually higher than female mentor-coaches. Therefore the there should be a preferences to "
        "pair as many female coachees as possible with a female mentor coach who has the best matching critarias with mentors. But this should not be a hard requirement.\n\n"
        "2. This Adds a small score boost when both the coachee and the mentor identify as female.\n\n"
        "3. The remaining female coachees will be paired with a male mentor coach.\n\n"
        "4. The Weitage of gender prefereces usually kept low for not to suppress the other important critaria.\n\n"
        "📌 *Adjust the weight accordingly.*"
    )
)

# Weight Summary 
total_w = w_spec + w_deg + w_prof + w_pers + w_iit + w_back
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Weight Summary")

weight_data = {
    "Criterion": ["Specialisation", "Degree", "Professional", "Personal", "IIT Context", "Background"],
    "Weight":    [w_spec, w_deg, w_prof, w_pers, w_iit, w_back],
}
weight_df = pd.DataFrame(weight_data)
weight_df["Share (%)"] = (
    (weight_df["Weight"] / total_w * 100).round(1) if total_w > 0 else 0
)
st.sidebar.dataframe(
    weight_df.set_index("Criterion"),
    use_container_width=True,
    height=235,
)

if abs(total_w - 1.0) > 0.01:
    st.sidebar.warning(
        f"⚠️ Weights sum to **{total_w:.2f}** (not 1.0). "
        "Scores are still valid but consider normalising for best results."
    )
else:
    st.sidebar.success(f"✅ Weights sum to **{total_w:.2f}** — balanced!")

# ─── Helper Functions ──────────────────────────────────────────────────────────
def clean(text):
    return str(text).lower().strip() if pd.notnull(text) else ""

def get_degree_group(text):
    t = clean(text)
    if any(x in t for x in ['b.tech', 'btech', 'b. tech', 'undergraduate', 'bachelor']): return 1
    if any(x in t for x in ['dual degree', 'b.s - m.s', 'bs-ms', 'iddd']): return 2
    if any(x in t for x in ['masters', 'm.tech', 'm. tech', 'm.s', 'msc']): return 3
    if any(x in t for x in ['mba', 'emba']): return 4
    if any(x in t for x in ['phd', 'doctorate']): return 5
    return 0

c_branch_map = {
    'engineering design': 1, 'biotechnology': 2, 'biological': 2, 'civil': 3,
    'physics': 4, 'mechanical': 5, 'applied mechanics': 5, 'chemical': 6,
    'computer science': 7, 'data science': 7, 'electrical': 8, 'electronics': 8,
    'metallurgical': 9, 'aerospace': 10, 'management': 11, 'naval': 12,
    'ocean': 12, 'mathematics': 13, 'humanities': 14,
}
m_spec_map = {
    'microbiology': 1, 'bio': 1, 'physics': 2, 'civil': 3, 'mechanical': 4,
    'chemical': 5, 'computer': 6, 'cs': 6, 'electrical': 7, 'electronics': 7,
    'metallurgical': 8, 'aeronautical': 9, 'management': 10, 'finance': 10,
    'naval': 11, 'math': 12,
}
spec_match_logic = {
    1: [4], 2: [1], 3: [3], 4: [2], 5: [4], 6: [5], 7: [6], 8: [7],
    9: [8], 10: [9], 11: [10], 12: [11], 13: [12], 14: [],
}

def get_group(text, mapping):
    t = clean(text)
    for k, v in mapping.items():
        if k in t: return v
    return 0

def load_data(file):
    """Detects file type and loads into a Pandas DataFrame."""
    file_name = file.name.lower()
    try:
        if file_name.endswith('.csv'):
            try:
                return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
        elif file_name.endswith('.xlsx'):
            return pd.read_excel(file, engine='openpyxl')
        elif file_name.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                all_rows = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        all_rows.extend(table)
                if all_rows:
                    df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
                    df.dropna(how='all', inplace=True)
                    return df
                else:
                    st.error(f"⚠️ Could not find a readable table in {file.name}.")
                    return None
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return None

# ─── File Upload Section ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📂 Upload Data Files")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="upload-label">👩‍🎓 Coachee Data</div>', unsafe_allow_html=True)
    coachee_file = st.file_uploader(
        "Upload Coachee responses",
        type=['csv', 'xlsx', 'pdf'],
        label_visibility="collapsed",
        key="coachee_upload",
        help="Upload the file containing coachee survey responses. Supported: CSV, Excel (.xlsx), PDF.",
    )
    if coachee_file:
        st.success(f"✅ Loaded: `{coachee_file.name}`")

with col2:
    st.markdown('<div class="upload-label">👨‍💼 Mentor Data</div>', unsafe_allow_html=True)
    mentor_file = st.file_uploader(
        "Upload Mentor responses",
        type=['csv', 'xlsx', 'pdf'],
        label_visibility="collapsed",
        key="mentor_upload",
        help="Upload the file containing mentor survey responses. Supported: CSV, Excel (.xlsx), PDF.",
    )
    if mentor_file:
        st.success(f"✅ Loaded: `{mentor_file.name}`")

# ─── Run Matching ──────────────────────────────────────────────────────────────
st.markdown("---")

if coachee_file and mentor_file:
    if st.button("🚀 Run Matching Algorithm", type="primary", use_container_width=True):
        with st.spinner("Analysing text and calculating optimal matches…"):

            coachee_df = load_data(coachee_file)
            mentor_df  = load_data(mentor_file)

            if coachee_df is None or mentor_df is None:
                st.stop()

            # ── Preprocess ──────────────────────────────────────────────────
            coachee_df['Batch'] = coachee_df['Map Code/Coachee mapping'].astype(str).apply(
                lambda x: x.split('-')[1] if '-' in x else '0')
            mentor_df['Batch']  = mentor_df['Mentor ID'].astype(str).apply(
                lambda x: x.split('-')[1] if '-' in x else '0')

            coachee_df['Deg_Grp']    = coachee_df['Program at IIT Madras'].apply(get_degree_group)
            mentor_df['Deg_Grp']     = mentor_df['Degree'].apply(get_degree_group)
            coachee_df['Branch_Grp'] = coachee_df['Branch at IIT Madras'].apply(
                lambda x: get_group(x, c_branch_map))
            mentor_df['Spec_Grp']    = mentor_df['Specialisation'].apply(
                lambda x: get_group(x, m_spec_map))

            # ── Combine Text Columns ─────────────────────────────────────────
            def combine(df, cols):
                return df[cols].fillna('').apply(lambda x: ' '.join(x), axis=1).apply(clean)

            coachee_df['Txt_Prof'] = coachee_df['Career plan'].apply(clean)
            mentor_df['Txt_Prof']  = mentor_df['Career snapshot'].apply(clean)
            coachee_df['Txt_Pers'] = combine(coachee_df, ['Top 3 interests', 'Main passions'])
            mentor_df['Txt_Pers']  = mentor_df['Interests'].apply(clean)
            coachee_df['Txt_IIT']  = combine(coachee_df, ['IIT trajectory', 'Career plan'])
            mentor_df['Txt_IIT']   = mentor_df['IIT experience'].apply(clean)
            coachee_df['Txt_Back'] = combine(coachee_df, ['Family info and schooling', 'Roll Models'])
            mentor_df['Txt_Back']  = mentor_df['Growing up years'].apply(clean)

            # ── Vectorise ────────────────────────────────────────────────────
            vec_prof = TfidfVectorizer(stop_words='english').fit(
                pd.concat([coachee_df['Txt_Prof'], mentor_df['Txt_Prof']]))
            vec_pers = TfidfVectorizer(stop_words='english').fit(
                pd.concat([coachee_df['Txt_Pers'], mentor_df['Txt_Pers']]))
            vec_iit  = TfidfVectorizer(stop_words='english').fit(
                pd.concat([coachee_df['Txt_IIT'], mentor_df['Txt_IIT']]))
            vec_back = TfidfVectorizer(stop_words='english').fit(
                pd.concat([coachee_df['Txt_Back'], mentor_df['Txt_Back']]))

            def normalize(arr):
                if len(arr) < 2 or arr.max() == 0: return arr
                return (arr - arr.min()) / (arr.max() - arr.min())

            final_matches = []

            for idx, c_row in coachee_df.iterrows():
                c_id     = c_row['Map Code/Coachee mapping']
                batch    = c_row['Batch']
                c_gender = clean(c_row.get('Gender ', c_row.get('Gender', '')))

                candidates = mentor_df[mentor_df['Batch'] == batch].copy()
                if candidates.empty:
                    continue

                s_prof = normalize(cosine_similarity(
                    vec_prof.transform([c_row['Txt_Prof']]),
                    vec_prof.transform(candidates['Txt_Prof'])).flatten())
                s_pers = normalize(cosine_similarity(
                    vec_pers.transform([c_row['Txt_Pers']]),
                    vec_pers.transform(candidates['Txt_Pers'])).flatten())
                s_iit  = normalize(cosine_similarity(
                    vec_iit.transform([c_row['Txt_IIT']]),
                    vec_iit.transform(candidates['Txt_IIT'])).flatten())
                s_back = normalize(cosine_similarity(
                    vec_back.transform([c_row['Txt_Back']]),
                    vec_back.transform(candidates['Txt_Back'])).flatten())

                scores = []
                for i, (_, m_row) in enumerate(candidates.iterrows()):
                    sc_spec = 1.0 if (
                        c_row['Branch_Grp'] in spec_match_logic and
                        m_row['Spec_Grp'] in spec_match_logic[c_row['Branch_Grp']]
                    ) else 0.0
                    sc_deg = 1.0 if (
                        (c_row['Deg_Grp'] == m_row['Deg_Grp'] and c_row['Deg_Grp'] != 0) or
                        (c_row['Deg_Grp'] == 1 and m_row['Deg_Grp'] == 2) or
                        (c_row['Deg_Grp'] == 2 and m_row['Deg_Grp'] == 1)
                    ) else 0.0

                    total = (
                        (sc_spec   * w_spec) +
                        (sc_deg    * w_deg)  +
                        (s_prof[i] * w_prof) +
                        (s_pers[i] * w_pers) +
                        (s_iit[i]  * w_iit)  +
                        (s_back[i] * w_back)
                    )
                    if 'female' in c_gender and 'female' in clean(m_row.get('Gender', '')):
                        total += bonus_female

                    # include detailed breakdown in the details string for transparency
                    details_str = f"(Tot:{total:.2f}), (H:SP{int(sc_spec)}D{int(sc_deg)}), (S:Pr{s_prof[i]:.1f}, Pe{s_pers[i]:.1f}, IX{s_iit[i]:.1f}, FB{s_back[i]:.1f})"

                    # Append to the scores list 
                    scores.append({'id': m_row['Mentor ID'], 'score': total, 'details': details_str})

                scores.sort(key=lambda x: x['score'], reverse=True)
                top3, seen = [], set()
                for s in scores:
                    if s['id'] not in seen:
                        top3.append(s); seen.add(s['id'])
                    if len(top3) == 3: break

                row = {'Coachee Code': c_id}
                for k in range(3):
                    if k < len(top3):
                        row[f'Option {k+1} Mentor ID']  = top3[k]['id']
                        row[f'Option {k+1} Score (%)']  = round(top3[k]['score'] * 100, 1)
                        row[f'Option {k+1} Details']    = top3[k]['details']
                    else:
                        row[f'Option {k+1} Mentor ID']  = "N/A"
                        row[f'Option {k+1} Score (%)']  = "—"
                        row[f'Option {k+1} Details']    = "N/A"  
                final_matches.append(row)

            res_df = pd.DataFrame(final_matches)

        # ── Results ──────────────────────────────────────────────────────────
        st.success("✅ Matching complete!")
        st.markdown('<div class="results-title">📋 Top Match Results</div>', unsafe_allow_html=True)

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Coachees Processed", len(res_df))
        m2.metric("Total Mentors Available",  len(mentor_df))
        try:
            avg_score = pd.to_numeric(
                res_df['Option 1 Score (%)'].replace("—", np.nan), errors='coerce'
            ).mean()
            m3.metric("Avg Top-1 Match Score", f"{avg_score:.1f}%")
        except Exception:
            m3.metric("Avg Top-1 Match Score", "—")

        st.markdown("#### Preview (first 10 rows)")
        st.dataframe(res_df.head(10), use_container_width=True, hide_index=True)

        with st.expander("🔍 View All Results", expanded=False):
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv,
            file_name="Mentor_Coachee_Matches.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.markdown(
        """
        <div style="text-align:center; padding:40px; background:#f8fafc;
                    border-radius:12px; border:2px dashed #cbd5e1; color:#64748b;">
            <div style="font-size:2.5rem;">📁</div>
            <div style="font-size:1.1rem; font-weight:600; margin-top:8px;">
                Upload both files above to get started
            </div>
            <div style="font-size:0.9rem; margin-top:6px;">
                Load your Coachee and Mentor datasets, then click <strong>Run Matching Algorithm</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
