"""
app.py — Non-atrophic AD CDSS (Clinical Decision Support System)

실행:
  pip install streamlit matplotlib numpy
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stage_a import classify_subtype, compute_domain_zscores, plot_radar_chart, generate_profile_summary
from stage_b import compute_priority, generate_guideline_text, generate_priority_table
from stage_c import get_all_evidence
from config import CN_REFERENCE, CLUSTER_CENTERS_RAW, COLORS

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Non-atrophic AD CDSS",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 비위축 AD 임상 보조 시스템 (CDSS)")
st.markdown("""
**Non-atrophic Alzheimer's Disease — Clinical Decision Support System**  
해마 위축이 없는 AD 환자의 인지 프로파일을 분석하고, 개인화된 중재 가이드라인을 제공합니다.

> ⚠️ 본 시스템은 ADNI 기반 연구 결과에 근거한 **참고 도구**이며, 최종 치료 결정은 담당 의사의 판단에 따릅니다.
""")

st.divider()

# ============================================================
# Sidebar: 환자 점수 입력
# ============================================================
st.sidebar.header("📝 환자 인지검사 점수 입력")
st.sidebar.markdown("6개 인지검사 점수를 입력하세요.")

patient_id = st.sidebar.text_input("환자 ID", value="Patient_001")

scores = {}
st.sidebar.subheader("검사 점수")

scores['CATANIMSC'] = st.sidebar.number_input(
    "🗣️ CATANIMSC (Category Fluency - Animals)",
    min_value=0, max_value=40, value=10,
    help="Number of animals named in 1 min. CN mean ≈ 20. ⬆ Higher = Better"
)

scores['TRABSCOR'] = st.sidebar.number_input(
    "⏱️ TRABSCOR (Trail Making B, seconds)",
    min_value=30, max_value=500, value=200,
    help="Time to complete TMT-B (sec). CN mean ≈ 82. ⬇ Lower = Better"
)

scores['LDELTOTL'] = st.sidebar.number_input(
    "🧠 LDELTOTL (Logical Memory Delayed Recall)",
    min_value=0, max_value=25, value=2,
    help="Delayed recall score. CN mean ≈ 7.5. ⬆ Higher = Better"
)

scores['MMSCORE'] = st.sidebar.number_input(
    "📋 MMSCORE (MMSE)",
    min_value=0, max_value=30, value=23,
    help="Mini-Mental State Exam. CN mean ≈ 29. ⬆ Higher = Better"
)

scores['CDRSB'] = st.sidebar.number_input(
    "📊 CDRSB (CDR Sum of Boxes)",
    min_value=0.0, max_value=18.0, value=4.5, step=0.5,
    help="Dementia severity. CN ≈ 0. ⬇ Lower = Better"
)

scores['ADASTT13'] = st.sidebar.number_input(
    "📝 ADASTT13 (ADAS-Cog 13)",
    min_value=0, max_value=85, value=30,
    help="AD cognitive assessment. CN mean ≈ 9. ⬇ Lower = Better"
)

run_button = st.sidebar.button("🔍 분석 실행", type="primary", use_container_width=True)

# ============================================================
# 예시 환자 버튼
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("📋 예시 환자")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("보존형 예시", use_container_width=True):
        scores = {'CATANIMSC': 16, 'TRABSCOR': 130, 'LDELTOTL': 3, 'MMSCORE': 25, 'CDRSB': 3.5, 'ADASTT13': 23}
        patient_id = "Example_Preserved"
        run_button = True
with col2:
    if st.button("저하형 예시", use_container_width=True):
        scores = {'CATANIMSC': 8, 'TRABSCOR': 250, 'LDELTOTL': 1, 'MMSCORE': 22, 'CDRSB': 5.5, 'ADASTT13': 35}
        patient_id = "Example_Impaired"
        run_button = True


# ============================================================
# Main: 분석 결과
# ============================================================
if run_button:

    # ------ Stage A ------
    st.header("Stage A: 환자 프로파일링")

    classification = classify_subtype(scores)
    domain_results = compute_domain_zscores(scores)

    # Subtype 표시
    col_sub1, col_sub2, col_sub3 = st.columns([1, 1, 1])
    with col_sub1:
        color = COLORS['preserved'] if classification['cluster'] == 0 else COLORS['impaired']
        st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:white; margin:0;">{classification['subtype']}</h2>
            <p style="color:white; margin:5px 0 0 0;">Confidence: {classification['confidence']:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(classification['confidence'])
    with col_sub2:
        worst = min(domain_results.items(), key=lambda x: x[1]['z_score'])
        st.metric("가장 취약한 영역", worst[0], f"Z = {worst[1]['z_score']:.1f}")
    with col_sub3:
        normal_count = sum(1 for d in domain_results.values() if d['z_score'] > -1.0)
        st.metric("정상 범위 영역", f"{normal_count}/5", "")

    # 레이더 차트
    st.subheader("Cognitive Domain Profile (vs. CN)")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        fig = plot_radar_chart(domain_results, patient_id)
        st.pyplot(fig, use_container_width=False)
    plt.close()

    # 영역별 상세 테이블
    st.subheader("영역별 상세")

    # ← 여기부터 추가
    for domain, info in sorted(domain_results.items(), key=lambda x: x[1]['z_score']):
        if info['z_score'] > -1.0:
            st.success(f"✅ {domain}: Z={info['z_score']:.1f} — Normal Range")
        elif info['z_score'] > -2.0:
            st.warning(f"⚠️ {domain}: Z={info['z_score']:.1f} — Mild Impairment")
        else:
            st.error(f"🔴 {domain}: Z={info['z_score']:.1f} — Moderate/Severe Impairment")
    # ← 여기까지 추가

    domain_table = []
    
    
    domain_table = []
    for domain, info in sorted(domain_results.items(), key=lambda x: x[1]['z_score']):
        domain_table.append({
            '영역': domain,
            'Z-score': f"{info['z_score']:.1f}",
            '심각도': info['severity'],
            '관련 뇌 영역': info['brain_region'],
            '설명': info['description'],
        })
    st.dataframe(pd.DataFrame(domain_table), use_container_width=True, hide_index=True)

    st.divider()

    # ------ Stage B ------
    st.header("Stage B: 중재 우선순위 가이드라인")

    priorities = compute_priority(domain_results)

    # 우선순위 테이블
    priority_table = generate_priority_table(priorities)
    st.dataframe(pd.DataFrame(priority_table), use_container_width=True, hide_index=True)

    # ← 여기부터 추가
    actionable = [p for p in priorities if p['needs_intervention']]
    if actionable:
        priority_df = pd.DataFrame([
            {'Domain': p['domain'], 'Priority Score': p['priority_score']}
            for p in actionable
        ])
        st.bar_chart(priority_df.set_index('Domain'))
    # ← 여기까지 추가


    # 가이드라인 텍스트
    guideline = generate_guideline_text(classification, priorities)
    st.markdown(guideline)

    st.divider()

    # ------ Stage C ------
    st.header("Stage C: 근거 문헌 검색 (RAG)")
    st.markdown("*각 중재 영역에 대한 관련 논문을 검색합니다.*")

    evidence = get_all_evidence(priorities)

    for domain, ev_info in evidence.items():
        with st.expander(f"📄 {domain} — 근거 문헌 ({len(ev_info['papers'])}편)", expanded=False):
            st.markdown(ev_info['formatted'])

    # ------ 전체 요약 ------
    st.divider()
    st.header("📋 전체 요약")
    summary = generate_profile_summary(classification, domain_results)
    st.markdown(summary)

    # 입력 점수 확인
    with st.expander("입력된 점수 확인"):
        score_df = pd.DataFrame([{
            '변수': var,
            '입력값': scores[var],
            'CN 평균': CN_REFERENCE[var]['mean'],
            'CN SD': CN_REFERENCE[var]['std'],
        } for var in scores])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Enter scores in the sidebar and click 'Run Analysis'.")

    st.subheader("How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### Stage A
        **Patient Profiling**
        - Subtype classification
        - Z-score radar chart
        - Cognitive domain analysis
        """)
    with col2:
        st.markdown("""
        ### Stage B
        **Priority Guidelines**
        - Severity × Modifiability
        - Ranked intervention targets
        - Domain-specific recommendations
        """)
    with col3:
        st.markdown("""
        ### Stage C
        **Evidence Search**
        - Literature-based evidence
        - Top 3 papers per domain
        - RCT & meta-analysis priority
        """)
# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("""
**Non-atrophic AD CDSS v0.1** | ADNI 데이터 기반 | 연구 참고용  
본 시스템의 결과는 임상적 판단을 대체하지 않습니다.
""")
