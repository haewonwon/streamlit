import io
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(
    page_title="🌱 극지식물 최적 EC 농도 연구",
    layout="wide",
)

# 한글 폰트(앱 전체) - Streamlit CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"


# =========================
# 상수(실험 조건)
# =========================
SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]

EC_TARGET_BY_SCHOOL = {
    "송도고": 1.0,
    "하늘고": 2.0,  # (최적) 강조
    "아라고": 4.0,
    "동산고": 8.0,
}

COLOR_BY_SCHOOL = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728",
}


# =========================
# 유니코드 정규화 유틸 (수정됨)
# =========================
def _norm_variants(text: str) -> Tuple[str, str]:
    """NFC/NFD 두 형태를 모두 반환."""
    return (
        unicodedata.normalize("NFC", text),
        unicodedata.normalize("NFD", text),
    )


def _contains_token(text: str, token: str) -> bool:
    """텍스트에 토큰이 포함되는지(한글 자소 분리 무시) 확인"""
    text_nfc, text_nfd = _norm_variants(text)
    token_nfc, token_nfd = _norm_variants(token)
    return (token_nfc in text_nfc) or (token_nfd in text_nfd) or (token_nfc in text_nfd) or (token_nfd in text_nfc)


def _find_data_files(data_dir: Path) -> Tuple[Dict[str, Path], List[Path]]:
    """
    CSV 환경 데이터와 (XLSX 또는 CSV) 생육 데이터를 모두 찾습니다.
    """
    env_csv_by_school: Dict[str, Path] = {}
    growth_files: List[Path] = []  # xlsx 하나거나, csv 여러개

    if not data_dir.exists():
        return env_csv_by_school, growth_files

    for p in data_dir.iterdir():
        if not p.is_file():
            continue

        name = p.name

        # 1. 환경 데이터 (CSV) 찾기 ("환경" + 학교명)
        if _contains_token(name, "환경"):
            for sc in SCHOOLS:
                if _contains_token(name, sc):
                    env_csv_by_school[sc] = p
                    break

        # 2. 생육 데이터 (XLSX 또는 CSV) 찾기 ("생육" 또는 "결과")
        # 엑셀 파일이든, 분리된 CSV 파일이든 '생육'이라는 글자가 있으면 후보로 등록
        if _contains_token(name, "생육") or _contains_token(name, "결과"):
            if name.lower().endswith((".xlsx", ".csv")):
                growth_files.append(p)

    return env_csv_by_school, growth_files


# =========================
# 데이터 로딩(캐시)
# =========================
@st.cache_data(show_spinner=False)
def load_environment_data(data_dir: str) -> pd.DataFrame:
    """
    data/의 학교별 환경 CSV를 모두 읽어서 하나로 합침.
    컬럼: time, temperature, humidity, ph, ec, school
    """
    data_path = Path(data_dir)
    env_csv_by_school, _ = _find_data_files(data_path)

    if not env_csv_by_school:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for sc, fp in env_csv_by_school.items():
        try:
            df = pd.read_csv(fp)  # encoding="utf-8-sig" or "cp949" may be needed sometimes
            df["school"] = sc
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    env = pd.concat(frames, ignore_index=True)

    # (이하 데이터 전처리 로직은 기존과 동일)
    if "time" in env.columns:
        env["time"] = pd.to_datetime(env["time"], errors="coerce")
    else:
        env["time"] = pd.NaT

    for c in ["temperature", "humidity", "ph", "ec"]:
        if c in env.columns:
            env[c] = pd.to_numeric(env[c], errors="coerce")
        else:
            env[c] = pd.NA

    return env.sort_values(["school", "time"], kind="stable")


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir: str) -> pd.DataFrame:
    """
    data/의 xlsx(4개 시트)를 모두 읽어서 하나로 합침.
    ✅ 시트 이름 하드코딩 금지: ExcelFile().sheet_names 동적 사용
    컬럼(원본): 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g)
    + school, ec_target
    """
    data_path = Path(data_dir)
    _, growth_files = _find_data_files(data_path)

    if not growth_files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []

    for fp in growth_files:
        try:
            # 엑셀 파일인 경우 (기존 로직)
            if fp.suffix.lower() == ".xlsx":
                xl = pd.ExcelFile(fp, engine="openpyxl")
                for sh in xl.sheet_names:
                    df = pd.read_excel(xl, sheet_name=sh)
                    # 시트명에서 학교 찾기
                    matched_school = None
                    for sc in SCHOOLS:
                        if _contains_token(str(sh), sc):
                            matched_school = sc
                            break

                    if matched_school:
                        df["school"] = matched_school
                        df["ec_target"] = EC_TARGET_BY_SCHOOL.get(matched_school, pd.NA)
                        frames.append(df)

            # CSV 파일인 경우 (새로 추가된 로직)
            elif fp.suffix.lower() == ".csv":
                # 파일명에서 학교 찾기 (예: "4개교_생육결과데이터... - 동산고.csv")
                matched_school = None
                for sc in SCHOOLS:
                    if _contains_token(fp.name, sc):
                        matched_school = sc
                        break

                if matched_school:
                    df = pd.read_csv(fp)
                    df["school"] = matched_school
                    df["ec_target"] = EC_TARGET_BY_SCHOOL.get(matched_school, pd.NA)
                    frames.append(df)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    growth = pd.concat(frames, ignore_index=True)

    num_cols = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    for c in num_cols:
        if c in growth.columns:
            growth[c] = pd.to_numeric(growth[c], errors="coerce")

    return growth


# =========================
# 시각화 헬퍼
# =========================
def apply_plotly_korean_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
    return fig


def _statsmodels_available() -> bool:
    """statsmodels 가 설치되어 있는지 안전하게 검사합니다."""
    try:
        import statsmodels  # type: ignore

        return True
    except Exception:
        return False


def safe_mean(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return None
    return float(s.mean())


def format_float(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def make_ec_target_table() -> pd.DataFrame:
    rows = []
    for sc in SCHOOLS:
        rows.append(
            {
                "학교명": sc,
                "EC 목표": EC_TARGET_BY_SCHOOL.get(sc),
                "개체수(시트 기준)": None,  # 생육 데이터 로딩 후 채움
                "색상": COLOR_BY_SCHOOL.get(sc),
            }
        )
    return pd.DataFrame(rows)


# =========================
# 사이드바
# =========================
st.sidebar.title("⚙️ 설정")
selected_school = st.sidebar.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)

data_dir = str(Path(__file__).parent / "data")


# =========================
# 데이터 로딩(스피너 + 에러)
# =========================
with st.spinner("데이터를 불러오는 중..."):
    env_df = load_environment_data(data_dir)
    growth_df = load_growth_data(data_dir)

if env_df.empty:
    st.error("환경 데이터(CSV)를 찾거나 읽지 못했습니다. data/ 폴더의 파일명(한글 포함)과 형식을 확인해주세요.")
if growth_df.empty:
    st.error("생육 결과 데이터(XLSX)를 찾거나 읽지 못했습니다. data/ 폴더의 엑셀 파일 및 시트를 확인해주세요.")

st.title("🌱 극지식물 최적 EC 농도 연구")


# =========================
# 공통: 필터링
# =========================
def filter_by_school(df: pd.DataFrame, school: str) -> pd.DataFrame:
    if df.empty:
        return df
    if school == "전체":
        return df
    if "school" not in df.columns:
        return df
    return df[df["school"] == school]


env_sel = filter_by_school(env_df, selected_school)
growth_sel = filter_by_school(growth_df, selected_school)


# =========================
# 탭 구성
# =========================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ==========================================================
# Tab 1: 실험 개요
# ==========================================================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
극지식물의 생육 최적화를 위해 **EC(양액 전기전도도)** 농도 조건을 달리하여 생육 결과를 비교합니다.  
4개 학교는 서로 다른 EC 목표 조건에서 재배를 수행했으며, 각 학교의 환경(온도/습도/pH/EC)과 생육 지표를 종합해 **최적 EC 농도**를 도출합니다.
"""
    )

    # 학교별 EC 조건 표
    st.subheader("학교별 EC 조건")
    ec_table = make_ec_target_table()

    # 생육 시트 기준 개체수 채우기(가능하면)
    if not growth_df.empty and "school" in growth_df.columns:
        counts = growth_df.groupby("school", dropna=False).size().to_dict()
        ec_table["개체수(시트 기준)"] = ec_table["학교명"].map(counts).fillna(0).astype(int)

    st.dataframe(ec_table, use_container_width=True)

    # 주요 지표 카드 4개
    st.subheader("주요 지표")

    total_individuals = None
    if not growth_df.empty:
        total_individuals = int(growth_df.shape[0])

    avg_temp = None
    avg_hum = None
    if not env_df.empty:
        avg_temp = safe_mean(env_df.get("temperature", pd.Series(dtype=float)))
        avg_hum = safe_mean(env_df.get("humidity", pd.Series(dtype=float)))

    # 최적 EC: 생중량 평균이 최대인 EC
    optimal_ec = None
    if not growth_df.empty and "생중량(g)" in growth_df.columns and "ec_target" in growth_df.columns:
        tmp = growth_df.dropna(subset=["ec_target", "생중량(g)"]).copy()
        if not tmp.empty:
            by_ec = tmp.groupby("ec_target")["생중량(g)"].mean().sort_values(ascending=False)
            if not by_ec.empty:
                optimal_ec = float(by_ec.index[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", "-" if total_individuals is None else f"{total_individuals:,}")
    c2.metric("평균 온도(°C)", format_float(avg_temp, 2))
    c3.metric("평균 습도(%)", format_float(avg_hum, 2))
    c4.metric("최적 EC", "-" if optimal_ec is None else f"{optimal_ec:.1f}")


# ==========================================================
# Tab 2: 환경 데이터
# ==========================================================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    if env_df.empty:
        st.error("환경 데이터가 없어 시각화를 표시할 수 없습니다.")
    else:
        # 학교별 평균
        env_mean = (
            env_df.groupby("school", dropna=False)[["temperature", "humidity", "ph", "ec"]]
            .mean(numeric_only=True)
            .reset_index()
        )

        # 목표 EC 컬럼 추가
        env_mean["ec_target"] = env_mean["school"].map(EC_TARGET_BY_SCHOOL)

        # 2x2 서브플롯
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC"),
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        # 공통 x
        x_sch = env_mean["school"].tolist()

        # (1,1) 평균 온도
        fig.add_trace(
            go.Bar(
                x=x_sch,
                y=env_mean["temperature"],
                name="평균 온도",
            ),
            row=1,
            col=1,
        )

        # (1,2) 평균 습도
        fig.add_trace(
            go.Bar(
                x=x_sch,
                y=env_mean["humidity"],
                name="평균 습도",
            ),
            row=1,
            col=2,
        )

        # (2,1) 평균 pH
        fig.add_trace(
            go.Bar(
                x=x_sch,
                y=env_mean["ph"],
                name="평균 pH",
            ),
            row=2,
            col=1,
        )

        # (2,2) 목표 EC vs 실측 EC (이중 막대)
        fig.add_trace(
            go.Bar(
                x=x_sch,
                y=env_mean["ec_target"],
                name="목표 EC",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=x_sch,
                y=env_mean["ec"],
                name="실측 EC(평균)",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            barmode="group",
            height=700,
            margin=dict(l=30, r=30, t=70, b=30),
        )
        fig = apply_plotly_korean_font(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    if env_sel.empty:
        st.error("선택한 조건에 해당하는 환경 데이터가 없습니다.")
    else:
        # time이 NaT인 경우 대비
        if env_sel["time"].isna().all():
            st.error("time 컬럼을 날짜/시간으로 변환하지 못했습니다. CSV의 time 형식을 확인해주세요.")
        else:
            env_ts = env_sel.dropna(subset=["time"]).copy()

            target_ec = None
            if selected_school != "전체":
                target_ec = EC_TARGET_BY_SCHOOL.get(selected_school)

            # 온도
            fig_t = px.line(env_ts, x="time", y="temperature", title="온도 변화")
            fig_t = apply_plotly_korean_font(fig_t)
            st.plotly_chart(fig_t, use_container_width=True)

            # 습도
            fig_h = px.line(env_ts, x="time", y="humidity", title="습도 변화")
            fig_h = apply_plotly_korean_font(fig_h)
            st.plotly_chart(fig_h, use_container_width=True)

            # EC (목표 EC 수평선)
            fig_ec = px.line(env_ts, x="time", y="ec", title="EC 변화 (목표 EC 포함)")
            if target_ec is not None:
                fig_ec.add_hline(y=target_ec, line_dash="dash", annotation_text=f"목표 EC {target_ec:.1f}")
            fig_ec = apply_plotly_korean_font(fig_ec)
            st.plotly_chart(fig_ec, use_container_width=True)

    st.divider()
    with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
        if env_sel.empty:
            st.error("표시할 환경 데이터가 없습니다.")
        else:
            st.dataframe(env_sel, use_container_width=True)

            csv_bytes = env_sel.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="환경 데이터 CSV 다운로드",
                data=csv_bytes,
                file_name="환경데이터_선택.csv",
                mime="text/csv",
            )


# ==========================================================
# Tab 3: 생육 결과
# ==========================================================
with tab3:
    st.subheader("🥇 핵심 결과")

    if growth_df.empty:
        st.error("생육 결과 데이터가 없어 시각화를 표시할 수 없습니다.")
    else:
        # EC별 평균 생중량
        core_ready = growth_df.dropna(subset=["ec_target", "생중량(g)"]).copy()
        if core_ready.empty:
            st.error("ec_target 또는 생중량(g) 데이터가 부족해 핵심 결과를 계산할 수 없습니다.")
        else:
            mean_weight_by_ec = (
                core_ready.groupby("ec_target")["생중량(g)"].mean().sort_index().reset_index(name="평균 생중량")
            )

            # 최댓값(강조)
            max_row = mean_weight_by_ec.loc[mean_weight_by_ec["평균 생중량"].idxmax()]
            max_ec = float(max_row["ec_target"])
            max_w = float(max_row["평균 생중량"])

            # 카드(하늘고 EC 2.0 최적 강조: 데이터상 최댓값이 2.0이면 자연히 강조됨)
            c1, c2 = st.columns([1, 2])
            c1.metric("최대 평균 생중량", f"{max_w:.3f} g", delta=f"EC {max_ec:.1f}")

            # 표/그래프
            fig_core = px.bar(
                mean_weight_by_ec,
                x="ec_target",
                y="평균 생중량",
                title="EC별 평균 생중량",
                labels={"ec_target": "EC", "평균 생중량": "평균 생중량(g)"},
            )
            # 최댓값 포인트 표시
            fig_core.add_trace(
                go.Scatter(
                    x=[max_ec],
                    y=[max_w],
                    mode="markers+text",
                    text=["최댓값"],
                    textposition="top center",
                    name="최댓값",
                )
            )
            fig_core = apply_plotly_korean_font(fig_core)
            c2.plotly_chart(fig_core, use_container_width=True)

    st.divider()
    st.subheader("EC별 생육 비교 (2x2)")

    if growth_df.empty:
        st.error("생육 결과 데이터가 없습니다.")
    else:
        g = growth_df.copy()
        g["ec_target"] = pd.to_numeric(g.get("ec_target", pd.Series(dtype=float)), errors="coerce")

        # 그룹 통계
        agg = (
            g.groupby("ec_target", dropna=True)
            .agg(
                평균_생중량=("생중량(g)", "mean"),
                평균_잎수=("잎 수(장)", "mean"),
                평균_지상부=("지상부 길이(mm)", "mean"),
                개체수=("생중량(g)", "size"),
            )
            .reset_index()
            .sort_values("ec_target")
        )

        if agg.empty:
            st.error("EC별 집계에 필요한 데이터가 부족합니다.")
        else:
            fig2 = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("평균 생중량(⭐ 가장 중요)", "평균 잎 수", "평균 지상부 길이", "개체수 비교"),
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )

            x_ec = agg["ec_target"].astype(float).tolist()

            fig2.add_trace(go.Bar(x=x_ec, y=agg["평균_생중량"], name="평균 생중량"), row=1, col=1)
            fig2.add_trace(go.Bar(x=x_ec, y=agg["평균_잎수"], name="평균 잎 수"), row=1, col=2)
            fig2.add_trace(go.Bar(x=x_ec, y=agg["평균_지상부"], name="평균 지상부 길이"), row=2, col=1)
            fig2.add_trace(go.Bar(x=x_ec, y=agg["개체수"], name="개체수"), row=2, col=2)

            fig2.update_layout(
                height=700,
                margin=dict(l=30, r=30, t=70, b=30),
            )
            fig2 = apply_plotly_korean_font(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포")

    if growth_sel.empty:
        st.error("선택한 조건에 해당하는 생육 데이터가 없습니다.")
    else:
        if selected_school == "전체":
            fig_box = px.box(
                growth_sel,
                x="school",
                y="생중량(g)",
                title="학교별 생중량 분포 (Box Plot)",
                points="outliers",
            )
        else:
            fig_box = px.box(
                growth_sel,
                y="생중량(g)",
                title=f"{selected_school} 생중량 분포 (Box Plot)",
                points="outliers",
            )
        fig_box = apply_plotly_korean_font(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.subheader("상관관계 분석 (산점도 2개)")

    if growth_sel.empty:
        st.error("상관관계 분석을 위한 생육 데이터가 없습니다.")
    else:
        c1, c2 = st.columns(2)

        # trendline(OLS)은 statsmodels에 의존하므로 설치 여부를 확인해서 안전하게 사용합니다.
        sm_ok = _statsmodels_available()
        want_trend_sc1 = len(growth_sel.dropna(subset=["잎 수(장)", "생중량(g)"])) >= 5
        if want_trend_sc1 and not sm_ok:
            st.warning("statsmodels가 설치되어 있지 않아 산점도에 추세선(OLS)을 표시할 수 없습니다. 'statsmodels'를 requirements.txt에 추가하고 재배포하세요.")

        fig_sc1 = px.scatter(
            growth_sel,
            x="잎 수(장)",
            y="생중량(g)",
            color="school" if selected_school == "전체" else None,
            title="잎 수 vs 생중량",
            trendline="ols" if (sm_ok and want_trend_sc1) else None,
        )
        fig_sc1 = apply_plotly_korean_font(fig_sc1)
        c1.plotly_chart(fig_sc1, use_container_width=True)

        want_trend_sc2 = len(growth_sel.dropna(subset=["지상부 길이(mm)", "생중량(g)"])) >= 5
        if want_trend_sc2 and not sm_ok:
            st.warning("statsmodels가 설치되어 있지 않아 산점도에 추세선(OLS)을 표시할 수 없습니다. 'statsmodels'를 requirements.txt에 추가하고 재배포하세요.")

        fig_sc2 = px.scatter(
            growth_sel,
            x="지상부 길이(mm)",
            y="생중량(g)",
            color="school" if selected_school == "전체" else None,
            title="지상부 길이 vs 생중량",
            trendline="ols" if (sm_ok and want_trend_sc2) else None,
        )
        fig_sc2 = apply_plotly_korean_font(fig_sc2)
        c2.plotly_chart(fig_sc2, use_container_width=True)

    st.divider()
    with st.expander("📄 학교별 생육 데이터 원본 + XLSX 다운로드"):
        if growth_sel.empty:
            st.error("표시할 생육 데이터가 없습니다.")
        else:
            st.dataframe(growth_sel, use_container_width=True)

            buffer = io.BytesIO()
            # ✅ TypeError 방지: 파일 경로 없이 BytesIO로
            growth_sel.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)

            st.download_button(
                label="생육 데이터 XLSX 다운로드",
                data=buffer,
                file_name="생육데이터_선택.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
