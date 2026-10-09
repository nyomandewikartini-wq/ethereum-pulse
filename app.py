# -*- coding: utf-8 -*-
"""
Ethereum Pulse: LSTM Forecast + Complex Event Processing (CEP)

LSTM:
- Menggunakan model harian yang dilatih di notebook preprocessing/EDA/modeling.
- Model dan scaler harus disimpan ke folder models/ sebelum deployment.
- Prediksi adalah estimasi harga penutupan harian berikutnya, bukan prediksi harga per menit.

CEP:
- Memproses observasi intraday ETH-USD dari Yahoo Finance (interval 1 menit).
- Mendeteksi threshold + durasi, trend, sequence harga-volume, dan absence feed.
- Polling bukan jaminan feed real-time tick-by-tick.
"""

from collections import deque
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model

# ------------------------- Page setup -------------------------
st.set_page_config(
    page_title="Ethereum Pulse | LSTM + CEP",
    page_icon="🧩",
    layout="wide",
)
st.title("🧩 Ethereum Pulse")
st.caption(
    "Monitoring intraday ETH-USD dengan CEP dan estimasi penutupan harian "
    "berbasis LSTM. CEP memakai data intraday 1 menit; LSTM memakai data harian "
    "sesuai frekuensi saat pelatihan."
)

# ------------------------- Paths and model artifacts -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "eth_lstm_best.keras"
SCALER_PATH = MODEL_DIR / "eth_close_scaler.joblib"
CONFIG_PATH = MODEL_DIR / "eth_lstm_config.json"

# ------------------------- Sidebar controls -------------------------
st.sidebar.header("⚙️ Pengaturan")
TICKER = st.sidebar.text_input("Ticker", value="ETH-USD").strip().upper()
WINDOW_SIZE = int(st.sidebar.number_input(
    "Window rolling CEP (event)", min_value=5, max_value=100, value=20
))
MAX_BUFFER = int(st.sidebar.number_input(
    "Ukuran buffer intraday", min_value=30, max_value=1000, value=200
))
POLL_SECONDS = int(st.sidebar.number_input(
    "Interval polling (detik)", min_value=15, max_value=300, value=60
))
Z_THRESH = float(st.sidebar.slider(
    "Ambang Z-score", min_value=1.0, max_value=4.0, value=2.0, step=0.1
))
MIN_CONSECUTIVE = int(st.sidebar.number_input(
    "Threshold: event beruntun minimum", min_value=2, max_value=20, value=3
))
TREND_LEN = int(st.sidebar.number_input(
    "Trend: perubahan rolling mean konsisten", min_value=2, max_value=20, value=3
))
SEQ_WINDOW = int(st.sidebar.number_input(
    "Sequence: window harga → volume", min_value=2, max_value=30, value=5
))
ABSENCE_TOLERANCE = float(st.sidebar.slider(
    "Absence: toleransi × interval polling",
    min_value=1.5, max_value=10.0, value=3.0, step=0.5
))
is_running = st.sidebar.toggle("Auto-refresh", value=True)

if st.sidebar.button("Reset buffer"):
    st.session_state.pop("eth_stream_buffer", None)
    st.session_state.pop("eth_last_timestamp", None)
    st.rerun()

# ------------------------- Data fetching -------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_intraday(ticker: str) -> pd.DataFrame:
    """Fetch recent 1-minute bars; Yahoo Finance availability/rate limits apply."""
    raw = yf.download(
        tickers=ticker,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    required = {"Close"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    out["price"] = pd.to_numeric(raw["Close"], errors="coerce")
    out["volume"] = (
        pd.to_numeric(raw["Volume"], errors="coerce")
        if "Volume" in raw.columns else np.nan
    )
    out.index.name = "timestamp"
    out = out.dropna(subset=["price"]).reset_index()
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return out.tail(1000).reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_close(ticker: str) -> pd.Series:
    """Fetch daily Close history used as the LSTM input frequency."""
    raw = yf.download(
        tickers=ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" not in raw.columns:
        return pd.Series(dtype=float)

    close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.astype(float)


# ------------------------- CEP helper functions -------------------------
def consecutive_true_count(mask: pd.Series) -> pd.Series:
    counts = np.zeros(len(mask), dtype=int)
    running = 0
    for i, value in enumerate(mask.fillna(False).to_numpy()):
        running = running + 1 if value else 0
        counts[i] = running
    return pd.Series(counts, index=mask.index)


def consecutive_same_sign(diff: pd.Series) -> pd.Series:
    signs = np.sign(diff.fillna(0)).to_numpy()
    counts = np.zeros(len(signs), dtype=int)
    running, previous = 0, 0
    for i, sign in enumerate(signs):
        if sign != 0 and sign == previous:
            running += 1
        elif sign != 0:
            running = 1
        else:
            running = 0
        counts[i] = running
        previous = sign
    return pd.Series(counts, index=diff.index)


def analyze_buffer(buffer_data: deque) -> pd.DataFrame:
    df = pd.DataFrame(list(buffer_data))
    if df.empty:
        return df

    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True)
    )
    df["rolling_mean"] = df["price"].rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).mean()
    df["rolling_sd"] = df["price"].rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).std()
    df["upper_bound"] = df["rolling_mean"] + 2 * df["rolling_sd"]
    df["lower_bound"] = df["rolling_mean"] - 2 * df["rolling_sd"]

    prior_mean = df["price"].shift(1).rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).mean()
    prior_sd = df["price"].shift(1).rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).std()
    df["z_score"] = (df["price"] - prior_mean) / prior_sd.replace(0, np.nan)
    df["anomaly"] = df["z_score"].abs() > Z_THRESH

    vol_prior_mean = df["volume"].shift(1).rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).mean()
    vol_prior_sd = df["volume"].shift(1).rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).std()
    df["vol_z_score"] = (
        (df["volume"] - vol_prior_mean) / vol_prior_sd.replace(0, np.nan)
    )
    df["vol_anomaly"] = df["vol_z_score"].abs() > Z_THRESH

    df["streak_upper"] = consecutive_true_count(df["price"] > df["upper_bound"])
    df["streak_lower"] = consecutive_true_count(df["price"] < df["lower_bound"])
    df["threshold_pattern"] = (
        (df["streak_upper"] >= MIN_CONSECUTIVE)
        | (df["streak_lower"] >= MIN_CONSECUTIVE)
    )
    df["threshold_side"] = np.select(
        [
            df["streak_upper"] >= MIN_CONSECUTIVE,
            df["streak_lower"] >= MIN_CONSECUTIVE,
        ],
        ["atas (UCL)", "bawah (LCL)"],
        default="-",
    )

    mean_diff = df["rolling_mean"].diff()
    df["trend_streak"] = consecutive_same_sign(mean_diff)
    df["trend_pattern"] = df["trend_streak"] >= TREND_LEN
    df["trend_direction"] = np.select(
        [mean_diff > 0, mean_diff < 0], ["naik", "turun"], default="-"
    )

    prior_price_anomaly = (
        df["anomaly"].shift(1).rolling(SEQ_WINDOW, min_periods=1)
        .max().fillna(0).astype(bool)
    )
    df["sequence_pattern"] = df["vol_anomaly"].fillna(False) & prior_price_anomaly
    return df


def build_event_log(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["timestamp", "pola", "deskripsi", "harga"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    events = []
    for _, row in df.iterrows():
        if bool(row.get("threshold_pattern", False)):
            streak = int(max(row["streak_upper"], row["streak_lower"]))
            events.append({
                "timestamp": row["timestamp"],
                "pola": "Threshold + Durasi",
                "deskripsi": f"Harga di {row['threshold_side']} batas selama {streak} event beruntun",
                "harga": row["price"],
            })
        if bool(row.get("trend_pattern", False)):
            events.append({
                "timestamp": row["timestamp"],
                "pola": "Trend",
                "deskripsi": f"Rolling mean {row['trend_direction']} konsisten {int(row['trend_streak'])} event",
                "harga": row["price"],
            })
        if bool(row.get("sequence_pattern", False)):
            events.append({
                "timestamp": row["timestamp"],
                "pola": "Sequence + Correlation",
                "deskripsi": f"Anomali volume mengikuti anomali harga dalam {SEQ_WINDOW} event",
                "harga": row["price"],
            })

    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(events).sort_values("timestamp", ascending=False).head(50).reset_index(drop=True)


def absence_status(last_timestamp):
    if last_timestamp is None:
        return True, None
    ts = pd.Timestamp(last_timestamp)
    now = pd.Timestamp.now(tz=ts.tz) if ts.tz is not None else pd.Timestamp.now()
    gap = max(0.0, (now - ts).total_seconds())
    return gap > POLL_SECONDS * ABSENCE_TOLERANCE, gap


# ------------------------- LSTM artifacts and forecast -------------------------
@st.cache_resource
def load_lstm_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None, None, (
            "Model/scaler belum ditemukan. Letakkan eth_lstm_best.keras, "
            "eth_close_scaler.joblib, dan eth_lstm_config.json di folder models/."
        )
    try:
        model = load_model(MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
        return model, scaler, config, None
    except Exception as exc:
        return None, None, None, f"Gagal memuat model/scaler: {exc}"


# ------------------------- Get and update stream buffer -------------------------
with st.spinner("Mengambil data intraday ETH-USD..."):
    intraday = fetch_intraday(TICKER)

if "eth_stream_buffer" not in st.session_state:
    st.session_state.eth_stream_buffer = deque(maxlen=MAX_BUFFER)

if not intraday.empty:
    for record in intraday.to_dict("records"):
        record["timestamp"] = pd.Timestamp(record["timestamp"])
        record["price"] = float(record["price"])
        record["volume"] = float(record["volume"]) if pd.notna(record["volume"]) else np.nan
        # Seed the buffer from recent bars once, then append only unseen timestamps.
        if not any(
            pd.Timestamp(existing["timestamp"]) == record["timestamp"]
            for existing in st.session_state.eth_stream_buffer
        ):
            st.session_state.eth_stream_buffer.append(record)

if len(st.session_state.eth_stream_buffer) > MAX_BUFFER:
    st.session_state.eth_stream_buffer = deque(
        list(st.session_state.eth_stream_buffer)[-MAX_BUFFER:],
        maxlen=MAX_BUFFER,
    )

df_analyzed = analyze_buffer(st.session_state.eth_stream_buffer)

# ------------------------- Dashboard: stream status -------------------------
if intraday.empty:
    st.warning(
        "Data intraday belum tersedia dari Yahoo Finance. Coba refresh beberapa saat lagi. "
        "Di luar jam/ketentuan data intraday atau saat rate limit, data dapat kosong."
    )
else:
    latest = df_analyzed.iloc[-1]
    st.caption(f"Observasi intraday terbaru yang diterima: {latest['timestamp']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Harga ETH terakhir", f"${latest['price']:,.2f}")
    c2.metric(
        f"Rolling mean ({WINDOW_SIZE} event)",
        f"${latest['rolling_mean']:,.2f}" if pd.notna(latest["rolling_mean"]) else "Menunggu data",
    )
    c3.metric("Z-score harga", f"{latest['z_score']:.2f}" if pd.notna(latest["z_score"]) else "—")
    c4.metric("Z-score volume", f"{latest['vol_z_score']:.2f}" if pd.notna(latest["vol_z_score"]) else "—")

    st.subheader("Status pola CEP")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.write("**Threshold + Durasi**")
        if bool(latest["threshold_pattern"]):
            st.error(f"Breach {latest['threshold_side']} — {int(max(latest['streak_upper'], latest['streak_lower']))} event")
        else:
            st.success("Tidak terdeteksi")
    with p2:
        st.write("**Trend**")
        if bool(latest["trend_pattern"]):
            st.warning(f"Rolling mean {latest['trend_direction']} konsisten")
        else:
            st.success("Tidak terdeteksi")
    with p3:
        st.write("**Sequence harga → volume**")
        if bool(latest["sequence_pattern"]):
            st.error("Anomali volume menyusul anomali harga")
        else:
            st.success("Tidak terdeteksi")
    with p4:
        st.write("**Absence feed**")
        absent, gap_seconds = absence_status(latest["timestamp"])
        if absent:
            st.error(f"Tidak ada event baru selama {gap_seconds:,.0f} detik")
        else:
            st.success(f"Timestamp terakhir berjarak {gap_seconds:,.0f} detik")

    # Intraday CEP chart
    st.subheader("Stream intraday dan pola CEP")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.70, 0.30], vertical_spacing=0.07,
        subplot_titles=("Harga ETH dan batas rolling", "Volume"),
    )
    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["price"],
                             mode="lines", name="Harga", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["rolling_mean"],
                             mode="lines", name="Rolling mean"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["upper_bound"],
                             mode="lines", name="Upper bound", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["lower_bound"],
                             mode="lines", name="Lower bound", line=dict(dash="dash")), row=1, col=1)

    anomalies = df_analyzed[df_analyzed["anomaly"].fillna(False)]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["price"],
                                 mode="markers", name="Anomali Z-score",
                                 marker=dict(size=8, symbol="x")), row=1, col=1)
    threshold_events = df_analyzed[df_analyzed["threshold_pattern"].fillna(False)]
    if not threshold_events.empty:
        fig.add_trace(go.Scatter(x=threshold_events["timestamp"], y=threshold_events["price"],
                                 mode="markers", name="Threshold + durasi",
                                 marker=dict(size=10, symbol="diamond")), row=1, col=1)
    trend_events = df_analyzed[df_analyzed["trend_pattern"].fillna(False)]
    if not trend_events.empty:
        fig.add_trace(go.Scatter(x=trend_events["timestamp"], y=trend_events["price"],
                                 mode="markers", name="Pola trend",
                                 marker=dict(size=8, symbol="triangle-up")), row=1, col=1)
    seq_events = df_analyzed[df_analyzed["sequence_pattern"].fillna(False)]
    if not seq_events.empty:
        fig.add_trace(go.Scatter(x=seq_events["timestamp"], y=seq_events["price"],
                                 mode="markers", name="Sequence CEP",
                                 marker=dict(size=10, symbol="star")), row=1, col=1)

    fig.add_trace(go.Bar(x=df_analyzed["timestamp"], y=df_analyzed["volume"], name="Volume"), row=2, col=1)
    fig.update_layout(template="plotly_white", height=650, margin=dict(l=15, r=15, t=60, b=15),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_yaxes(title_text="Harga (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Waktu", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Log event CEP")
    event_log = build_event_log(df_analyzed)
    if event_log.empty:
        st.info("Belum ada pola CEP terdeteksi pada buffer saat ini.")
    else:
        st.dataframe(event_log.round({"harga": 2}), use_container_width=True, hide_index=True)

    with st.expander("Data intraday terakhir"):
        st.dataframe(df_analyzed.tail(20), use_container_width=True, hide_index=True)

# ------------------------- LSTM daily forecast -------------------------
st.divider()
st.subheader("Peramalan penutupan harian dengan LSTM")
st.caption(
    "Model ini memakai urutan harga Close harian seperti saat pelatihan. "
    "Hasilnya bukan prediksi harga pada menit berikutnya."
)

model, scaler, config, model_error = load_lstm_artifacts()
if model_error:
    st.warning(model_error)
    st.info(
        "Dari notebook, simpan model terbaik 14 hari/128 unit dan scaler training "
        "ke folder models/. Jangan menggunakan model 32 unit dari eksperimen awal."
    )
else:
    with st.spinner("Mengambil histori harga harian untuk input LSTM..."):
        daily_close = fetch_daily_close(TICKER)

    lookback = int(config.get("lookback", 14))
    expected_units = int(config.get("lstm_units", 128))

    if daily_close.empty or len(daily_close) < lookback:
        st.error("Data Close harian tidak cukup untuk membentuk input LSTM.")
    else:
        recent = daily_close.tail(lookback).to_numpy(dtype=float).reshape(-1, 1)
        scaled_recent = scaler.transform(recent)
        lstm_input = scaled_recent.reshape(1, lookback, 1)
        forecast_scaled = model.predict(lstm_input, verbose=0).reshape(-1, 1)
        forecast_price = float(scaler.inverse_transform(forecast_scaled)[0, 0])

        last_daily_price = float(daily_close.iloc[-1])
        last_daily_date = daily_close.index[-1]
        delta = forecast_price - last_daily_price
        pct = (delta / last_daily_price * 100) if last_daily_price else np.nan

        f1, f2, f3 = st.columns(3)
        f1.metric("Close harian terakhir", f"${last_daily_price:,.2f}", help=str(last_daily_date))
        f2.metric("Estimasi Close berikutnya", f"${forecast_price:,.2f}", f"{delta:+,.2f} USD")
        f3.metric("Perubahan indikatif", f"{pct:+.2f}%")

        st.caption(
            f"Model terpasang: lookback {lookback} hari, {expected_units} unit LSTM. "
            "Ini adalah estimasi satu langkah berdasarkan histori Close harian yang tersedia; "
            "bukan jaminan arah pasar."
        )

        daily_plot = go.Figure()
        daily_plot.add_trace(go.Scatter(
            x=daily_close.index[-90:], y=daily_close.iloc[-90:],
            mode="lines", name="Close harian historis"
        ))
        forecast_date = daily_close.index[-1] + pd.Timedelta(days=1)
        daily_plot.add_trace(go.Scatter(
            x=[last_daily_date, forecast_date],
            y=[last_daily_price, forecast_price],
            mode="lines+markers", name="Estimasi langkah berikutnya",
            line=dict(dash="dash")
        ))
        daily_plot.update_layout(
            template="plotly_white", height=400,
            title="Close harian dan estimasi satu langkah ke depan",
            xaxis_title="Tanggal", yaxis_title="Harga (USD)"
        )
        st.plotly_chart(daily_plot, use_container_width=True)

st.caption(
    "Catatan: Yahoo Finance bukan feed transaksi tick-by-tick. Data intraday dapat tertunda, "
    "dibatasi, atau kosong; hasil CEP dan LSTM bersifat analitis, bukan saran investasi."
)

# ------------------------- Polling refresh -------------------------
if is_running:
    time.sleep(POLL_SECONDS)
    st.rerun()
