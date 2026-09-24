import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import time

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pedestrian Detection & Perception Suite",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CSS Styling for Modern Dashboard Layout
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top Hero Banner */
    .dashboard-header {
        background: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .dashboard-title {
        color: #0f172a !important;
        font-weight: 800;
        font-size: 2.2rem !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .dashboard-subtitle {
        color: #475569 !important;
        font-size: 1.1rem !important;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Top Stat Cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        color: #2563eb;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* Risk Badges */
    .status-badge-safe {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .status-badge-warn {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    /* Section Cards */
    .content-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Tab Headers */
    button[data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        padding: 0.8rem 1.2rem !important;
    }
    button[aria-selected="true"] {
        color: #2563eb !important;
        border-bottom-color: #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ ADS Control Unit")
st.sidebar.markdown("*Automated Driving System — Perception Module*")

st.sidebar.markdown("---")
default_weights = r"runs/detect/runs/detect/citypersons_fast_full/weights/best.pt"

model_path = st.sidebar.text_input(
    "Weights Checkpoint", 
    value=default_weights,
    help="Relative path to your best.pt model weights file"
)

conf_thresh = st.sidebar.slider("Confidence Cutoff", 0.05, 1.0, 0.25, 0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.45, 0.05)

st.sidebar.info("💡 Bounding Box colors are now dynamically controlled by Proximity Logic (Distance to Vehicle).")

# -----------------------------------------------------------------------------
# 4. Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, str(e)

model, err = load_yolo_model(model_path)

# -----------------------------------------------------------------------------
# 5. Top Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">
        🚗 Autonomous Vehicle Pedestrian Perception Suite
    </div>
    <div class="dashboard-subtitle">
        Real-time Object Detection, Spatial Safety Telemetry, and Proximity Analytics
    </div>
</div>
""", unsafe_allow_html=True)

if err:
    st.error(f"⚠️ Failed to load model weights from `{model_path}`.")
    st.info("💡 Please verify the weights file path in the sidebar.")
else:
    # -----------------------------------------------------------------------------
    # 6. Main Dashboard Content Area
    # -----------------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload Driving Scene Frame (PNG, JPG, JPEG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        h, w, _ = img_np.shape

        # Start Inference Timer
        start_time = time.time()
        results = model.predict(
            source=image, 
            conf=conf_thresh, 
            iou=iou_thresh, 
            device="cpu"
        )[0]
        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        boxes = results.boxes
        telemetry_data = []
        annotated_img = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)

        # Proximity Counters
        critical_count = 0
        warning_count = 0
        safe_count = 0

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            box_w = x2 - x1
            box_h = y2 - y1

            # Proximity Zone Logic based on Bounding Box Height
            if box_h > 250:
                risk_zone = "CRITICAL (< 10m)"
                zone_color = (0, 0, 255) # Red in BGR
                critical_count += 1
            elif box_h > 100:
                risk_zone = "WARNING (10m-25m)"
                zone_color = (0, 165, 255) # Orange in BGR
                warning_count += 1
            else:
                risk_zone = "SAFE (> 25m)"
                zone_color = (0, 255, 0) # Green in BGR
                safe_count += 1

            telemetry_data.append({
                "Detection ID": f"PED-{idx+1:02d}",
                "Confidence": round(conf, 3),
                "Risk Zone": risk_zone,
                "Width (px)": box_w,
                "Height (px)": box_h,
                "Area (px²)": box_w * box_h,
                "Xmin": x1,
                "Ymin": y1,
                "Xmax": x2,
                "Ymax": y2
            })

            # Bounding box rendering using dynamic Zone Color
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), zone_color, 3)
            label = f"PED-{idx+1:02d} | {conf:.2f}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, y1 - text_h - 6), (x1 + text_w, y1), zone_color, -1)
            cv2.putText(
                annotated_img, label, (x1, y1 - 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if risk_zone != "SAFE (> 25m)" else (0, 0, 0), 2, cv2.LINE_AA
            )

        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        df_telemetry = pd.DataFrame(telemetry_data)
        pedestrian_count = len(telemetry_data)

        # -----------------------------------------------------------------------------
        # 7. Top Metrics Cards Grid
        # -----------------------------------------------------------------------------
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Pedestrians</div>
                <div class="stat-value">{pedestrian_count}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            avg_conf = f"{df_telemetry['Confidence'].mean():.2f}" if pedestrian_count > 0 else "N/A"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{avg_conf}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Frame Resolution</div>
                <div class="stat-value">{w}×{h}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            if critical_count > 0:
                status_badge = '<span class="status-badge-warn" style="background:#fee2e2; color:#b91c1c; border-color:#fca5a5;">🚨 CRITICAL HAZARD</span>'
            elif pedestrian_count > 0:
                status_badge = '<span class="status-badge-warn">⚠️ Caution Required</span>'
            else:
                status_badge = '<span class="status-badge-safe">✅ Clear Roadway</span>'
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Collision Status</div>
                <div style="margin-top:0.5rem;">{status_badge}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # -----------------------------------------------------------------------------
        # 8. Proximity & Latency Matrix
        # -----------------------------------------------------------------------------
        st.subheader("🚨 Proximity & Collision Risk Matrix")
        rk_col1, rk_col2, rk_col3, rk_col4 = st.columns(4)
        
        with rk_col1:
            st.markdown(f"""
            <div class="stat-card" style="border-bottom: 4px solid #ef4444;">
                <div class="stat-label">🔴 Critical (&lt; 10m)</div>
                <div class="stat-value" style="color: #ef4444;">{critical_count}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rk_col2:
            st.markdown(f"""
            <div class="stat-card" style="border-bottom: 4px solid #f97316;">
                <div class="stat-label">🟡 Warning (10m - 25m)</div>
                <div class="stat-value" style="color: #f97316;">{warning_count}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rk_col3:
            st.markdown(f"""
            <div class="stat-card" style="border-bottom: 4px solid #22c55e;">
                <div class="stat-label">🟢 Safe (&gt; 25m)</div>
                <div class="stat-value" style="color: #22c55e;">{safe_count}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rk_col4:
            st.markdown(f"""
            <div class="stat-card" style="border-bottom: 4px solid #3b82f6;">
                <div class="stat-label">⏱️ Inference Latency</div>
                <div class="stat-value" style="color: #3b82f6; font-size: 1.8rem;">{inference_time_ms} ms</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------------
        # 9. Main Visual Side-by-Side Comparison Feed
        # -----------------------------------------------------------------------------
        st.subheader("📷 Camera Perception Feed")
        col_orig, col_det = st.columns(2)

        with col_orig:
            st.markdown("**Original Input Frame**")
            st.image(image, use_container_width=True)

        with col_det:
            st.markdown("**Detection & Bounding Box Overlay**")
            st.image(annotated_img_rgb, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------------
        # 10. Tabbed Section Below Imagery
        # -----------------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["📊 Telemetry Analytics", "📋 Detailed Data Table", "⚙️ System Specs"])

        with tab1:
            st.subheader("Spatial & Confidence Analytics")
            if pedestrian_count > 0:
                c1, c2 = st.columns(2)

                with c1:
                    fig_conf = px.bar(
                        df_telemetry, 
                        x="Detection ID", 
                        y="Confidence", 
                        color="Risk Zone",
                        color_discrete_map={
                            "CRITICAL (< 10m)": "#ef4444",
                            "WARNING (10m-25m)": "#f97316",
                            "SAFE (> 25m)": "#22c55e"
                        },
                        title="Confidence Score per Pedestrian Instance",
                        template="plotly_white"
                    )
                    fig_conf.update_layout(
                        paper_bgcolor="#ffffff", 
                        plot_bgcolor="#f8fafc",
                        font=dict(size=13, color="#0f172a")
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

                with c2:
                    fig_size = px.scatter(
                        df_telemetry,
                        x="Width (px)",
                        y="Height (px)",
                        size="Area (px²)",
                        color="Risk Zone",
                        color_discrete_map={
                            "CRITICAL (< 10m)": "#ef4444",
                            "WARNING (10m-25m)": "#f97316",
                            "SAFE (> 25m)": "#22c55e"
                        },
                        hover_name="Detection ID",
                        title="Bounding Box Dimensions (Proximity Mapping)",
                        template="plotly_white"
                    )
                    fig_size.update_layout(
                        paper_bgcolor="#ffffff", 
                        plot_bgcolor="#f8fafc",
                        font=dict(size=13, color="#0f172a")
                    )
                    st.plotly_chart(fig_size, use_container_width=True)
            else:
                st.info("No pedestrian instances detected in this frame.")

        with tab2:
            st.subheader("Bounding Box Bounding Telemetry")
            if pedestrian_count > 0:
                st.dataframe(df_telemetry, use_container_width=True, hide_index=True)
                csv_data = df_telemetry.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Telemetry CSV",
                    data=csv_data,
                    file_name="pedestrian_telemetry.csv",
                    mime="text/csv"
                )
            else:
                st.info("No telemetry records available.")

        with tab3:
            st.subheader("Model & Execution Specifications")
            spec_col1, spec_col2 = st.columns(2)
            
            with spec_col1:
                st.markdown("""
                **Model Architecture Specs:**
                * **Base Model:** YOLO11 Nano (`yolo11n.pt`)
                * **Task:** 2D Bounding Box Pedestrian Detection
                * **Training Dataset:** CityPersons / Cityscapes
                * **Target Class:** Pedestrian / Person (`class 0`)
                """)

            with spec_col2:
                st.markdown("""
                **Inference Environment:**
                * **Execution Device:** CPU Mode (`device="cpu"`)
                * **Input Resolution:** Scaled dynamically to inference model dims
                * **NMS Post-Processing:** IoU threshold adjustable
                * **Proximity Logic:** Heuristic mapping of Bounding Box Height to distance thresholds.
                """)
    else:
        st.info("📌 Please upload an image from CityPersons or Cityscapes to initiate the perception pipeline.")