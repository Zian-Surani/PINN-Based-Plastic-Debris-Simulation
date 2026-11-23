"""Interactive front end for the same versioned experiment pipeline as CLI."""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import json
from pathlib import Path
from config import TERRAIN, SEASON_DISCHARGE, make_experiment
from core.pinn_inference import load_pinn
from simulation.validation_suite import evaluate
from simulation.uncertainty_quantification import residual_interval_audit
from core.sensitivity import retention_sensitivity

st.set_page_config(page_title="Tracer transport verification", layout="wide")
st.title("River tracer transport — numerical verification")
st.write(
    "Synthetic depth-integrated advection–diffusion scenarios. Terrain and season choose documented hydraulic assumptions, not measured conditions at a real river."
)
terrain = st.sidebar.selectbox("Terrain", list(TERRAIN))
season = st.sidebar.selectbox("Season", list(SEASON_DISCHARGE), index=1)
exp = make_experiment(terrain, season)
st.sidebar.write(
    f"Domain: {exp.length:g} × {exp.width:g} m; {exp.nx} × {exp.ny} cells; horizon {exp.horizon:g} s"
)
st.sidebar.write(f"Velocity: {exp.u:.4f} m/s; dispersion: {exp.diffusion:.4f} m²/s")
st.caption(
    "The hybrid PINN uses numerical supervision and physics losses. Nonnegative output, a clean-water inlet and wall/outlet derivative conditions are structural; conservation and approximation accuracy are measured, not guaranteed by the network."
)
if st.sidebar.button("Run verification"):
    try:
        torch.set_num_threads(2)
        with st.spinner("Comparing against an independently computed finer grid..."):
            model = load_pinn()
            result = evaluate(model, exp)
        st.session_state["result"] = (exp, result)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
if "result" in st.session_state:
    shown, r = st.session_state["result"]
    m = r["metrics"]
    st.subheader(f"{shown.terrain} / {shown.season} — {shown.horizon:g} s")
    st.dataframe(pd.DataFrame([m]).drop(columns=["acceptance"]))
    st.write("Numerical acceptance checks", m["acceptance"])
    if not m["accepted_numerically"]:
        st.warning(
            "One or more numerical acceptance checks failed. Do not treat this scenario as a validated prediction."
        )
    x, y = shown.grid()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3))
    for ax, field, title in zip(
        axs,
        [
            r["prediction"][-1],
            r["reference"][-1],
            r["prediction"][-1] - r["reference"][-1],
        ],
        ["Hybrid PINN", "Finer-grid reference", "Signed error"],
    ):
        im = ax.pcolormesh(x, y, field.T, shading="auto")
        ax.set(xlabel="x (m)", ylabel="y (m)", title=title)
        fig.colorbar(im, ax=ax, label="kg/m²")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(shown.times(), r["mass"]["mass_kg"], label="Tracer remaining")
    ax.plot(
        shown.times(),
        r["mass"]["integrated_outward_flux_kg"],
        label="Net tracer exported",
    )
    ax.set(xlabel="Time (s)", ylabel="Mass (kg)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    interval = residual_interval_audit(r["prediction"], r["reference"])
    st.write(
        f"Descriptive residual-interval coverage: {interval['empirical_coverage']:.1%} at nominal {interval['nominal_level']:.0%}."
    )
    st.caption(interval["interpretation"])
    if not interval["nominal_coverage_met_on_evaluation"]:
        st.warning(
            "Residual intervals under-cover on this numerical trajectory. They are not reliable 90% prediction bounds."
        )
st.info(
    "This is a numerical verification tool. No field calibration or operational river forecast is claimed. Archived version-1 checkpoints and manuscript tables are incompatible with this pipeline."
)

st.subheader("Discharge sensitivity of tracer retention")
st.write("Explore how the fraction of tracer remaining after 300 seconds changes with discharge in the synthetic plain-terrain case. This demonstration holds the initial pulse, width, roughness and slope fixed; velocity and dispersion follow the documented Manning relationship.")
multiplier=st.slider("Discharge / baseline discharge",.85,1.15,1.0,.01)
if st.button("Calculate retention and its discharge derivative"):
    try:
        torch.set_num_threads(2)
        sensitivity=retention_sensitivity(load_pinn(),multiplier)
        st.write(f"Discharge: {sensitivity['discharge_m3_s']:.0f} m³/s")
        st.write(f"Predicted retained mass: {sensitivity['retained_mass_fraction']:.2%}")
        st.write(f"Derivative of retained fraction: {sensitivity['derivative_per_m3_s']:.6g} per (m³/s)")
        st.caption(sensitivity['scope'])
        st.download_button("Download sensitivity result",json.dumps(sensitivity,indent=2),file_name="retention_sensitivity.json",mime="application/json")
    except (ValueError,FileNotFoundError,RuntimeError) as exc:st.error(str(exc))
audit_path=Path(__file__).resolve().parent/'outputs/reviewer_strengthening/reference_accuracy.json'
if audit_path.exists():
    audit=json.loads(audit_path.read_text())
    st.warning(f"Recorded baseline reference audit: the original upwind FVM reference understates approximation error. Against the independently refined reference, baseline neural cell-average space–time L2 error is {audit['neural_gauss_averages_vs_refined_spacetime']:.2%}. The verification button above still reports agreement with the original upwind reference; these are different comparisons.")
