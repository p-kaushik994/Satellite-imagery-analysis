import io, numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from PIL import Image
import core
from core import CLASSES

st.set_page_config(page_title="Satellite Analyzer & Mission Planner", layout="wide")


@st.cache_resource
def get_model():
    return core.load_model("eurosat_resnet18.pth")


@st.cache_data(show_spinner=False)
def analyse_bytes(b):
    """Cached per uploaded file: class probabilities + cloud estimate."""
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return core.classify(get_model(), img), core.est_cloud(img)


try:
    model = get_model()
except FileNotFoundError:
    st.error("eurosat_resnet18.pth not found. Run `python train.py` first (or copy the weights next to app.py).")
    st.stop()

st.title("🛰️ Satellite Image Analyzer & Mission Planner")
tab1, tab2, tab3 = st.tabs(["1 · Analyze image", "2 · Change detection", "3 · Mission planner"])

# ------------------------------------------------------------------ tab 1
with tab1:
    f = st.file_uploader("Upload a satellite tile (EuroSAT-style, ~64x64 px works best)",
                         type=["png", "jpg", "jpeg"], key="one")
    if f:
        img = Image.open(f).convert("RGB")
        p = core.classify(model, img)
        cam, c, prob = core.gradcam(model, img)
        a, b = st.columns(2)
        a.image(img.resize((256, 256)), caption="Input")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img.resize((128, 128))); ax.imshow(cam, cmap="jet", alpha=0.45); ax.axis("off")
        b.pyplot(fig); b.caption("Grad-CAM: red = pixels that pushed the model toward its prediction")
        st.success(f"Prediction: **{CLASSES[c]}** ({prob:.1%})")
        st.bar_chart(pd.Series(p, index=CLASSES))
        st.caption(f"Estimated cloud fraction (brightness heuristic): {core.est_cloud(img):.0%}")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Upload two images of the **same area** from different dates. "
             "Two methods: pixel difference, and per-cell land-cover classification.")
    c1, c2 = st.columns(2)
    fa = c1.file_uploader("Before", type=["png", "jpg", "jpeg"], key="before")
    fb = c2.file_uploader("After", type=["png", "jpg", "jpeg"], key="after")
    if fa and fb:
        ia, ib = Image.open(fa).convert("RGB"), Image.open(fb).convert("RGB")
        cells = st.slider("Grid size (cells per side)", 2, 8, 4)
        thr = st.slider("Pixel-change threshold", 0.1, 0.9, 0.5, 0.05)

        d = core.pixel_change(ia, ib)
        mask = d > thr
        pa, pb = core.grid_classes(model, ia, cells), core.grid_classes(model, ib, cells)
        ca, cb = pa.argmax(1), pb.argmax(1)
        changed = ca != cb

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        axes[0].imshow(ia.resize((256, 256))); axes[0].set_title("Before")
        axes[1].imshow(ib.resize((256, 256))); axes[1].set_title("After")
        axes[2].imshow(ib.resize((256, 256)))
        axes[2].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.6)
        axes[2].set_title(f"Pixel change ({mask.mean():.0%} of area)")
        axes[3].imshow(ib.resize((256, 256)))
        s = 256 / cells
        for k in range(cells * cells):
            r, cc = divmod(k, cells)
            if changed[k]:
                axes[3].add_patch(plt.Rectangle((cc * s, r * s), s, s, fill=False, ec="red", lw=2))
        axes[3].set_title(f"Land-cover change ({changed.sum()}/{cells*cells} cells)")
        for a in axes: a.axis("off")
        st.pyplot(fig)

        m1, m2 = st.columns(2)
        m1.metric("Pixels changed", f"{mask.mean():.1%}")
        m2.metric("Cells with class change", f"{changed.sum()} / {cells*cells}")
        if changed.any():
            rows = [dict(cell=f"r{k//cells}c{k%cells}", before=CLASSES[ca[k]], after=CLASSES[cb[k]],
                         conf_before=round(float(pa[k].max()), 2), conf_after=round(float(pb[k].max()), 2))
                    for k in np.where(changed)[0]]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            tr = pd.Series([f"{r['before']} → {r['after']}" for r in rows]).value_counts()
            st.bar_chart(tr)
        st.caption("Pixel difference catches any visual change (including lighting); the grid "
                   "classifier reports only changes in land-cover class. Use both.")

# ------------------------------------------------------------------ tab 3
with tab3:
    with st.expander("Mission settings", expanded=True):
        a, b, c, d_ = st.columns(4)
        budget = a.slider("Time budget (min)", 5, 40, 12)
        img_min = b.slider("Imaging time / target (min)", 0.5, 3.0, 1.0)
        slew = c.slider("Re-pointing speed (km/min)", 50, 300, 100)
        n_sim = d_.slider("Simulated extra targets", 0, 60, 30)
    with st.expander("Class priorities (mission goal)"):
        cols = st.columns(5)
        pri = {k: cols[i % 5].slider(k, 0.0, 1.0, v, 0.1) for i, (k, v) in enumerate(core.DEFAULT_PRIORITY.items())}

    files = st.file_uploader("Upload tiles as real targets. Name them like River_19.5_74.2.png to give a real location.",
                             type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="many")
    rng = np.random.default_rng(42)
    rows = []
    for f in files:
        p, cloud = analyse_bytes(f.getvalue())
        co = core.coords_from_name(f.name)
        lat, lon = co if co else (rng.uniform(15, 21), rng.uniform(72, 78))
        rows.append(dict(name=f.name, cls=CLASSES[p.argmax()], conf=float(p.max()), cloud=cloud,
                         lat=lat, lon=lon, src="uploaded, real location" if co else "uploaded, simulated location"))
    for k in range(n_sim):
        rows.append(dict(name=f"sim_{k}", cls=str(rng.choice(CLASSES)), conf=float(rng.beta(8, 1.5)),
                         cloud=float(rng.beta(2, 4)), lat=rng.uniform(15, 21), lon=rng.uniform(72, 78),
                         src="simulated"))
    if not rows:
        st.info("Add some simulated targets or upload tiles.")
    else:
        T = pd.DataFrame(rows)
        T["value"] = T.cls.map(pri) * T.conf * (1 - T.cloud)
        kw = dict(budget=budget, img_min=img_min, slew=slew)
        route, gv, gt = core.plan(T, "greedy", **kw)
        rv = core.random_baseline(T, 100, **kw)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Targets imaged", len(route)); m2.metric("Time used", f"{gt:.1f} / {budget} min")
        m3.metric("Greedy value", f"{gv:.2f}")
        m4.metric("vs random plan", f"{100*(gv/max(rv,1e-9)-1):+.0f}%")

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(T.lon, T.lat, s=T.value * 300 + 10, c=T.cloud, cmap="Blues_r", edgecolor="gray", alpha=.7)
        path = [core.START] + [(T.lat[i], T.lon[i]) for i in route]
        ax.plot([q[1] for q in path], [q[0] for q in path], "r-o", ms=4)
        for n, i in enumerate(route, 1):
            ax.text(T.lon[i], T.lat[i], f" {n}:{T.cls[i]}", fontsize=7)
        plt.colorbar(sc, label="Cloud cover (estimate)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Planned imaging sequence (size = value)")
        st.pyplot(fig)

        st.dataframe(T.loc[route, ["name", "cls", "conf", "cloud", "value", "src"]].round(2), hide_index=True)

        if st.checkbox("Show value vs time budget (greedy vs random)"):
            bs = list(range(5, 41, 5))
            sweep = pd.DataFrame({"greedy": [core.plan(T, "greedy", budget=x, img_min=img_min, slew=slew)[1] for x in bs],
                                  "random": [core.random_baseline(T, 50, budget=x, img_min=img_min, slew=slew) for x in bs]},
                                 index=bs)
            st.line_chart(sweep)
        st.caption("Real: classification, cloud estimate (brightness heuristic), planner. "
                   "Simulated: locations unless in the filename, and all 'simulated' rows.")
