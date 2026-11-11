"""Helpers shared by the main and supplementary figure modules."""

DESIGN_ORDER = ["same-section", "adjacent-section (same mouse, artificial)", "external (another lab)", "registered-cells", "adjacent-section"]
SN, STR = "Substantia Nigra", "Striatum"
MOUSE_OF = {"085_a1": "mPD1", "085_b1": "mPD3", "085_c1": "mPD4", "109-a1": "mPD1", "109-b1": "mPD3", "109-c1": "mPD4"}
DA_IONS = [421.19167, 688.29533, 674.27902, 941.38106]          # dopamine FMP-10 derivatives (m/z)
SYN = ["TH", "DDC", "SLC6A3", "SLC18A2"]                         # dopamine synthesis / transport genes


def _cbar(F, p, x, y, w, h, im, label, ticks=None):
    ca = F.ax(p, x, y, w, h); cb = F.fig.colorbar(im, cax=ca, orientation="horizontal"); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.set_label(label, fontsize=5, labelpad=1)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


def variant_name(v):
    v = str(v)
    if v.startswith("ion:"):
        return f"ion m/z {float(v[4:]):.0f}"
    if v.startswith("gene:"):
        return f"$\\it{{{v[5:][0]}{v[5:][1:].lower()}}}$ only"
    if v.startswith("target"):
        return v.split(":")[-1].replace("MSN_", "MSN ").replace("MSN D", "D") + " targets"
    return {"base": "Base", "shift": "CV shift", "res100": "100 µm", "res400": "400 µm", "detrended": "Detrended"}.get(v, v)


def _mouse(sample):
    for k, v in MOUSE_OF.items():
        if k in sample:
            return v
    return "?"


def _legend(F, p, handles, x, y, ncol=1, **kw):
    lg = F.fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x / F.W, 1 - y / F.H), ncol=ncol, borderaxespad=0, **kw)
    F.text_panel[id(lg)] = (p, "text", lg)
    for t in lg.get_texts():
        F.text_panel[id(t)] = (p, "text", t)
    return lg
