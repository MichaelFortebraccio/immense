from __future__ import annotations
import os
import pandas as pd
from collections import Counter

# ---------------------------------------------------------------------
# STATIC PATHS 
# ---------------------------------------------------------------------
PROJECT_ROOT = "/Users/mike/Desktop/Tirocinio-tesi/Progetto"
IMMENSE_DIR  = f"{PROJECT_ROOT}/immense"
DATASET_DIR  = f"{PROJECT_ROOT}/small_dataset"
SHAP_DIR     = f"{IMMENSE_DIR}/shap"

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def _safe_read_csv(path, **kwargs):
    """Read CSV or raise an explicit FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[explanations] File non trovato: {path}")
    return pd.read_csv(path, **kwargs)

def _clean_and_tokenize(text: str) -> list[str]:
    """Basic text cleaning + tokenization (lowercase, strip urls/mentions/hashtags, keep alnum>=3)."""
    import re
    if not isinstance(text, str):
        return []
    t = text.lower()
    t = re.sub(r"http\S+|@\w+|#\w+|[^a-zà-ù0-9\s]", " ", t)
    return [w for w in t.split() if len(w) > 2]

def _latest_shap_csv(shap_dir: str) -> str | None:
    """Pick the latest SHAP CSV, preferring those containing 'with_predictions'."""
    best = None
    for root, _dirs, files in os.walk(shap_dir):
        for f in files:
            if f.endswith(".csv") and "shap_output" in f:
                path = os.path.join(root, f)
                score = 2 if "with_predictions" in f else 1
                mtime = os.path.getmtime(path)
                if best is None or (score, mtime) > (best[0], best[1]):
                    best = (score, mtime, path)
    return None if best is None else best[2]

def _read_edges_edg(path: str) -> pd.DataFrame | None:
    """Read .edg edge list: social (2 cols) or spatial (3 cols with optional weight)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=None, sep=None, engine="python")
    if df.shape[1] == 2:
        df.columns = ["src", "dst"]
    elif df.shape[1] >= 3:
        df = df.iloc[:, :3]
        df.columns = ["src", "dst", "w"]
    else:
        raise ValueError(f"[explanations] Edge list non valida: {path}")
    return df

def _map_numeric_label_to_str(x) -> str:
    """Map numeric predicted label 0/1 to 'safe'/'risky' (fallback to 'unknown')."""
    try:
        xi = int(x)
        return "risky" if xi == 1 else "safe"
    except Exception:
        s = str(x).strip().lower()
        if s in {"1", "risky"}: return "risky"
        if s in {"0", "safe"}:  return "safe"
        return "unknown"

# ---------------------------------------------------------------------
# NEGATIVE CONTENT LEXICON
# ---------------------------------------------------------------------
def _load_negative_lexicon(params: dict) -> dict[str, float] | None:
    """
    Build a token -> weight dictionary from the negative content CSV.
    Expected columns: 'text' (free-form snippet) and optional 'level' ('3 - High', '2 - Mid', '1 - Low').
    If the file is missing, returns None (feature disabled).
    """
    cfg = params.get("explanations", {})
    path = cfg.get("negative_content_csv", os.path.join(PROJECT_ROOT, "negative content.csv"))
    if not os.path.exists(path):
        return None

    df = _safe_read_csv(path)

    # Map level labels to numeric weights (customizable)
    level_weights = {
        "3 - High": 3.0, "High": 3.0,
        "2 - Mid":  2.0, "Mid":  2.0,
        "1 - Low":  1.0, "Low":  1.0,
    }

    lex = Counter()
    if "text" not in df.columns:
        raise KeyError("[explanations] Il CSV di contenuti negativi deve avere la colonna 'text'.")

    for _, row in df.iterrows():
        txt = row.get("text", "")
        toks = _clean_and_tokenize(txt)
        base_w = 1.0
        if "level" in df.columns:
            lvl = str(row.get("level", "")).strip()
            base_w = level_weights.get(lvl, 1.0)
        # Update weight per token (sum across rows if token repeats)
        for t in set(toks):
            lex[t] += base_w

    # Optional filtering of very weak tokens
    min_w = float(cfg.get("negative_min_weight", 1.0))
    lexicon = {t: float(w) for t, w in lex.items() if w >= min_w}
    return lexicon if lexicon else None

# ---------------------------------------------------------------------
# MAIN ARTIFACT LOADER
# ---------------------------------------------------------------------
def _load_artifacts(params: dict) -> dict:
    #SHAP CSV (prefer 'with_predictions')
    shap_csv = _latest_shap_csv(SHAP_DIR)
    if shap_csv is None:
        raise FileNotFoundError(f"[explanations] Nessun CSV SHAP trovato in: {SHAP_DIR}")
    shap_df = _safe_read_csv(shap_csv)

    #Minimal dataframe: id + predicted label as string + AE + SHAP components
    must_cols = ["user_id", "predicted_label", "AE_safe_err", "AE_risk_err",
                 "SHAP_content", "SHAP_rel", "SHAP_spat"]
    for c in must_cols:
        if c not in shap_df.columns:
            raise KeyError(f"[explanations] Colonna mancante nello SHAP: {c}")
    test_df = shap_df[must_cols].drop_duplicates().copy()
    test_df["pred_label"] = test_df["predicted_label"].map(_map_numeric_label_to_str)

    #Texts: id -> user_id, text_cleaned -> text (list of posts per user)
    texts = _safe_read_csv(os.path.join(DATASET_DIR, "test.csv"))
    if "id" not in texts.columns or "text_cleaned" not in texts.columns:
        raise KeyError("[explanations] test.csv deve contenere colonne 'id' e 'text_cleaned'.")
    texts = texts.rename(columns={"id": "user_id", "text_cleaned": "text"})
    texts = texts[["user_id", "text"]]
    texts_grouped = texts.groupby("user_id")["text"].apply(list).reset_index()

    #Test graph edges: social and (optionally) spatial
    rel_edges  = _read_edges_edg(os.path.join(DATASET_DIR, "graph", "social_network_test.edg"))
    spat_edges = None
    if params.get("explanations", {}).get("perspectives", {}).get("use_spatial", False):
        spat_edges = _read_edges_edg(os.path.join(DATASET_DIR, "graph", "spatial_network_test.edg"))

    #Merge texts into test_df
    test_df = test_df.merge(texts_grouped, on="user_id", how="left")

    #Optional negative lexicon
    neg_lex = None
    if params.get("explanations", {}).get("use_negative_lexicon", True):
        neg_lex = _load_negative_lexicon(params)

    return {
        "test_df": test_df,          # user_id, pred_label, AE_*, SHAP_*, text (list[str])
        "shap_df": shap_df,          # full SHAP CSV
        "rel_edges": rel_edges,      # src,dst
        "spatial_edges": spat_edges, # src,dst,(w)
        "negative_lexicon": neg_lex, # dict token->weight (or None)
        "paths": {"shap_csv": shap_csv}
    }

# ---------------------------------------------------------------------
# EXTRACTORS
# ---------------------------------------------------------------------
def _extract_shap_for_user(uid: str, artifacts: dict) -> dict | None:
    """Get the SHAP row for a specific user_id."""
    shap_df = artifacts.get("shap_df")
    if shap_df is None or shap_df.empty:
        return None
    row = shap_df.loc[shap_df["user_id"].astype(str) == str(uid)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def _get_pred_label(uid: str, artifacts: dict) -> str:
    """Return predicted label as 'safe'/'risky' or 'unknown'."""
    df = artifacts.get("test_df")
    if df is None or df.empty:
        return "unknown"
    row = df.loc[df["user_id"].astype(str) == str(uid)]
    if row.empty:
        return "unknown"
    return str(row.iloc[0].get("pred_label", "unknown"))

def _get_pred_prob(uid: str, artifacts: dict) -> float:
    """No predicted probability available in SHAP CSV -> return NaN."""
    return float("nan")

# ---------------------------------------------------------------------
# SIGNAL COMPUTATION
# ---------------------------------------------------------------------
def compute_text_signals(uid: str, artifacts: dict, top_terms: int = 6, include_examples: bool = False) -> dict:
    """Compute user-level text signals: top terms, AE deltas, negative lexicon overlap."""
    df = artifacts["test_df"]
    row = df.loc[df["user_id"].astype(str) == str(uid)]
    texts = row.iloc[0]["text"] if not row.empty else []
    if not isinstance(texts, list):
        texts = [] if pd.isna(texts) else [texts]

    # Top-N terms over all user posts
    cnt = Counter()
    for t in texts:
        cnt.update(_clean_and_tokenize(t))
    top_terms_user = [w for (w, _) in cnt.most_common(top_terms)]

    # First example post (optional)
    example_post = None
    if include_examples and texts:
        for t in texts:
            toks = _clean_and_tokenize(t)
            if len(toks) >= 1:
                example_post = t.strip()
                break

    # AE delta: risky - safe (<0 → closer to risky archetype; >0 → closer to safe)
    ae_safe = row.iloc[0].get("AE_safe_err", None) if not row.empty else None
    ae_risk = row.iloc[0].get("AE_risk_err", None) if not row.empty else None
    ae_delta = None
    try:
        if ae_safe is not None and ae_risk is not None:
            ae_delta = float(ae_risk) - float(ae_safe)
    except Exception:
        pass

    # Negative lexicon overlap (optional): score = user_term_freq * lexicon_weight
    neg_lex = artifacts.get("negative_lexicon")
    negative_overlap_terms = []
    negative_score = 0.0
    if neg_lex:
        scores = []
        for term, f in cnt.items():
            w = neg_lex.get(term)
            if w:
                scores.append((term, f * w))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            take = min(top_terms, len(scores))
            negative_overlap_terms = [term for term, _ in scores[:take]]
            negative_score = float(sum(s for _, s in scores))

    return {
        "top_terms_user": top_terms_user,
        "ae_delta": ae_delta,
        "example_post": example_post,
        "neg_terms": negative_overlap_terms,
        "neg_score": negative_score,
    }

def compute_network_signals(uid: str, artifacts: dict, neighbor_k="all", use_pred: bool = True, top_terms: int = 6) -> dict:
    """Relational perspective: degree, risky ratio among neighbors, homophily delta, neighbors' top terms."""
    rel_edges = artifacts.get("rel_edges")
    test_df   = artifacts["test_df"]

    if rel_edges is None or rel_edges.empty:
        return {
            "degree_total": 0,
            "degree_used": 0,
            "risky_rate_neighbors": None,
            "homophily_delta": None,
            "top_terms_neighbors": [],
        }

    if use_pred is False:
        print("[explanations][WARN] use_predicted_neighbors=false è sconsigliato: ignoreremo comunque le true label e useremo le PREDIZIONI.")

    uid_str = str(uid)

    # Undirected neighborhood
    neighbors_all = set(rel_edges.loc[rel_edges["src"].astype(str) == uid_str, "dst"].astype(str)) | \
                    set(rel_edges.loc[rel_edges["dst"].astype(str) == uid_str, "src"].astype(str))
    neighbors_all = list(neighbors_all)
    degree_total = len(neighbors_all)

    if degree_total == 0:
        return {
            "degree_total": 0,
            "degree_used": 0,
            "risky_rate_neighbors": None,
            "homophily_delta": None,
            "top_terms_neighbors": [],
        }

    # neighbor_k: "all" or integer cutoff
    use_all = (isinstance(neighbor_k, str) and neighbor_k.strip().lower() == "all")
    if use_all:
        neighbors_used = neighbors_all
    else:
        try:
            k = int(neighbor_k)
            neighbors_used = neighbors_all[:max(0, k)]
        except Exception:
            neighbors_used = neighbors_all

    degree_used = len(neighbors_used)
    if degree_used == 0:
        return {
            "degree_total": degree_total,
            "degree_used": 0,
            "risky_rate_neighbors": None,
            "homophily_delta": None,
            "top_terms_neighbors": [],
        }

    # Use predicted labels of neighbors
    sub = test_df.loc[test_df["user_id"].astype(str).isin(neighbors_used)].copy()
    risky_rate_neighbors = float((sub["pred_label"] == "risky").mean()) if not sub.empty else None

    # Homophily delta relative to global test base rate
    base_rate = float((test_df["pred_label"] == "risky").mean())
    homophily_delta = None if risky_rate_neighbors is None else (risky_rate_neighbors - base_rate)

    # Neighbors' top terms (aggregated texts)
    cnt = Counter()
    all_texts = []
    for ts in sub["text"]:
        if isinstance(ts, list):
            all_texts.extend(ts)
        elif isinstance(ts, str):
            all_texts.append(ts)
    for t in all_texts:
        cnt.update(_clean_and_tokenize(t))
    top_terms_neighbors = [w for (w, _) in cnt.most_common(top_terms)]

    return {
        "degree_total": int(degree_total),
        "degree_used": int(degree_used),
        "risky_rate_neighbors": risky_rate_neighbors,
        "homophily_delta": homophily_delta,
        "top_terms_neighbors": top_terms_neighbors,
    }

def compute_spatial_signals(uid: str, artifacts: dict, neighbor_k="all", use_pred: bool = True, top_terms: int = 6) -> dict:
    """Spatial perspective: same logic as relational; optional edge weights summary is kept (not printed)."""
    spat_edges = artifacts.get("spatial_edges")
    test_df    = artifacts["test_df"]

    if spat_edges is None or spat_edges.empty:
        return {
            "degree_total": 0,
            "degree_used": 0,
            "risky_rate_neighbors": None,
            "homophily_delta": None,
            "top_terms_neighbors": [],
            "w_stats": None,
        }

    if use_pred is False:
        print("[explanations][WARN] use_predicted_neighbors=false è sconsigliato: useremo comunque le PREDIZIONI.")

    uid_str = str(uid)

    # Undirected neighborhood (spatial)
    neighbors_all = set(spat_edges.loc[spat_edges["src"].astype(str) == uid_str, "dst"].astype(str)) | \
                    set(spat_edges.loc[spat_edges["dst"].astype(str) == uid_str, "src"].astype(str))
    neighbors_all = list(neighbors_all)
    degree_total = len(neighbors_all)

    if degree_total == 0:
        return {
            "degree_total": 0,
            "degree_used": 0,
            "risky_rate_neighbors": None,
            "homophily_delta": None,
            "top_terms_neighbors": [],
            "w_stats": None,
        }

    # neighbor_k: "all" or integer cutoff
    use_all = (isinstance(neighbor_k, str) and neighbor_k.strip().lower() == "all")
    if use_all:
        neighbors_used = neighbors_all
    else:
        try:
            k = int(neighbor_k)
            neighbors_used = neighbors_all[:max(0, k)]
        except Exception:
            neighbors_used = neighbors_all

    degree_used = len(neighbors_used)
    if degree_used == 0:
        return {
            "degree_total": degree_total,
            "degree_used": 0,
            "risky_rate_neighbors": None,
            "homophily_delta": None,
            "top_terms_neighbors": [],
            "w_stats": None,
        }

    # Use predicted labels of spatial neighbors
    sub = test_df.loc[test_df["user_id"].astype(str).isin(neighbors_used)].copy()
    risky_rate_neighbors = float((sub["pred_label"] == "risky").mean()) if not sub.empty else None

    # Homophily delta relative to global test base rate
    base_rate = float((test_df["pred_label"] == "risky").mean())
    homophily_delta = None if risky_rate_neighbors is None else (risky_rate_neighbors - base_rate)

    # Neighbors' top terms (spatial)
    cnt = Counter()
    all_texts = []
    for ts in sub["text"]:
        if isinstance(ts, list):
            all_texts.extend(ts)
        elif isinstance(ts, str):
            all_texts.append(ts)
    for t in all_texts:
        cnt.update(_clean_and_tokenize(t))
    top_terms_neighbors = [w for (w, _) in cnt.most_common(top_terms)]

    # Optional spatial weights summary (not printed, kept for future use)
    w_stats = None
    if "w" in spat_edges.columns:
        mask = (spat_edges["src"].astype(str) == uid_str) | (spat_edges["dst"].astype(str) == uid_str)
        w_series = spat_edges.loc[mask, "w"]
        if not w_series.empty:
            w_stats = {"w_mean": float(w_series.mean()), "w_sum": float(w_series.sum())}

    return {
        "degree_total": int(degree_total),
        "degree_used": int(degree_used),
        "risky_rate_neighbors": risky_rate_neighbors,
        "homophily_delta": homophily_delta,
        "top_terms_neighbors": top_terms_neighbors,
        "w_stats": w_stats,
    }

# ---------------------------------------------------------------------
# WORDING HELPERS 
# ---------------------------------------------------------------------
def _join_terms(terms: list[str]) -> str:
    """Format list as Italian natural language: “a”, “b” e “c”."""
    if not terms:
        return ""
    if len(terms) == 1:
        return f"“{terms[0]}”"
    if len(terms) == 2:
        return f"“{terms[0]}” e “{terms[1]}”"
    return " , ".join([f"“{t}”" for t in terms[:-1]]) + f" e “{terms[-1]}”"

def _qual_share(p: float | None) -> str:
    """Qualitative share text for ratios in [0,1] (Italian phrasing)."""
    if p is None or p != p:
        return "non è possibile stimare la quota degli utenti predetti 'risky'"
    pct = round(100 * float(p))
    if p >= 0.95:
        return f"quasi tutti (≈{pct}%)"
    if p >= 0.75:
        return f"la grande maggioranza (≈{pct}%)"
    if p >= 0.55:
        return f"la maggioranza (≈{pct}%)"
    if p > 0.45:
        return "in parti quasi uguali (≈50%)"
    if p > 0.25:
        return f"una minoranza (≈{pct}%)"
    return f"pochissimi (≈{pct}%)"

def _qual_degree_sentence(deg_tot: int, deg_used: int | None = None) -> str:
    """Natural sentence explaining degree as number of social connections."""
    if deg_tot <= 0:
        return "L’utente non risulta connesso ad altri nella rete sociale."
    base = f"L’utente è connesso con {deg_tot} altri utenti nella rete sociale"
    if deg_used is not None and deg_used < deg_tot:
        base += f" (l’analisi considera {deg_used} di queste connessioni)"
    return base + "."

def _qual_degree_sentence_spatial(deg_tot: int, deg_used: int | None = None) -> str:
    """Natural sentence for spatial connectivity."""
    if deg_tot <= 0:
        return "Nella prospettiva spaziale non risultano connessioni."
    base = f"Nella prospettiva spaziale risulta connesso a {deg_tot} altri utenti"
    if deg_used is not None and deg_used < deg_tot:
        base += f" (l’analisi considera {deg_used} di queste connessioni)"
    return base + "."

# ---------------------------------------------------------------------
# RENDER 
# ---------------------------------------------------------------------
def render_explanation(uid: str, pred_label: str, pred_prob: float,
                       shap_signals: dict, all_signals: dict) -> str:
    # SHAP components
    shap_content = float(shap_signals.get("SHAP_content", 0.0) or 0.0)
    shap_rel     = float(shap_signals.get("SHAP_rel", 0.0) or 0.0)
    shap_spat    = float(shap_signals.get("SHAP_spat", 0.0) or 0.0)
    tot_shap     = shap_content + shap_rel + shap_spat

    def _fmt_pct(x):
        try:
            return f"{100.0*float(x):.0f}%"
        except Exception:
            return "—"

    lines = []


    lines.append(f"Utente {uid} — Predetto: {pred_label}.")
    use_content = ("content" in all_signals)
    use_rel     = ("relational" in all_signals)
    use_spatial = ("spatial" in all_signals)

    # (nome_italiano, chiave_logica, valore_shap)
    active = []
    if use_content:
        active.append(("contenutistica", "content", float(shap_content)))
    if use_rel:
        active.append(("relazionale", "rel", float(shap_rel)))
    if use_spatial:
        active.append(("spaziale", "spat", float(shap_spat)))

    tot_active = sum(v for _, __, v in active)

    def _fmt_pct(x):
        try:
            return f"{100.0*float(x):.0f}%"
        except Exception:
            return "—"

    # SHAP dominance among the active perspectives only
    if tot_active > 0 and active:
        dom_name, dom_key, dom_val = max(active, key=lambda t: t[2])
        lines.append(
            f"La decisione è dipesa soprattutto dalla prospettiva {dom_name} "
            f"({_fmt_pct(dom_val/tot_active)} del contributo SHAP)."
        )

    # Percentage breakdown among active perspectives (sums to 100% across active)
    if tot_active > 0 and active:
        label_map = {"content": "Contenuti", "rel": "Relazioni", "spat": "Spaziale"}
        parts = [f"{label_map[k]}: {_fmt_pct(v/tot_active)}" for _, k, v in active]
        lines.append("Ripartizione SHAP stimata — " + ", ".join(parts) + ".")

    # Flag for the "conflicts" section below
    dom_key = None
    if tot_active > 0 and active:
        dom_name, dom_key, dom_val = max(active, key=lambda t: t[2])

    dom_is_cont = (dom_key == "content")
    dom_is_rel  = (dom_key == "rel")
    dom_is_spat = (dom_key == "spat")

    # --- User textual content ---
    c = all_signals.get("content", {})
    if c:
        ae_delta = c.get("ae_delta")
        tt_user  = _join_terms(c.get("top_terms_user", []))
        if ae_delta is not None:
            if ae_delta < 0:
                lines.append("\n" + "Un'analisi del contenuto postato indica che il suo linguaggio è più vicino all'archetipo 'risky'")
            elif ae_delta > 0:
                lines.append("\n" + "Un'analisi del contenuto postato indica che il suo linguaggio è più vicino all'archetipo 'safe'")
        if tt_user:
            lines.append(f"Sono state individuate le seguenti parole chiave : {tt_user}.")
        if c.get("neg_terms"):
            lines.append("\n" + f"Sono presenti anche termini associati a contenuti negativi: {_join_terms(c['neg_terms'])}.")

    # --- Social network perspective ---
    r = all_signals.get("relational", {})
    if r:
        deg_tot  = r.get("degree_total", r.get("degree", 0))
        deg_used = r.get("degree_used", deg_tot)
        lines.append(_qual_degree_sentence(deg_tot, deg_used))

        rrn = r.get("risky_rate_neighbors")
        if rrn is not None:
            share_txt = _qual_share(rrn)
            if r.get("homophily_delta") is not None:
                base = rrn - r["homophily_delta"]
                lines.append("\n" + f"Tra gli utenti con cui è connesso, {share_txt} risultano predetti 'risky' (nella media del test ≈{_fmt_pct(base)}).")
            else:
                lines.append("\n" + f"Tra gli utenti con cui è connesso, {share_txt} risultano predetti 'risky'.")

        tt_neigh = _join_terms(r.get("top_terms_neighbors", []))
        if tt_neigh:
            lines.append("\n" + f"Per loro sono state individuate le seguenti parole chiave: {tt_neigh}.")

        if deg_tot <= 3:
            lines.append("\n" + "Nota: l’utente ha pochissime connessioni, potrebbe essere un profilo isolato e quindi più difficile da classificare, le percentuali potrebbero essere instabili.")

    # --- Spatial perspective ---
    s = all_signals.get("spatial")
    if s:
        deg_tot_s  = s.get("degree_total", s.get("spatial_degree", 0))
        deg_used_s = s.get("degree_used", deg_tot_s)
        lines.append(_qual_degree_sentence_spatial(deg_tot_s, deg_used_s))

        rrn_s = s.get("risky_rate_neighbors")
        if rrn_s is not None:
            share_txt_s = _qual_share(rrn_s)
            if s.get("homophily_delta") is not None:
                base_s = rrn_s - s["homophily_delta"]
                lines.append("\n" + f"In questa prospettiva, {share_txt_s} degli utenti collegati risultano predetti 'risky' (media del test ≈{_fmt_pct(base_s)}).")
            else:
                lines.append("\n" + f"In questa prospettiva, {share_txt_s} degli utenti collegati risultano predetti 'risky'.")

        tt_neigh_s = _join_terms(s.get("top_terms_neighbors", []))
        if tt_neigh_s:
            lines.append("\n" + f"I termini caratteristici degli utenti vicini (prospettiva spaziale) sono: {tt_neigh_s}.")

        if deg_tot_s <= 3:
            lines.append("\n" + "Nota: nella prospettiva spaziale le connessioni sono pochissime, potrebbe essere un profilo isolato e quindi più difficile da classificare, le percentuali potrebbero essere instabili.")

    # --- Conflict clarity across perspectives ---
    if c and tot_shap > 0:
        ae_delta = c.get("ae_delta")  # >0 => closer to 'safe'; <0 => closer to 'risky'
        dom_val = max(shap_content, shap_rel, shap_spat)
        dom_is_rel = (dom_val == shap_rel)
        dom_is_cont = (dom_val == shap_content)
        dom_is_spat = (dom_val == shap_spat)

        # Relational dominant
        if all_signals.get("relational") and dom_is_rel:
            if ae_delta is not None and ae_delta > 0 and pred_label == "risky":
                lines.append("\n" + "Scelta finale: sebbene i contenuti rispecchino quelli di un utente 'safe', ha prevalso la struttura relazionale: "
                             "l’utente è collegato a profili perlopiù 'risky' e il loro lessico è coerente con quella classe.")
            if ae_delta is not None and ae_delta < 0 and pred_label == "safe":
                lines.append("\n" + "Scelta finale: sebbene i contenuti rispecchino quelli di un utente 'risky', la configurazione delle connessioni sociali "
                             "e il peso complessivo delle feature hanno portato alla classificazione 'safe'.")

        # Spatial dominant
        if all_signals.get("spatial") and dom_is_spat:
            rrn_s = all_signals["spatial"].get("risky_rate_neighbors")
            if ae_delta is not None and ae_delta > 0 and pred_label == "risky":
                share_txt_s = _qual_share(rrn_s) if rrn_s is not None else "una quota rilevante"
                lines.append("\n" + "Scelta finale: sebbene i contenuti rispecchino quelli di un utente 'safe', ha pesato la struttura spaziale: "
                             f"nella prossimità spaziale {share_txt_s} degli utenti collegati risultano 'risky' e il loro lessico è coerente con quella classe.")
            if ae_delta is not None and ae_delta < 0 and pred_label == "safe":
                if rrn_s is not None and rrn_s <= 0.5:
                    share_txt_s = _qual_share(rrn_s)
                    lines.append("\n" + "Scelta finale: nonostante i contenuti rispecchino quelli di un utente 'risky', la struttura spaziale ha attenuato questo segnale: "
                                 f"tra gli utenti collegati spazialmente {share_txt_s} risultano 'risky', favorendo la classificazione 'safe'.")
                else:
                    lines.append("\n" + "Scelta finale: seppur i contenuti rispecchino quelli di un utente 'risky', la configurazione spaziale delle connessioni "
                                 "e il peso complessivo delle feature hanno portato alla classificazione 'safe'.")

        # Content dominant (note)
        if dom_is_cont and ae_delta is not None:
            if ae_delta > 0 and pred_label == "risky":
                lines.append("Nota: i contenuti rispecchiano quelli di un utente 'safe', ma altri fattori (relazionali/spaziali) hanno spinto verso 'risky'.")
            if ae_delta < 0 and pred_label == "safe":
                lines.append("Nota: i contenuti rispecchiano quelli di un utente 'risky', ma il bilancio complessivo delle altre prospettive ha spinto verso 'safe'.")

    # --- Summary ---
    lines.append("\n" + "In sintesi, questi elementi spiegano la classificazione assegnata dal modello.")
    return "\n".join(lines)

# ---------------------------------------------------------------------
# PUBLIC INTERFACE (interactive hook)
# ---------------------------------------------------------------------
def explain_user_interactive(params: dict) -> None:
    """
    Interactive hook at the end of test: ask for a user_id and print a concise explanation.
    Repeats until the user presses Enter on an empty input (or types 'q').
    Uses content + relational + (optional) spatial perspectives, guided by SHAP.
    """
    if not params.get("explanations", {}).get("enable_interactive", False):
        print("[explanations] Funzionalità disabilitata (enable_interactive=false).")
        return

    try:
        artifacts = _load_artifacts(params)  
    except Exception as e:
        print(f"[explanations] Errore nel caricamento degli artifact: {e}")
        return

    exp_cfg = params.get("explanations", {})
    pers    = exp_cfg.get("perspectives", {})

    print("— Modalità spiegazioni interattive —")
    print("Inserisci un ID utente per ottenere la spiegazione.")
    print("Lascia vuoto e premi Invio per uscire.\n")

    while True:
        uid = input("ID utente: ").strip()
        if not uid or uid.lower() in {"q", "quit", "exit"}:
            print("Uscita dalla modalità interattiva.")
            break

        try:
            shap_row = _extract_shap_for_user(uid, artifacts)
            if shap_row is None:
                print(f"[explanations] Nessuna riga SHAP trovata per utente: {uid}\n")
                continue

            
            signals = {}
            if pers.get("use_content", True):
                signals["content"] = compute_text_signals(
                    uid, artifacts,
                    top_terms=exp_cfg.get("top_terms", 6),
                    include_examples=exp_cfg.get("include_examples", False)
                )
            if pers.get("use_relational", True):
                signals["relational"] = compute_network_signals(
                    uid, artifacts,
                    neighbor_k=exp_cfg.get("neighbor_k", "all"),
                    use_pred=exp_cfg.get("use_predicted_neighbors", True),
                    top_terms=exp_cfg.get("top_terms", 6)
                )
            if pers.get("use_spatial", False):
                signals["spatial"] = compute_spatial_signals(
                    uid, artifacts,
                    neighbor_k=exp_cfg.get("neighbor_k", "all"),
                    use_pred=exp_cfg.get("use_predicted_neighbors", True),
                    top_terms=exp_cfg.get("top_terms", 6)
                )

            explanation = render_explanation(
                uid=uid,
                pred_label=_get_pred_label(uid, artifacts),
                pred_prob=_get_pred_prob(uid, artifacts),  # NaN by design
                shap_signals=shap_row,
                all_signals=signals,
            )

            print("\n" + explanation + "\n")

        except KeyboardInterrupt:
            print("\nInterrotto dall'utente. Uscita.")
            break
        except Exception as e:
            print(f"[explanations] Errore durante la spiegazione di {uid}: {e}\n")
            continue
