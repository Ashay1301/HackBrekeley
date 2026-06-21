# Feature Implementation Plan

Three features to implement in order. Each builds on the previous.

**Prerequisite:** Redis session storage from `sponsor_integrations.md §2` must be implemented first —
multi-night dashboard and dream journal both need persistent sessions.

---

## Build order

| # | Feature | Time | Depends on |
|---|---------|------|-----------|
| 1 | Sleep Disorder Risk Screener | 1.5 hrs | Nothing — pure logic |
| 2 | Multi-Night Trend Dashboard | 2 hrs | Redis sessions |
| 3 | Dream Journal + REM Correlation | 2 hrs | Redis sessions + Claude |

---

## Feature 1: Sleep Disorder Risk Screener

Rule-based flags computed immediately after inference. No new model. Shown as warning cards
below the main metrics panel. Flags likely apnea, insomnia, and REM suppression from patterns
already present in `sleep_summary`.

---

### Create `backend/disorder_screener.py`

```python
"""
Rule-based sleep disorder risk screener.
Runs on the sleep_summary dict returned by analyze_predictions().
Returns a list of risk flags — empty list means no concerns flagged.

Thresholds are conservative (favour recall over precision) and include
"talk to a doctor" disclaimers. This is a screener, not a diagnosis.
"""


# AASM population norms used as thresholds
_NORMS = {
    "rem_ideal_pct":    20.0,   # adults: 20–25%
    "deep_ideal_pct":   15.0,   # adults: 13–23%
    "efficiency_good":  85.0,
    "latency_ok_min":   20.0,
    "latency_bad_min":  30.0,
}


def screen(summary: dict) -> list[dict]:
    """
    Parameters
    ----------
    summary : dict
        The sleep_summary block from analyze_predictions() output.

    Returns
    -------
    list[dict]
        Each flag: {condition, risk_level, indicators, recommendation}
        risk_level: "info" | "moderate" | "high"
    """
    flags = []

    efficiency  = float(summary.get("efficiency_pct", 100))
    latency     = float(summary.get("latency_min", 0))
    awakenings  = int(summary.get("awakenings", 0))
    total_sleep = float(summary.get("total_sleep_min", 480))
    pcts        = summary.get("pct_in_stage", {})

    # Normalise stage keys — handles both EEG ("REM","N3") and wrist ("REM","Deep")
    rem_pct  = next((v for k, v in pcts.items() if k.upper() in ("REM",)), 0.0)
    deep_pct = next((v for k, v in pcts.items() if k.upper() in ("N3", "DEEP")), 0.0)

    # ── 1. Sleep Apnea ──────────────────────────────────────────────────────────
    # Proxy: frequent awakenings + low efficiency (true apnea requires oximetry)
    if awakenings >= 5:
        flags.append({
            "condition":      "Sleep Apnea Risk",
            "risk_level":     "high",
            "indicators":     [
                f"{awakenings} awakenings (normal: ≤2)",
                f"{efficiency}% sleep efficiency",
            ],
            "recommendation": (
                "Frequent awakenings with low efficiency are associated with "
                "sleep-disordered breathing. If you snore, gasp, or feel "
                "unrefreshed after a full night, speak with a doctor about "
                "a formal sleep study (polysomnography)."
            ),
        })
    elif awakenings >= 3 and efficiency < 80:
        flags.append({
            "condition":      "Sleep Apnea Risk",
            "risk_level":     "moderate",
            "indicators":     [
                f"{awakenings} awakenings",
                f"{efficiency}% sleep efficiency",
            ],
            "recommendation": (
                "Mildly fragmented sleep. Common causes include sleep apnea, "
                "restless legs, or environmental disruption. Track this over "
                "several nights — if it persists, discuss with your doctor."
            ),
        })

    # ── 2. Insomnia ─────────────────────────────────────────────────────────────
    if latency > _NORMS["latency_bad_min"] and efficiency < _NORMS["efficiency_good"]:
        flags.append({
            "condition":      "Insomnia",
            "risk_level":     "high",
            "indicators":     [
                f"{latency} min to fall asleep (normal: <20 min)",
                f"{efficiency}% sleep efficiency (normal: >85%)",
            ],
            "recommendation": (
                "Long sleep onset combined with poor efficiency is a hallmark "
                "of insomnia disorder. Cognitive Behavioural Therapy for "
                "Insomnia (CBT-I) is the first-line evidence-based treatment — "
                "more effective than medication long-term."
            ),
        })
    elif latency > _NORMS["latency_ok_min"] or efficiency < _NORMS["efficiency_good"]:
        flags.append({
            "condition":      "Insomnia",
            "risk_level":     "moderate",
            "indicators":     [
                f"{latency} min to fall asleep",
                f"{efficiency}% sleep efficiency",
            ],
            "recommendation": (
                "Mild difficulty falling or staying asleep. Consistent "
                "sleep/wake times, avoiding screens 1 hour before bed, and "
                "keeping the bedroom cool (65–68°F) are the highest-evidence "
                "behavioural changes."
            ),
        })

    # ── 3. REM Suppression ──────────────────────────────────────────────────────
    if rem_pct > 0:   # only flag if REM is measurable (wrist model sometimes can't)
        if rem_pct < 8:
            flags.append({
                "condition":      "REM Suppression",
                "risk_level":     "high",
                "indicators":     [f"{rem_pct}% REM sleep (normal: 20–25%)"],
                "recommendation": (
                    "Very low REM is commonly caused by alcohol within 3 hours "
                    "of bedtime, certain antidepressants (SSRIs, SNRIs), "
                    "benzodiazepines, or untreated sleep apnea. It affects "
                    "memory consolidation and emotional regulation. Worth "
                    "discussing with your doctor."
                ),
            })
        elif rem_pct < 15:
            flags.append({
                "condition":      "REM Suppression",
                "risk_level":     "moderate",
                "indicators":     [f"{rem_pct}% REM sleep (normal: 20–25%)"],
                "recommendation": (
                    "Below-average REM. Most common cause: alcohol or cannabis "
                    "within a few hours of bedtime. Also affected by irregular "
                    "sleep schedules — REM is disproportionately concentrated "
                    "in the last third of a full night."
                ),
            })

    # ── 4. Short Sleep ──────────────────────────────────────────────────────────
    if total_sleep < 300:   # under 5 hours
        flags.append({
            "condition":      "Short Sleep",
            "risk_level":     "high",
            "indicators":     [f"{round(total_sleep / 60, 1)} hours total sleep"],
            "recommendation": (
                "Less than 5 hours of sleep significantly impairs cognitive "
                "function, immune response, and metabolic health. The CDC "
                "recommends 7–9 hours for adults. Chronic short sleep is "
                "associated with increased cardiovascular and diabetes risk."
            ),
        })
    elif total_sleep < 360:  # under 6 hours
        flags.append({
            "condition":      "Short Sleep",
            "risk_level":     "moderate",
            "indicators":     [f"{round(total_sleep / 60, 1)} hours total sleep"],
            "recommendation": (
                "Slightly below the recommended 7–9 hours. Even one hour of "
                "sleep debt measurably reduces next-day attention and reaction "
                "time. Prioritise an earlier bedtime over a later wake time."
            ),
        })

    return flags
```

---

### `backend/main.py` — call screener in `/api/analyze`

Add import at top:
```python
from backend.disorder_screener import screen as disorder_screen
```

Inside `POST /api/analyze`, after `result = eeg_predictor.predict(data)` (or `wrist_predictor.predict`),
before `return JSONResponse(result)`:

```python
    result["disorder_flags"] = disorder_screen(result.get("sleep_summary", {}))
```

Also add it in `GET /api/demo` inside `_normalise_demo()`, at the end before return:

```python
    out["disorder_flags"] = disorder_screen(out["sleep_summary"])
    return out
```

---

### `frontend/index.html` — disorder cards

Add this HTML block inside the results section, after the metrics grid and before the charts:

```html
<div id="disorderAlerts" style="display:none; margin-bottom: 24px;"></div>
```

Add this JS function and call it inside `renderResults(data)`:

```javascript
function renderDisorderFlags(flags) {
    const container = document.getElementById('disorderAlerts');
    if (!flags || flags.length === 0) { container.style.display = 'none'; return; }

    const colorMap = { high: '#FF7059', moderate: '#FFB347', info: '#4DFFD2' };
    const iconMap  = { high: '⚠️', moderate: '⚡', info: 'ℹ️' };

    container.innerHTML = flags.map(f => `
        <div style="
            border-left: 3px solid ${colorMap[f.risk_level] || '#FFB347'};
            background: rgba(255,255,255,0.04);
            border-radius: 2px;
            padding: 14px 18px;
            margin-bottom: 10px;
        ">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <span>${iconMap[f.risk_level] || '⚡'}</span>
                <strong style="color:#fff; font-size:14px;">${f.condition}</strong>
                <span style="
                    font-size:10px; font-weight:700; letter-spacing:0.1em;
                    text-transform:uppercase; color:${colorMap[f.risk_level]};
                    padding:2px 6px; border:1px solid ${colorMap[f.risk_level]};
                    border-radius:2px;
                ">${f.risk_level}</span>
            </div>
            <div style="font-size:12px; color:#6E8099; margin-bottom:6px;">
                ${f.indicators.join(' · ')}
            </div>
            <div style="font-size:13px; color:#aab; line-height:1.5;">
                ${f.recommendation}
            </div>
        </div>
    `).join('');
    container.style.display = 'block';
}

// Inside renderResults(data):
renderDisorderFlags(data.disorder_flags);
```

---

### Verify

```bash
# Run demo — check disorder_flags in response
curl http://localhost:8000/api/demo | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Flags:', json.dumps(d.get('disorder_flags', []), indent=2))
"

# Manual test: craft a summary with high awakenings
curl -X POST http://localhost:8000/api/analyze -F "file=@path/to/file.edf"
# disorder_flags should appear in JSON if thresholds are met
```

---

## Feature 2: Multi-Night Trend Dashboard

Requires Redis from `sponsor_integrations.md §2`. Sessions must be saving before this works.

Shows a slide-in history panel: Chart.js trend lines for score/efficiency/REM over the last
10 nights, plus a Claude-generated weekly narrative.

---

### `backend/session.py` — add two functions

Append to the existing `session.py` created in `sponsor_integrations.md`:

```python
def record_session_index(session_id: str, timestamp_epoch: float) -> None:
    """Add session_id to the global time-sorted index."""
    r = _redis()
    if r:
        r.zadd("sessions:index", {session_id: timestamp_epoch})
    else:
        _memory_store.setdefault("sessions:index", []).append(
            (timestamp_epoch, session_id)
        )


def get_recent_session_ids(limit: int = 10) -> list[str]:
    """Return up to `limit` most recent session IDs, newest first."""
    r = _redis()
    if r:
        return r.zrevrange("sessions:index", 0, limit - 1)
    entries = _memory_store.get("sessions:index", [])
    return [sid for _, sid in sorted(entries, reverse=True)[:limit]]
```

---

### `backend/main.py` — hook + two new routes

**In `POST /api/analyze`**, after `session_store.save_analysis(sid, result)`:

```python
    import time
    session_store.record_session_index(sid, time.time())
```

**New route — GET /api/history:**

```python
@app.get("/api/history")
async def get_history(limit: int = 10):
    """Return the last `limit` analyses for trend display."""
    ids = session_store.get_recent_session_ids(limit=limit)
    analyses = []
    for sid in ids:
        data = session_store.load_analysis(sid)
        if data:
            # Return lightweight summary only — no epoch_by_epoch_data
            s = data.get("sleep_summary", {})
            m = data.get("metadata", {})
            analyses.append({
                "session_id":     sid,
                "date":           m.get("recording_start", "")[:10],
                "analysis_type":  m.get("analysis_type", ""),
                "quality_score":  s.get("quality_score"),
                "quality_grade":  s.get("quality_grade"),
                "efficiency_pct": s.get("efficiency_pct"),
                "latency_min":    s.get("latency_min"),
                "awakenings":     s.get("awakenings"),
                "rem_pct":        s.get("pct_in_stage", {}).get("REM")
                                  or s.get("pct_in_stage", {}).get("REM", 0),
                "deep_pct":       s.get("pct_in_stage", {}).get("N3")
                                  or s.get("pct_in_stage", {}).get("Deep", 0),
                "total_sleep_min": s.get("total_sleep_min"),
            })
    return {"nights": analyses}
```

**New route — POST /api/weekly-summary:**

```python
class WeeklySummaryRequest(BaseModel):
    nights: list[dict]   # list of lightweight night dicts from /api/history

@app.post("/api/weekly-summary")
async def weekly_summary(req: WeeklySummaryRequest):
    """Claude writes a narrative weekly sleep summary."""
    if not req.nights:
        raise HTTPException(status_code=400, detail="No nights provided.")
    try:
        summary_text = chat_module.generate_weekly_summary(req.nights)
        return {"summary": summary_text}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
```

---

### `backend/chat.py` — add weekly summary function

Append to the existing `chat.py`:

```python
def generate_weekly_summary(nights: list[dict]) -> str:
    """
    Generate a 2–3 paragraph narrative summary of the user's sleep over
    multiple nights. `nights` is a list of lightweight night dicts from
    /api/history (date, quality_score, efficiency_pct, rem_pct, etc.).
    """
    import json

    # Build a readable table for Claude
    rows = []
    for n in sorted(nights, key=lambda x: x.get("date", "")):
        rows.append(
            f"  {n.get('date','?')}: score={n.get('quality_score','?')},"
            f" efficiency={n.get('efficiency_pct','?')}%,"
            f" latency={n.get('latency_min','?')}min,"
            f" awakenings={n.get('awakenings','?')},"
            f" REM={n.get('rem_pct','?')}%"
        )

    prompt = (
        "Here is the user's sleep data over the past several nights:\n\n"
        + "\n".join(rows)
        + "\n\nWrite a 2–3 paragraph narrative weekly sleep summary. "
        "Cover: the overall trend (improving, declining, or stable), "
        "the best and worst night and what might explain the difference, "
        "and one specific, actionable recommendation. "
        "Be warm, specific, and evidence-based. Plain language only."
    )

    response = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

---

### `frontend/index.html` — history panel

**Add history button to the header:**

```html
<button id="historyBtn" style="...">📅 History</button>
```

**Add slide-in panel (add before closing `</body>`):**

```html
<div id="historyPanel" style="
    position: fixed; top: 0; right: -420px; width: 420px; height: 100vh;
    background: #131E30; border-left: 1px solid #253347;
    overflow-y: auto; transition: right 0.3s ease; z-index: 1000;
    padding: 28px 24px;
">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <h3 style="color:#fff; font-size:18px; margin:0;">Sleep History</h3>
        <button id="historyClose" style="background:none; border:none; color:#6E8099; font-size:20px; cursor:pointer;">✕</button>
    </div>
    <canvas id="historyChart" height="160" style="margin-bottom:20px;"></canvas>
    <div id="historyTable" style="margin-bottom:20px;"></div>
    <div id="weeklySummary" style="
        background:rgba(77,255,210,0.06); border:1px solid #253347;
        border-radius:2px; padding:14px; font-size:13px; color:#aab; line-height:1.6;
    "></div>
</div>
<div id="historyOverlay" style="
    display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:999;
"></div>
```

**JS — history panel logic:**

```javascript
let historyChartInstance = null;

document.getElementById('historyBtn').addEventListener('click', openHistory);
document.getElementById('historyClose').addEventListener('click', closeHistory);
document.getElementById('historyOverlay').addEventListener('click', closeHistory);

function openHistory() {
    document.getElementById('historyPanel').style.right = '0';
    document.getElementById('historyOverlay').style.display = 'block';
    loadHistory();
}

function closeHistory() {
    document.getElementById('historyPanel').style.right = '-420px';
    document.getElementById('historyOverlay').style.display = 'none';
}

async function loadHistory() {
    const res  = await fetch('/api/history?limit=10');
    const data = await res.json();
    const nights = data.nights || [];

    if (!nights.length) {
        document.getElementById('historyTable').innerHTML =
            '<p style="color:#6E8099; font-size:13px;">No history yet. Analyse a file first.</p>';
        return;
    }

    // ── Chart
    const labels = nights.map(n => n.date || n.session_id.slice(0,8));
    const scores = nights.map(n => n.quality_score);
    const efficiencies = nights.map(n => n.efficiency_pct);
    const remPcts = nights.map(n => n.rem_pct || 0);

    if (historyChartInstance) historyChartInstance.destroy();
    historyChartInstance = new Chart(document.getElementById('historyChart'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Quality Score',
                    data: scores,
                    borderColor: '#4DFFD2',
                    backgroundColor: 'rgba(77,255,210,0.08)',
                    tension: 0.3, fill: true, pointRadius: 4,
                },
                {
                    label: 'Efficiency %',
                    data: efficiencies,
                    borderColor: '#FFB347',
                    backgroundColor: 'transparent',
                    tension: 0.3, pointRadius: 3, borderDash: [4,3],
                },
                {
                    label: 'REM %',
                    data: remPcts,
                    borderColor: '#c792ea',
                    backgroundColor: 'transparent',
                    tension: 0.3, pointRadius: 3, borderDash: [2,3],
                },
            ],
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#6E8099', font: { size: 11 } } } },
            scales: {
                x: { ticks: { color: '#6E8099', font: { size: 11 } }, grid: { color: '#1A2840' } },
                y: { ticks: { color: '#6E8099', font: { size: 11 } }, grid: { color: '#1A2840' }, min: 0, max: 100 },
            },
        },
    });

    // ── Table
    document.getElementById('historyTable').innerHTML = `
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
            <thead>
                <tr style="border-bottom:1px solid #253347; color:#6E8099;">
                    <th style="padding:6px 8px; text-align:left;">Date</th>
                    <th style="padding:6px 8px; text-align:right;">Score</th>
                    <th style="padding:6px 8px; text-align:right;">Eff%</th>
                    <th style="padding:6px 8px; text-align:right;">REM%</th>
                    <th style="padding:6px 8px; text-align:right;">Awk</th>
                </tr>
            </thead>
            <tbody>
                ${nights.map(n => `
                    <tr style="border-bottom:1px solid rgba(37,51,71,0.4); cursor:pointer;"
                        onclick="loadSession('${n.session_id}'); closeHistory();">
                        <td style="padding:8px; color:#D4DDE8;">${n.date}</td>
                        <td style="padding:8px; text-align:right; color:#4DFFD2; font-weight:700;">${n.quality_score} ${n.quality_grade}</td>
                        <td style="padding:8px; text-align:right; color:#aab;">${n.efficiency_pct}%</td>
                        <td style="padding:8px; text-align:right; color:#aab;">${n.rem_pct || '—'}%</td>
                        <td style="padding:8px; text-align:right; color:#aab;">${n.awakenings}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    // ── Weekly summary
    document.getElementById('weeklySummary').textContent = 'Generating weekly summary…';
    try {
        const sr = await fetch('/api/weekly-summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nights }),
        });
        const sd = await sr.json();
        document.getElementById('weeklySummary').textContent = sd.summary || '';
    } catch (e) {
        document.getElementById('weeklySummary').textContent = '';
    }
}

// Load a specific past session by ID
async function loadSession(sessionId) {
    const res = await fetch(`/api/session/${sessionId}`);
    if (res.ok) {
        const data = await res.json();
        renderResults(data);
        if (data.chat_history) {
            data.chat_history.forEach(m => appendChatMessage(m.role, m.text));
        }
    }
}
```

---

### Verify

```bash
# 1. Run /api/demo twice to create 2 sessions
curl http://localhost:8000/api/demo > /dev/null
curl http://localhost:8000/api/demo > /dev/null

# 2. Check history endpoint
curl http://localhost:8000/api/history | python3 -m json.tool

# 3. Check weekly summary
curl -X POST http://localhost:8000/api/weekly-summary \
  -H "Content-Type: application/json" \
  -d '{"nights": [{"date":"2026-06-18","quality_score":72,"efficiency_pct":87,"rem_pct":21,"latency_min":15,"awakenings":3}]}'
```

---

## Feature 3: Dream Journal + REM Correlation

Users describe what they dreamed (text or voice via Deepgram mic). Claude correlates the dream
character (vivid, emotional, narrative richness) with their actual REM data: duration, timing
of REM epochs, and consecutive REM runs in the hypnogram.

Requires Redis (for storage) and `ANTHROPIC_API_KEY` (for analysis).

---

### Create `backend/dream_journal.py`

```python
"""
Dream journal storage and REM correlation via Claude.
Dreams are stored per session in Redis alongside the analysis.
"""

import json
import os
from typing import Optional

import anthropic


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic(api_key=api_key)


def _extract_rem_context(analysis: dict) -> dict:
    """
    Pull REM-specific stats from a full analysis dict.
    Returns a flat dict used in the Claude prompt.
    """
    summary = analysis.get("sleep_summary", {})
    epochs  = analysis.get("epoch_by_epoch_data", [])
    pcts    = summary.get("pct_in_stage", {})

    rem_pct   = next((v for k, v in pcts.items() if k.upper() == "REM"), 0.0)
    rem_min   = next((v for k, v in summary.get("time_in_stage_min", {}).items()
                      if k.upper() == "REM"), 0.0)

    # Find REM runs: start time and duration of each consecutive REM block
    rem_class_name = next((k for k in pcts if k.upper() == "REM"), "REM")
    rem_code = None
    if epochs:
        # Identify the integer code for REM from the first REM epoch
        for ep in epochs:
            if ep.get("predicted_stage_name", "").upper() == "REM":
                rem_code = ep.get("predicted_stage_code")
                break

    rem_runs = []
    if rem_code is not None:
        i = 0
        while i < len(epochs):
            if epochs[i].get("predicted_stage_code") == rem_code:
                start = epochs[i].get("timestamp", "")
                j = i
                while j < len(epochs) and epochs[j].get("predicted_stage_code") == rem_code:
                    j += 1
                run_min = round((j - i) * 0.5, 1)  # 30s epochs → 0.5 min each
                rem_runs.append({"start": start, "duration_min": run_min})
                i = j
            else:
                i += 1

    # Timing: first and last REM run
    first_rem = rem_runs[0]["start"] if rem_runs else "none"
    last_rem  = rem_runs[-1]["start"] if rem_runs else "none"
    longest_run = max((r["duration_min"] for r in rem_runs), default=0)

    return {
        "rem_pct":       rem_pct,
        "rem_min":       rem_min,
        "n_rem_runs":    len(rem_runs),
        "first_rem_at":  first_rem,
        "last_rem_at":   last_rem,
        "longest_rem_min": longest_run,
        "recording_start": analysis.get("metadata", {}).get("recording_start", ""),
    }


def analyze_dream(dream_text: str, analysis: dict) -> str:
    """
    Send dream description + REM context to Claude.
    Returns a 2–3 paragraph correlation analysis.
    """
    rem = _extract_rem_context(analysis)

    prompt = f"""The user described their dream:

"{dream_text}"

Their sleep data from last night:
- REM sleep: {rem['rem_pct']}% of total recording ({rem['rem_min']} minutes)
- Number of distinct REM periods: {rem['n_rem_runs']}
- First REM period began at: {rem['first_rem_at']}
- Longest single REM period: {rem['longest_rem_min']} minutes
- Last REM period started at: {rem['last_rem_at']}

Write a 2–3 paragraph response that:
1. Comments on the dream's characteristics (vivid vs. mundane, emotional vs. neutral, \
narrative complexity) and what those features tend to correlate with in sleep science.
2. Connects the dream to their actual REM data — timing, duration, number of REM periods. \
Note whether the dream characteristics match what we'd expect given the REM pattern. \
(e.g. vivid/emotional dreams often occur in the last REM period of the night which is longest.)
3. Offers one insight or takeaway — about their REM health, what the dream might reflect, \
or what could improve dream quality and REM depth.

Be warm, curious, and evidence-based. Avoid pseudo-scientific dream interpretation."""

    response = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def save_dream(session_id: str, dream_text: str, analysis_text: str) -> None:
    """Persist the dream and its analysis to Redis (or in-memory fallback)."""
    from backend import session as session_store   # avoid circular import
    payload = json.dumps({"dream": dream_text, "analysis": analysis_text})
    r = session_store._redis()
    if r:
        r.setex(f"dream:{session_id}", session_store.TTL, payload)
    else:
        session_store._memory_store[f"dream:{session_id}"] = payload


def load_dream(session_id: str) -> Optional[dict]:
    """Load stored dream + analysis for a session."""
    from backend import session as session_store
    r = session_store._redis()
    raw = (r.get(f"dream:{session_id}") if r
           else session_store._memory_store.get(f"dream:{session_id}"))
    return json.loads(raw) if raw else None
```

---

### `backend/main.py` — two new routes

Add import at top:
```python
from backend import dream_journal as dream_module
```

Add routes (after `/api/chat`):

```python
class DreamRequest(BaseModel):
    session_id: str
    text: str


@app.post("/api/dream")
async def submit_dream(req: DreamRequest):
    """
    Accept a dream description and the session's analysis.
    Returns Claude's REM correlation analysis.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Dream text cannot be empty.")

    # Load the analysis for this session to build REM context
    from backend import session as session_store
    analysis = session_store.load_analysis(req.session_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Session not found — analyse a file first.")

    try:
        correlation = dream_module.analyze_dream(req.text, analysis)
        dream_module.save_dream(req.session_id, req.text, correlation)
        return {"analysis": correlation}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dream/{session_id}")
async def get_dream(session_id: str):
    """Retrieve a previously submitted dream and its analysis."""
    data = dream_module.load_dream(session_id)
    if not data:
        return {"dream": None, "analysis": None}
    return data
```

---

### `frontend/index.html` — dream journal card

**Add this HTML block at the bottom of the results section, after the disorder alerts:**

```html
<div id="dreamSection" style="display:none; margin-top: 32px;">
    <h3 style="font-size:16px; color:#fff; margin-bottom:12px;">🌙 Dream Journal</h3>
    <p style="font-size:13px; color:#6E8099; margin-bottom:14px;">
        Describe what you dreamed about and Claude will correlate it with your REM data.
    </p>

    <div id="dreamInputArea">
        <textarea id="dreamText"
            placeholder="I was standing in a house I didn't recognise, and the rooms kept changing…"
            rows="4"
            style="
                width:100%; background:#0E1929; border:1px solid #253347;
                color:#D4DDE8; padding:12px; font-size:14px; border-radius:2px;
                resize:vertical; font-family: inherit; line-height:1.5;
            ">
        </textarea>
        <div style="display:flex; gap:10px; margin-top:10px;">
            <button id="dreamMicBtn" type="button" title="Hold to speak dream"
                style="padding:8px 14px; background:#172032; border:1px solid #253347;
                       color:#D4DDE8; border-radius:2px; cursor:pointer;">
                🎤 Speak
            </button>
            <button id="dreamSubmitBtn" type="button"
                style="flex:1; padding:8px 16px; background:#4DFFD2; color:#0B1120;
                       border:none; border-radius:2px; font-weight:700; cursor:pointer;">
                Analyse Dream
            </button>
        </div>
    </div>

    <div id="dreamResult" style="display:none; margin-top:16px;">
        <div id="dreamOriginalText" style="
            font-size:13px; color:#6E8099; font-style:italic;
            border-left:2px solid #253347; padding-left:12px; margin-bottom:14px;
        "></div>
        <div id="dreamAnalysisText" style="
            font-size:14px; color:#D4DDE8; line-height:1.7;
            background:rgba(77,255,210,0.04); border:1px solid #253347;
            border-radius:2px; padding:16px;
        "></div>
        <button id="dreamNewEntry" style="
            margin-top:10px; padding:6px 14px; background:none;
            border:1px solid #253347; color:#6E8099; border-radius:2px; cursor:pointer;
            font-size:12px;
        ">Write another entry</button>
    </div>
</div>
```

**JS — dream journal logic (add to script section):**

```javascript
// Show dream section after results are rendered
function showDreamSection(sessionId) {
    document.getElementById('dreamSection').style.display = 'block';
    document.getElementById('dreamSection').dataset.sessionId = sessionId;

    // Restore prior dream if it exists
    fetch(`/api/dream/${sessionId}`)
        .then(r => r.json())
        .then(data => {
            if (data.dream) showDreamResult(data.dream, data.analysis);
        });
}

// Call inside renderResults(data):
// showDreamSection(data.session_id);

document.getElementById('dreamSubmitBtn').addEventListener('click', async () => {
    const text      = document.getElementById('dreamText').value.trim();
    const sessionId = document.getElementById('dreamSection').dataset.sessionId;
    if (!text) return;

    document.getElementById('dreamSubmitBtn').textContent = 'Analysing…';
    document.getElementById('dreamSubmitBtn').disabled = true;

    try {
        const res = await fetch('/api/dream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, text }),
        });
        const data = await res.json();
        showDreamResult(text, data.analysis);
    } catch (e) {
        alert('Dream analysis failed. Check that ANTHROPIC_API_KEY is set.');
    } finally {
        document.getElementById('dreamSubmitBtn').textContent = 'Analyse Dream';
        document.getElementById('dreamSubmitBtn').disabled = false;
    }
});

function showDreamResult(dreamText, analysisText) {
    document.getElementById('dreamInputArea').style.display  = 'none';
    document.getElementById('dreamResult').style.display     = 'block';
    document.getElementById('dreamOriginalText').textContent = '"' + dreamText + '"';
    document.getElementById('dreamAnalysisText').textContent = analysisText;
}

document.getElementById('dreamNewEntry').addEventListener('click', () => {
    document.getElementById('dreamInputArea').style.display = 'block';
    document.getElementById('dreamResult').style.display    = 'none';
    document.getElementById('dreamText').value              = '';
});

// Voice input for dream (reuses Deepgram from sponsor_integrations.md)
// If Deepgram is integrated, wire dreamMicBtn the same way as chatMicBtn
// but target dreamText textarea instead of chatInput.
document.getElementById('dreamMicBtn').addEventListener('mousedown', async () => {
    if (typeof MediaRecorder === 'undefined') return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true }).catch(() => null);
    if (!stream) return;
    const chunks = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = e => chunks.push(e.data);
    rec.onstop = async () => {
        const fd = new FormData();
        fd.append('file', new Blob(chunks, { type: 'audio/webm' }), 'dream.webm');
        const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
        if (res.ok) {
            const { transcript } = await res.json();
            if (transcript) document.getElementById('dreamText').value = transcript;
        }
        stream.getTracks().forEach(t => t.stop());
    };
    rec.start();
    document.getElementById('dreamMicBtn').textContent = '⏹ Stop';
    document.getElementById('dreamMicBtn').onmouseup = () => {
        rec.stop();
        document.getElementById('dreamMicBtn').textContent = '🎤 Speak';
    };
});
```

---

### Verify

```bash
# 1. Get a session ID from demo
SID=$(curl -s http://localhost:8000/api/demo | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session: $SID"

# 2. Submit a dream
curl -X POST http://localhost:8000/api/dream \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SID\", \"text\": \"I was flying over a city I recognised but the streets kept rearranging. Very vivid colours.\"}"
# → {"analysis": "Your dream description suggests..."}

# 3. Retrieve stored dream
curl http://localhost:8000/api/dream/$SID
# → {"dream": "I was flying...", "analysis": "Your dream..."}
```

---

## All new files

| File | Action | Purpose |
|------|--------|---------|
| `backend/disorder_screener.py` | **Create** | Rule-based risk flags from sleep_summary |
| `backend/dream_journal.py` | **Create** | Dream storage + Claude REM correlation |
| `backend/session.py` | Edit | Add `record_session_index` and `get_recent_session_ids` |
| `backend/chat.py` | Edit | Add `generate_weekly_summary` |
| `backend/main.py` | Edit | Wire disorder_screen into /analyze, add 4 new routes |
| `frontend/index.html` | Edit | Disorder alert cards, history panel, dream journal card |

---

## New API routes summary

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/history` | Last 10 sessions (lightweight, no epoch data) |
| `POST` | `/api/weekly-summary` | Claude narrative from list of night dicts |
| `POST` | `/api/dream` | Submit dream text → Claude REM correlation |
| `GET` | `/api/dream/{session_id}` | Retrieve stored dream + analysis |
