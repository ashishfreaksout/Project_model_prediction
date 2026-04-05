// Config
Chart.defaults.font.family = 'Inter';

const THEME_STORAGE_KEY = 'nexusml-theme';

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function getThemeTokens() {
    return {
        textPrimary: cssVar('--text-primary'),
        textSecondary: cssVar('--text-secondary'),
        borderColor: cssVar('--border-color'),
        borderStrong: cssVar('--border-strong'),
        surfaceSoft: cssVar('--surface-soft'),
        surfaceSoftStrong: cssVar('--surface-soft-strong'),
        surfaceMuted: cssVar('--surface-muted'),
        surfacePrimarySoft: cssVar('--surface-primary-soft'),
        surfacePrimaryBorder: cssVar('--surface-primary-border'),
        surfaceSuccessSoft: cssVar('--surface-success-soft'),
        surfaceSuccessBorder: cssVar('--surface-success-border'),
        surfaceInfoSoft: cssVar('--surface-info-soft'),
        surfaceInfoBorder: cssVar('--surface-info-border'),
        surfaceWarningSoft: cssVar('--surface-warning-soft'),
        surfaceWarningBorder: cssVar('--surface-warning-border'),
        surfaceDangerSoft: cssVar('--surface-danger-soft'),
        surfaceDangerBorder: cssVar('--surface-danger-border'),
        primaryStrong: cssVar('--primary-strong'),
        accentStrong: cssVar('--accent-strong'),
        success: cssVar('--success'),
        warning: cssVar('--warning'),
        chartText: cssVar('--chart-text'),
        chartGrid: cssVar('--chart-grid'),
        chartTooltipBg: cssVar('--chart-tooltip-bg'),
        chartTooltipBorder: cssVar('--chart-tooltip-border'),
        chartTooltipText: cssVar('--chart-tooltip-text'),
        chartNeutralSoft: cssVar('--chart-neutral-soft'),
        chartPrimarySoft: cssVar('--chart-primary-soft'),
        chartPrimaryStrong: cssVar('--chart-primary-strong'),
        chartPositiveSoft: cssVar('--chart-positive-soft'),
        chartPositiveStrong: cssVar('--chart-positive-strong'),
        chartNegativeSoft: cssVar('--chart-negative-soft'),
        chartNegativeStrong: cssVar('--chart-negative-strong'),
        chartInfoSoft: cssVar('--chart-info-soft'),
        chartInfoStrong: cssVar('--chart-info-strong'),
        chartAmberSoft: cssVar('--chart-amber-soft'),
        chartLine: cssVar('--chart-line'),
    };
}

function syncChartTheme() {
    const theme = getThemeTokens();
    Chart.defaults.color = theme.chartText;
    Chart.defaults.borderColor = theme.chartGrid;
    Chart.defaults.plugins.legend.labels.color = theme.chartText;
    Chart.defaults.plugins.tooltip.backgroundColor = theme.chartTooltipBg;
    Chart.defaults.plugins.tooltip.titleColor = theme.chartTooltipText;
    Chart.defaults.plugins.tooltip.bodyColor = theme.chartTooltipText;
    Chart.defaults.plugins.tooltip.borderColor = theme.chartTooltipBorder;
    Chart.defaults.plugins.tooltip.borderWidth = 1;
}

function getPreferredTheme() {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function updateThemeToggle(theme) {
    const toggle = document.getElementById('theme-toggle');
    const label = document.getElementById('theme-toggle-label');
    const icon = document.getElementById('theme-toggle-icon');
    if (!toggle || !label || !icon) return;
    const isDark = theme === 'dark';
    label.innerText = isDark ? 'Dark Mode' : 'Light Mode';
    icon.innerText = isDark ? '☾' : '☀';
    toggle.setAttribute('aria-pressed', String(!isDark));
}

function refreshChartsForTheme() {
    if (!cachedEdaData && !lastTrainResultsData) return;
    if (!sections.analysis.classList.contains('hidden') && cachedEdaData) {
        renderEDACharts(cachedEdaData, cachedTargetSummary, cachedTargetCol);
    }
    if (!sections.results.classList.contains('hidden') && lastTrainResultsData) {
        renderResultsList(lastTrainResultsData);
    }
}

function applyTheme(theme, { persist = true, rerender = true } = {}) {
    document.documentElement.setAttribute('data-theme', theme);
    if (persist) localStorage.setItem(THEME_STORAGE_KEY, theme);
    syncChartTheme();
    updateThemeToggle(theme);
    if (rerender) refreshChartsForTheme();
}

function initTheme() {
    applyTheme(getPreferredTheme(), { persist: false, rerender: false });
    const toggle = document.getElementById('theme-toggle');
    toggle?.addEventListener('click', () => {
        const nextTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        applyTheme(nextTheme);
    });
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (event) => {
        if (!localStorage.getItem(THEME_STORAGE_KEY)) {
            applyTheme(event.matches ? 'light' : 'dark', { persist: false });
        }
    });
}

// State
let sessionId = null;
let datasetCols = [];
let datasetTypes = {};
let taskType = null;
let charts = {};
let predictionSchema = null;

// DOM Elements
const sections = {
    upload: document.getElementById('section-upload'),
    eda: document.getElementById('section-eda'),
    analysis: document.getElementById('section-analysis'),
    results: document.getElementById('section-results'),
    predict: document.getElementById('section-predict')
};

const loader = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');
const loaderSubtext = document.getElementById('loader-subtext');
let loaderCycleTimer = null;
let loaderFadeTimer = null;
let loaderCycleIndex = 0;
let candidateModelNames = [];

initTheme();

// Helper - Show Loader
function stopLoaderCycle() {
    if (loaderCycleTimer) {
        clearInterval(loaderCycleTimer);
        loaderCycleTimer = null;
    }
    if (loaderFadeTimer) {
        clearTimeout(loaderFadeTimer);
        loaderFadeTimer = null;
    }
    loaderCycleIndex = 0;
    loaderSubtext?.classList.remove('is-fading');
}

function startLoaderCycle(lines, interval = 1800) {
    if (!loaderSubtext || !Array.isArray(lines) || !lines.length) return;

    stopLoaderCycle();
    loaderSubtext.innerText = lines[0];
    loaderCycleIndex = 0;

    if (lines.length === 1) return;

    loaderCycleTimer = window.setInterval(() => {
        loaderSubtext.classList.add('is-fading');
        loaderFadeTimer = window.setTimeout(() => {
            loaderCycleIndex = (loaderCycleIndex + 1) % lines.length;
            loaderSubtext.innerText = lines[loaderCycleIndex];
            loaderSubtext.classList.remove('is-fading');
            loaderFadeTimer = null;
        }, 220);
    }, interval);
}

function showLoader(msg = "Processing...", options = {}) {
    const {
        detail = 'Preparing the workflow.',
        cycle = null
    } = options;

    loader.classList.remove('hidden');
    loaderText.innerText = msg;
    if (loaderSubtext) {
        loaderSubtext.innerText = detail;
    }
    if (Array.isArray(cycle) && cycle.length) {
        startLoaderCycle(cycle);
    } else {
        stopLoaderCycle();
    }
}
function hideLoader() {
    stopLoaderCycle();
    loader.classList.add('hidden');
}

// Helper - Change Section
function showSection(sectionId) {
    Object.values(sections).forEach(s => s.classList.add('hidden'));
    sections[sectionId].classList.remove('hidden');
    // Scroll top
    window.scrollTo({top: 0, behavior: 'smooth'});
}

function updateLogTransformOption(targetSummary = {}, currentTaskType = null) {
    const optionEl = document.getElementById('log-transform-option');
    const checkbox = document.getElementById('use-log-target');
    const helpEl = document.getElementById('log-transform-help');
    if (!optionEl || !checkbox || !helpEl) return;

    const supported = Boolean(targetSummary?.log_transform_supported);
    const suggested = Boolean(targetSummary?.log_transform_suggested);

    if (currentTaskType === 'regression' && supported) {
        optionEl.classList.remove('hidden');
        checkbox.disabled = false;
        checkbox.checked = suggested;
        helpEl.innerText = suggested
            ? 'Recommended for this target because it is positive and highly right-skewed. Metrics remain reported on the original target scale.'
            : 'Optional for positive regression targets. Use this when reducing right-skew and large-value compression is more important than fitting on the raw scale.';
        return;
    }

    checkbox.checked = false;
    checkbox.disabled = true;
    optionEl.classList.add('hidden');
}

document.getElementById('brand-home').addEventListener('click', () => {
    hideLoader();
    showSection('upload');
});

// 1. Upload Logic
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault(); dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleUpload(e.target.files[0]);
});

async function handleUpload(file) {
    if (!file.name.endsWith('.csv')) { alert('Please upload a CSV file.'); return; }
    
    showLoader("Ingesting dataset and computing metrics...", {
        detail: 'Scanning headers, shape, missingness, and preview rows.'
    });
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        
        if (res.status === 413) {
            throw new Error("File exceeds the maximum allowed size of 200MB.");
        }
        
        let data;
        try {
            data = await res.json();
        } catch(parseErr) {
            throw new Error(`Server returned an invalid response (Status: ${res.status}).`);
        }
        
        if (data.error) throw new Error(data.error);
        
        sessionId = data.session_id;
        datasetCols = data.columns;
        datasetTypes = data.types;
        
        // Populate Data Preview
        if (data.head && data.head.length > 0) {
            let tableHTML = '<thead><tr>';
            datasetCols.forEach(col => { 
                tableHTML += `<th style="padding: 1rem; text-align: left;">${col}<br><span style="background:var(--surface-soft-strong); padding:0.1rem 0.4rem; border-radius:4px; font-size:0.65rem; font-weight:normal; opacity:0.8;">${data.types[col]}</span></th>`; 
            });
            tableHTML += '</tr></thead><tbody>';
            data.head.forEach(row => {
                tableHTML += '<tr>';
                datasetCols.forEach(col => { tableHTML += `<td style="padding: 0.75rem 1rem; border-bottom:1px solid var(--border-subtle);">${row[col] !== null ? row[col] : ''}</td>`; });
                tableHTML += '</tr>';
            });
            tableHTML += '</tbody>';
            document.getElementById('data-preview-table').innerHTML = tableHTML;
        }
        
        // Populate EDA Stats
        document.getElementById('stat-rows').innerText = data.rows.toLocaleString();
        document.getElementById('stat-cols').innerText = data.columns.length.toLocaleString();
        document.getElementById('stat-num').innerText = (data.numeric_cols || 0).toLocaleString();
        document.getElementById('stat-cat').innerText = (data.categorical_cols || 0).toLocaleString();
        
        const totalMissing = Object.values(data.missing).reduce((a, b) => a + b, 0);
        const totalCells = data.rows * data.columns.length;
        const missingPct = totalCells > 0 ? ((totalMissing / totalCells) * 100).toFixed(1) : '0.0';
        const missEl = document.getElementById('stat-missing');
        missEl.innerText = totalMissing.toLocaleString();
        if (totalMissing === 0) { missEl.style.color = "var(--success)"; }
        document.getElementById('stat-missing-pct').innerText = `(${missingPct}%)`;
        
        document.getElementById('stat-duplicates').innerText = (data.duplicates || 0).toLocaleString() + " rows";
        if (data.high_missing_cols && data.high_missing_cols.length > 0) {
            document.getElementById('high-missing-alert').style.display = 'block';
            document.getElementById('high-missing-text').innerText = `${data.high_missing_cols.length} column(s) exceed 30% missing data`;
        } else {
            document.getElementById('high-missing-alert').style.display = 'none';
        }
        
        // Populate Target Dropdown
        const targetSelect = document.getElementById('target-select');
        targetSelect.innerHTML = '';
        datasetCols.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.innerText = `${col} (${data.types[col]})`;
            targetSelect.appendChild(opt);
        });
        
        // Populate Target Suggestions
        const stContainer = document.getElementById('suggested-targets-container');
        stContainer.innerHTML = '';
        if (data.suggested_targets && data.suggested_targets.length > 0) {
            data.suggested_targets.forEach(t => {
                const btn = document.createElement('button');
                btn.className = 'btn target-chip';
                btn.dataset.column = t.column;
                btn.style.cssText = 'padding:0.45rem 1rem; border-radius:20px; font-size:0.85rem; background:var(--surface-soft); border:1px solid var(--border-color); cursor:pointer; color:var(--text-primary); transition:all 0.2s ease;';
                btn.innerHTML = `<strong>${t.column}</strong> <span style="opacity:0.7; font-size:0.7rem; margin-left:6px; padding:0.15rem 0.4rem; background:var(--surface-primary-soft); border-radius:4px; color:var(--primary-strong);">${t.task}</span>`;
                btn.onmouseenter = () => { if (!btn.classList.contains('chip-active')) btn.style.borderColor = 'var(--border-strong)'; };
                btn.onmouseleave = () => { if (!btn.classList.contains('chip-active')) btn.style.borderColor = 'var(--border-color)'; };
                btn.onclick = () => {
                    targetSelect.value = t.column;
                    highlightActiveChip(t.column);
                    fetchTargetInsight(t.column);
                };
                stContainer.appendChild(btn);
            });
        } else {
            stContainer.innerHTML = '<span style="color:var(--text-secondary); font-size:0.8rem;">No suggested targets detected. Please select from the dropdown below.</span>';
        }
        
        // Bind auto-fetch
        targetSelect.addEventListener('change', (e) => {
            highlightActiveChip(e.target.value);
            fetchTargetInsight(e.target.value);
        });
        
        hideLoader();
        showSection('eda');
        
        // Trigger first insight automatically
        if (datasetCols.length > 0) fetchTargetInsight(targetSelect.value);
        
    } catch (e) {
        hideLoader();
        alert(e.message);
    }
}

// 1C. Chip Highlight Logic
function highlightActiveChip(colName) {
    document.querySelectorAll('.target-chip').forEach(chip => {
        if (chip.dataset.column === colName) {
            chip.classList.add('chip-active');
            chip.style.background = 'var(--surface-primary-soft)';
            chip.style.borderColor = 'var(--surface-primary-border)';
            chip.style.boxShadow = 'var(--shadow-primary)';
        } else {
            chip.classList.remove('chip-active');
            chip.style.background = 'var(--surface-soft)';
            chip.style.borderColor = 'var(--border-color)';
            chip.style.boxShadow = 'none';
        }
    });
}

// 1B. Target Insight Fetch Logic
async function fetchTargetInsight(targetCol) {
    document.getElementById('target-insights-card').style.opacity = '1';
    document.getElementById('target-insights-placeholder').classList.add('hidden');
    const content = document.getElementById('target-insights-content');
    content.classList.remove('hidden');
    
    document.getElementById('ti-column-name').innerText = targetCol;
    document.getElementById('ti-task-type').innerText = 'Analyzing...';
    
    try {
        const res = await fetch('/api/target_insight', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId, target_column: targetCol })
        });
        const tData = await res.json();
        if (tData.error) throw new Error(tData.error);
        
        document.getElementById('ti-task-type').innerText = tData.task_type.toUpperCase();
        
        document.getElementById('ti-stat-1').innerHTML = `<span style="color:var(--text-secondary); font-size:0.75rem; display:block; margin-bottom:0.15rem;">Missing Values</span><strong style="font-size:1.1rem;">${tData.target_summary.n_missing}</strong>`;
        document.getElementById('ti-stat-2').innerHTML = `<span style="color:var(--text-secondary); font-size:0.75rem; display:block; margin-bottom:0.15rem;">Unique Values</span><strong style="font-size:1.1rem;">${tData.target_summary.n_unique}</strong>`;
        
        if (tData.task_type === 'regression') {
            document.getElementById('ti-stat-3').innerHTML = `<span style="color:var(--text-secondary); font-size:0.75rem; display:block; margin-bottom:0.15rem;">Mean</span><strong style="font-size:1.1rem;">${tData.target_summary.mean ? tData.target_summary.mean.toFixed(2) : '-'}</strong>`;
            document.getElementById('ti-stat-4').innerHTML = `<span style="color:var(--text-secondary); font-size:0.75rem; display:block; margin-bottom:0.15rem;">Median</span><strong style="font-size:1.1rem;">${tData.target_summary.median ? tData.target_summary.median.toFixed(2) : '-'}</strong>`;
        } else {
            const classes = Object.keys(tData.target_summary.class_counts || {}).length;
            let imbalance = classes > 2 ? "Multi-class" : "Binary";
            document.getElementById('ti-stat-3').innerHTML = `<span style="color:var(--text-secondary); font-size:0.75rem; display:block; margin-bottom:0.15rem;">Classes Detected</span><strong style="font-size:1.1rem;">${classes}</strong>`;
            document.getElementById('ti-stat-4').innerHTML = `<span style="color:var(--text-secondary); font-size:0.75rem; display:block; margin-bottom:0.15rem;">Distribution</span><strong style="font-size:1.1rem;">${imbalance}</strong>`;
        }
        
    } catch (e) {
        console.error("Target Insight Plugin Error:", e);
    }
}

// 2. Analyze Logic
document.getElementById('btn-analyze').addEventListener('click', async () => {
    const target = document.getElementById('target-select').value;
    showLoader("Configuring AutoML Engine and producing insights...", {
        detail: 'Profiling the selected target and preparing candidate models.'
    });
    
    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId, target_column: target })
        });
        const data = await res.json();
        
        if (data.error) throw new Error(data.error);
        
        taskType = data.task_type;
        
        // --- Hero Header ---
        document.getElementById('analysis-title').innerText = taskType === 'regression' ? 'Regression Analysis Ready' : 'Classification Analysis Ready';
        document.getElementById('analysis-target-name').innerText = data.target || target;
        document.getElementById('analysis-task-badge').innerText = taskType.toUpperCase();
        document.getElementById('btn-train').innerText = taskType === 'regression' ? 'Train Regression Models' : 'Train Classification Models';
        
        const ts = data.target_summary || {};
        document.getElementById('analysis-usable-rows').innerText = (ts.usable_rows || '—').toLocaleString();
        
        // --- Target Summary Card ---
        document.getElementById('ts-target-label').innerText = data.target || target;
        const tsGrid = document.getElementById('target-summary-grid');
        tsGrid.innerHTML = '';
        
        function addStat(label, value) {
            tsGrid.innerHTML += `<div style="background:var(--surface-muted); padding:0.6rem 0.75rem; border-radius:8px;">
                <span style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-secondary); display:block; margin-bottom:0.15rem;">${label}</span>
                <strong style="font-size:1rem;">${value}</strong>
            </div>`;
        }
        
        addStat('Missing', ts.n_missing ?? '—');
        addStat('Usable Rows', (ts.usable_rows || '—').toLocaleString());
        
        if (taskType === 'regression') {
            addStat('Mean', ts.mean != null ? ts.mean.toLocaleString(undefined, {maximumFractionDigits:2}) : '—');
            addStat('Median', ts.median != null ? ts.median.toLocaleString(undefined, {maximumFractionDigits:2}) : '—');
            addStat('Min', ts.min != null ? ts.min.toLocaleString(undefined, {maximumFractionDigits:2}) : '—');
            addStat('Max', ts.max != null ? ts.max.toLocaleString(undefined, {maximumFractionDigits:2}) : '—');
            addStat('Std Dev', ts.std != null ? ts.std.toLocaleString(undefined, {maximumFractionDigits:2}) : '—');
            addStat('Skewness', ts.skewness != null ? ts.skewness.toFixed(2) : '—');
            addStat('Outliers (IQR)', ts.outlier_count ?? '—');
        } else {
            const cc = ts.class_counts || {};
            addStat('Classes', Object.keys(cc).length);
            addStat('Unique Values', ts.n_unique || '—');
            const entries = Object.entries(cc).slice(0, 4);
            entries.forEach(([k, v]) => addStat(k, v));
        }
        
        // Recommendation
        const recEl = document.getElementById('ts-recommendation');
        if (ts.log_transform_suggested) {
            recEl.classList.remove('hidden');
            document.getElementById('ts-rec-text').innerText = 'The target is highly skewed. Applying a log transform before training may significantly improve model accuracy.';
        } else {
            recEl.classList.add('hidden');
        }

        updateLogTransformOption(ts, taskType);
        
        // --- Data Quality Warnings ---
        const warnContainer = document.getElementById('warnings-container');
        const warnings = (data.eda && data.eda.warnings) || [];
        if (warnings.length > 0) {
            warnContainer.innerHTML = '';
            warnings.forEach(w => {
                const icon = w.severity === 'warning' ? '⚠️' : 'ℹ️';
                const bgColor = w.severity === 'warning' ? 'var(--surface-danger-soft)' : 'var(--surface-info-soft)';
                const borderColor = w.severity === 'warning' ? 'var(--surface-danger-border)' : 'var(--surface-info-border)';
                warnContainer.innerHTML += `<div style="padding:0.6rem 0.75rem; background:${bgColor}; border:1px solid ${borderColor}; border-radius:8px; font-size:0.85rem; color:var(--text-secondary);">
                    <span style="margin-right:0.4rem;">${icon}</span>${w.message}
                </div>`;
            });
        } else {
            warnContainer.innerHTML = '<div style="padding:0.6rem 0.75rem; background:var(--surface-success-soft); border:1px solid var(--surface-success-border); border-radius:8px; font-size:0.85rem; color:var(--success);">✅ No data quality issues detected. Dataset looks clean.</div>';
        }
        
        // --- Preprocessing Plan ---
        const ppList = document.getElementById('preprocess-list');
        const plan = data.preprocessing_plan || {};
        const planLabels = {
            target_transform: 'Target Transform',
            downsampling: 'Downsampling',
            numeric_imputation: 'Numeric Missing',
            categorical_imputation: 'Categorical Missing',
            encoding: 'Feature Encoding',
            scaling: 'Feature Scaling',
            validation: 'Validation Strategy',
            outlier_handling: 'Outlier Handling',
            duplicate_handling: 'Duplicates',
        };
        ppList.innerHTML = '';
        Object.entries(planLabels).forEach(([key, label]) => {
            if (plan[key]) {
                ppList.innerHTML += `<div style="display:flex; align-items:flex-start; gap:0.5rem; font-size:0.85rem;">
                    <span style="color:var(--success); flex-shrink:0; margin-top:2px;">✓</span>
                    <div><strong style="color:var(--text-primary);">${label}:</strong> <span style="color:var(--text-secondary);">${plan[key]}</span></div>
                </div>`;
            }
        });
        
        // --- Candidate Models ---
        const cmList = document.getElementById('candidate-models-list');
        const candidates = data.candidate_models || [];
        candidateModelNames = candidates.map(model => model.name).filter(Boolean);
        cmList.innerHTML = '';
        const tagColors = {
            baseline: {bg: 'var(--surface-soft)', border: 'var(--border-color)', color: 'var(--text-secondary)'},
            interpretable: {bg: 'var(--surface-info-soft)', border: 'var(--surface-info-border)', color: 'var(--accent-strong)'},
            nonlinear: {bg: 'var(--surface-success-soft)', border: 'var(--surface-success-border)', color: 'var(--success)'},
            ensemble: {bg: 'var(--surface-warning-soft)', border: 'var(--surface-warning-border)', color: 'var(--amber)'},
            recommended: {bg: 'var(--surface-primary-soft)', border: 'var(--surface-primary-border)', color: 'var(--primary-strong)'},
        };
        candidates.forEach(m => {
            const tc = tagColors[m.tag] || tagColors.baseline;
            cmList.innerHTML += `<div style="padding:0.5rem 0.85rem; background:${tc.bg}; border:1px solid ${tc.border}; border-radius:10px; font-size:0.85rem; display:flex; align-items:center; gap:0.5rem;">
                <strong style="color:var(--text-primary);">${m.name}</strong>
                <span style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.5px; color:${tc.color}; opacity:0.9;">${m.tag}</span>
            </div>`;
        });
        
        hideLoader();
        showSection('analysis');
        renderEDACharts(data.eda, ts, target);
        
    } catch (e) {
        hideLoader();
        const msg = e && e.message === 'Failed to fetch'
            ? 'Training request could not complete. The local server likely timed out or restarted while fitting models. Please retry after the backend finishes restarting.'
            : e.message;
        alert(msg);
    }
});

// Render EDA Charts
let cachedEdaData = null;
let cachedTargetCol = null;
let cachedTargetSummary = null;
let showAllCorrelations = false;
let lastTrainResultsData = null;

function formatCompactNumber(value, maximumFractionDigits = 1) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits
    }).format(Number(value));
}

function formatFullNumber(value, maximumFractionDigits = 0) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toLocaleString('en-US', { maximumFractionDigits });
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function setChartBodyMinHeight(canvasId, minHeight) {
    const canvas = document.getElementById(canvasId);
    const body = canvas?.closest('.chart-card-body');
    if (!body) return;
    body.style.minHeight = `${Math.round(minHeight)}px`;
}

function formatPredictionDefaultValue(value) {
    if (value == null || value === '') return 'blank';
    if (typeof value === 'number') {
        return Number.isInteger(value) ? `${value}` : Number(value).toFixed(Math.abs(value) >= 100 ? 0 : 2);
    }
    return String(value);
}

function coercePredictionFieldValue(field, rawValue) {
    if (rawValue == null) return undefined;
    const text = String(rawValue).trim();
    if (text === '') return undefined;

    if (field.value_type === 'boolean') {
        return text === 'true';
    }
    if (field.value_type === 'number') {
        const numeric = Number(text);
        return Number.isFinite(numeric) ? numeric : undefined;
    }
    return rawValue;
}

function createPredictionInput(field) {
    const group = document.createElement('div');
    group.className = 'input-group';

    const label = document.createElement('label');
    label.setAttribute('for', `input-${field.name}`);
    label.innerText = field.name;

    const hint = document.createElement('span');
    hint.className = 'field-hint';
    hint.innerText = `Optional. Blank uses ${field.default_label} (${formatPredictionDefaultValue(field.default_value)}).`;

    let control;
    if (field.kind === 'select') {
        control = document.createElement('select');
        const blankOption = document.createElement('option');
        blankOption.value = '';
        blankOption.innerText = `Use default (${formatPredictionDefaultValue(field.default_value)})`;
        control.appendChild(blankOption);
        field.options.forEach(option => {
            const optionEl = document.createElement('option');
            optionEl.value = String(option.value);
            optionEl.innerText = option.label;
            control.appendChild(optionEl);
        });
    } else {
        control = document.createElement('input');
        control.type = field.kind === 'number' ? 'number' : 'text';
        if (field.kind === 'number') control.step = 'any';
        control.placeholder = `Default: ${formatPredictionDefaultValue(field.default_value)}`;
    }

    control.name = field.name;
    control.id = `input-${field.name}`;
    control.dataset.valueType = field.value_type;
    control.dataset.kind = field.kind;

    group.appendChild(label);
    group.appendChild(control);
    group.appendChild(hint);
    return group;
}

function renderPredictionForm(schema) {
    const form = document.getElementById('prediction-form');
    const summary = document.getElementById('prediction-form-summary');
    form.innerHTML = '';

    if (!schema || !Array.isArray(schema.fields) || !schema.fields.length) {
        form.innerHTML = '<div class="chart-empty-state">Prediction inputs are not available for this trained model.</div>';
        summary.classList.add('hidden');
        summary.innerHTML = '';
        return;
    }

    const keyFields = schema.fields.filter(field => field.is_key);
    const advancedFields = schema.fields.filter(field => !field.is_key);

    summary.classList.remove('hidden');
    summary.innerHTML = `
        <div class="prediction-summary-grid">
            <div class="prediction-summary-card">
                <span class="prediction-summary-label">Key Inputs</span>
                <strong class="prediction-summary-value">${keyFields.length}</strong>
            </div>
            <div class="prediction-summary-card">
                <span class="prediction-summary-label">Advanced Inputs</span>
                <strong class="prediction-summary-value">${advancedFields.length}</strong>
            </div>
            <div class="prediction-summary-card">
                <span class="prediction-summary-label">Excluded Columns</span>
                <strong class="prediction-summary-value">${(schema.excluded_columns || []).length}</strong>
            </div>
        </div>
        <p class="prediction-form-note">Only model-used fields are shown. Key inputs are ranked by model influence when available. Low-cardinality categorical variables are rendered as dropdowns, and blank fields fall back to training defaults or imputers.</p>
    `;

    const keySection = document.createElement('div');
    keySection.className = 'prediction-form-section';
    keySection.innerHTML = '<div class="prediction-section-head"><h3>Key Inputs</h3><span class="prediction-section-note">Most influential model inputs</span></div>';
    const keyGrid = document.createElement('div');
    keyGrid.className = 'prediction-grid-inner';
    keyFields.forEach(field => keyGrid.appendChild(createPredictionInput(field)));
    keySection.appendChild(keyGrid);
    form.appendChild(keySection);

    if (advancedFields.length > 0) {
        const advancedSection = document.createElement('details');
        advancedSection.className = 'prediction-advanced';
        advancedSection.innerHTML = `
            <summary>Additional Optional Inputs <span>${advancedFields.length} field(s)</span></summary>
        `;
        const advancedGrid = document.createElement('div');
        advancedGrid.className = 'prediction-grid-inner';
        advancedFields.forEach(field => advancedGrid.appendChild(createPredictionInput(field)));
        advancedSection.appendChild(advancedGrid);
        form.appendChild(advancedSection);
    }

    const btnGroup = document.createElement('div');
    btnGroup.className = 'prediction-submit-row';
    btnGroup.innerHTML = '<button type="submit" class="btn btn-primary">Predict Now</button>';
    form.appendChild(btnGroup);
}

function getSignedCorrelationEntries(edaData, targetCol) {
    const matrix = edaData?.correlations?.matrix || {};
    const direct = matrix[targetCol];
    if (direct) {
        return Object.entries(direct)
            .filter(([feature, value]) => feature !== targetCol && Number.isFinite(value))
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    }

    return Object.entries(edaData?.correlations?.target_relationships || {})
        .filter(([, value]) => Number.isFinite(value))
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
}

function renderTargetDistributionMeta(edaData, targetSummary) {
    const metaEl = document.getElementById('target-distribution-meta');
    if (taskType !== 'regression' || !targetSummary) {
        metaEl.classList.add('hidden');
        metaEl.innerHTML = '';
        return;
    }

    const box = edaData?.target_boxplot;
    if (!box) {
        metaEl.classList.add('hidden');
        metaEl.innerHTML = '';
        return;
    }

    const valuesForScale = [box.min, box.q1, box.median, box.q3, box.p99, box.max]
        .map(v => Number(v))
        .filter(v => Number.isFinite(v) && v >= 0);
    const maxValue = Math.max(...valuesForScale, 1);
    const logMax = Math.log1p(maxValue);
    const position = (value) => {
        const safeValue = Number.isFinite(Number(value)) ? Math.max(Number(value), 0) : 0;
        return clamp((Math.log1p(safeValue) / logMax) * 100, 0, 100);
    };

    const whiskerStart = position(box.lower_whisker ?? box.min);
    const whiskerEnd = position(box.upper_whisker ?? box.q3);
    const boxStart = position(box.q1);
    const boxEnd = position(box.q3);
    const medianPos = position(box.median);
    const p99Pos = position(box.p99);
    const maxPos = position(box.max);

    metaEl.classList.remove('hidden');
    metaEl.innerHTML = `
        <div class="target-metrics-row">
            <span class="target-pill">Skewness <strong>${Number(targetSummary.skewness ?? 0).toFixed(2)}</strong></span>
            <span class="target-pill">IQR Outliers <strong>${formatFullNumber(box.outlier_count ?? targetSummary.outlier_count ?? 0)}</strong></span>
            <span class="target-pill">P99 <strong>${formatCompactNumber(targetSummary.p99)}</strong></span>
            <span class="target-pill">Max <strong>${formatCompactNumber(targetSummary.max)}</strong></span>
            <span class="target-pill warning">Min <strong>${formatFullNumber(targetSummary.min)}</strong></span>
        </div>
        <div class="target-boxplot-title">
            <span>Compact Outlier View</span>
            <span>log-spaced scale</span>
        </div>
        <div class="target-boxplot-track">
            <div class="target-boxplot-baseline"></div>
            <div class="target-boxplot-whisker" style="left:${whiskerStart}%; width:${Math.max(whiskerEnd - whiskerStart, 1)}%;"></div>
            <div class="target-boxplot-box" style="left:${boxStart}%; width:${Math.max(boxEnd - boxStart, 1)}%;"></div>
            <div class="target-boxplot-marker median" style="left:${medianPos}%;"></div>
            <div class="target-boxplot-marker p99" style="left:${p99Pos}%;"></div>
            <div class="target-boxplot-marker max" style="left:${maxPos}%;"></div>
        </div>
        <div class="target-boxplot-labels">
            <span>Q1 <strong>${formatCompactNumber(box.q1)}</strong></span>
            <span>Median <strong>${formatCompactNumber(box.median)}</strong></span>
            <span>Q3 <strong>${formatCompactNumber(box.q3)}</strong></span>
            <span>P99 <strong>${formatCompactNumber(box.p99)}</strong></span>
            <span>Max <strong>${formatCompactNumber(box.max)}</strong></span>
        </div>
    `;
}

function renderEDACharts(edaData, targetSummary, targetCol) {
    cachedEdaData = edaData;
    cachedTargetCol = targetCol;
    cachedTargetSummary = targetSummary;
    showAllCorrelations = false;
    if (charts.target) charts.target.destroy();
    if (charts.corr) charts.corr.destroy();
    
    // Determine if log histogram is available
    const hasLogHist = edaData.target_histogram_log && edaData.target_histogram_log.bins;
    const toggleContainer = document.getElementById('hist-toggle-container');
    
    if (taskType === 'regression' && hasLogHist) {
        toggleContainer.classList.remove('hidden');
        toggleContainer.style.display = 'flex';
        // Show log by default for skewed targets
        renderTargetHistogram(edaData, 'log');
        setHistToggleState('log');
    } else if (taskType === 'regression' && edaData.target_histogram && edaData.target_histogram.bins) {
        toggleContainer.classList.add('hidden');
        toggleContainer.style.display = 'none';
        renderTargetHistogram(edaData, 'raw');
    } else if (taskType === 'classification' && Object.keys(edaData.target_distribution || {}).length > 0) {
        toggleContainer.classList.add('hidden');
        toggleContainer.style.display = 'none';
        document.getElementById('target-chart-note').innerText = 'Frequency of the target variable within the dataset.';
        const ctxTarget = document.getElementById('targetChart').getContext('2d');
        const labels = Object.keys(edaData.target_distribution);
        const values = Object.values(edaData.target_distribution);
        const theme = getThemeTokens();
        setChartBodyMinHeight('targetChart', labels.length > 8 ? 340 : 310);
        charts.target = new Chart(ctxTarget, {
            type: 'bar',
            data: { labels, datasets: [{ label: 'Count', data: values, backgroundColor: theme.chartPrimarySoft, borderColor: theme.chartPrimaryStrong, borderWidth: 1, borderRadius: 4 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { top: 8, right: 10, bottom: 16, left: 6 } },
                plugins: { legend: { display: false } }
            }
        });
    } else {
        toggleContainer.classList.add('hidden');
        toggleContainer.style.display = 'none';
        document.getElementById('target-chart-note').innerText = 'Target distribution is not available.';
    }

    renderTargetDistributionMeta(edaData, targetSummary);
    
    // Bind toggle buttons
    document.getElementById('btn-hist-raw').onclick = () => { renderTargetHistogram(cachedEdaData, 'raw'); setHistToggleState('raw'); };
    document.getElementById('btn-hist-log').onclick = () => { renderTargetHistogram(cachedEdaData, 'log'); setHistToggleState('log'); };
    
    // Correlation Chart
    const signedEntries = getSignedCorrelationEntries(edaData, targetCol);
    if (signedEntries.length > 0) {
         renderCorrelationChart(signedEntries);
    }
}

function renderCorrelationChart(signedEntries) {
    const theme = getThemeTokens();
    const canvasCorr = document.getElementById('correlationChart');
    const scrollEl = document.getElementById('correlation-chart-scroll');
    const controlsEl = document.getElementById('correlation-controls');
    const toggleBtn = document.getElementById('btn-corr-toggle');
    const ctxCorr = canvasCorr.getContext('2d');
    const defaultVisibleCount = 8;
    const maxRenderableRows = 14;
    const rowHeight = 34;
    const chartPadding = 68;

    const visibleEntries = showAllCorrelations
        ? signedEntries.slice(0, maxRenderableRows)
        : signedEntries.slice(0, defaultVisibleCount);
    const fullLabels = visibleEntries.map(([feature]) => feature);
    const vals = visibleEntries.map(([, value]) => value);
    const maxAbs = Math.max(...vals.map(v => Math.abs(v)), 0.2);
    const axisLimit = Math.min(1, Math.max(0.25, maxAbs * 1.18));
    const maxLabelChars = fullLabels.reduce((max, label) => Math.max(max, label.length), 0);
    const yAxisWidth = Math.min(260, Math.max(150, maxLabelChars * 7.4));
    const desiredHeight = Math.max(220, visibleEntries.length * rowHeight + chartPadding);
    const scrollHeight = Math.min(desiredHeight, 420);
    const hasMore = signedEntries.length > defaultVisibleCount;
    const hiddenCount = Math.max(signedEntries.length - visibleEntries.length, 0);

    if (charts.corr) charts.corr.destroy();

    canvasCorr.height = desiredHeight;
    canvasCorr.style.height = `${desiredHeight}px`;
    scrollEl.style.height = `${scrollHeight}px`;
    scrollEl.style.maxHeight = '420px';
    scrollEl.scrollTop = 0;

    if (hasMore) {
        controlsEl.classList.remove('hidden');
        toggleBtn.textContent = showAllCorrelations
            ? `Show Top ${defaultVisibleCount}`
            : `Show ${Math.min(maxRenderableRows, signedEntries.length)} Features`;
        toggleBtn.onclick = () => {
            showAllCorrelations = !showAllCorrelations;
            renderCorrelationChart(signedEntries);
        };
    } else {
        controlsEl.classList.add('hidden');
        toggleBtn.onclick = null;
    }

    if (hiddenCount > 0) {
        toggleBtn.title = `${hiddenCount} additional correlated features remain hidden for readability.`;
    } else {
        toggleBtn.title = '';
    }

    charts.corr = new Chart(ctxCorr, {
        type: 'bar',
        data: {
            labels: fullLabels,
            datasets: [{
                label: 'Correlation',
                data: vals,
                backgroundColor: vals.map(v => v > 0 ? theme.chartPositiveSoft : theme.chartNegativeSoft),
                borderRadius: 5,
                barPercentage: 0.72,
                categoryPercentage: 0.82
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            animation: false,
            layout: { padding: { left: 8, right: 44 } },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => items[0].label,
                        label: (ctx) => `Pearson r = ${ctx.raw.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    min: -axisLimit,
                    max: axisLimit,
                    grid: { color: theme.chartGrid },
                    ticks: {
                        maxTicksLimit: 5,
                        callback: (value) => Number(value).toFixed(2)
                    },
                    title: { display: true, text: 'Pearson r', font: { size: 11 } }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        autoSkip: false,
                        color: theme.textSecondary,
                        font: { size: 11 },
                    },
                    afterFit(scale) {
                        scale.width = yAxisWidth;
                    }
                }
            }
        },
        plugins: [{
            id: 'correlationLabels',
            afterDatasetsDraw(chart) {
                const { ctx, chartArea } = chart;
                chart.data.datasets[0].data.forEach((val, i) => {
                    const meta = chart.getDatasetMeta(0).data[i];
                    if (!meta) return;
                    ctx.save();
                    ctx.fillStyle = theme.textSecondary;
                    ctx.font = '10px Inter';
                    ctx.textAlign = val >= 0 ? 'left' : 'right';
                    ctx.textBaseline = 'middle';
                    const xPos = val >= 0
                        ? Math.min(meta.x + 8, chartArea.right - 2)
                        : Math.max(meta.x - 8, chartArea.left + 2);
                    ctx.fillText(val.toFixed(3), xPos, meta.y);
                    ctx.restore();
                });
            }
        }]
    });
}

function renderTargetHistogram(edaData, mode) {
    if (charts.target) charts.target.destroy();
    const ctxTarget = document.getElementById('targetChart').getContext('2d');
    const noteEl = document.getElementById('target-chart-note');
    const theme = getThemeTokens();
    
    let histData, xLabel, barColor;
    if (mode === 'log' && edaData.target_histogram_log) {
        histData = edaData.target_histogram_log;
        xLabel = `log1p(${cachedTargetCol || 'target'})`;
        barColor = { bg: theme.chartPrimarySoft, border: theme.chartPrimaryStrong };
        noteEl.innerText = 'Showing log1p(target) by default because the raw target is highly right-skewed and large outliers compress the original scale.';
    } else {
        histData = edaData.target_histogram_display || edaData.target_histogram;
        xLabel = edaData.target_histogram_display ? `${cachedTargetCol || 'Target'} range (<= P99 shown)` : `${cachedTargetCol || 'Target'} range`;
        barColor = { bg: theme.chartPositiveSoft, border: theme.chartPositiveStrong };
        if (edaData.target_histogram_display) {
            noteEl.innerText = `Raw distribution shown with values above the 99th percentile clipped for visualization only. ${formatFullNumber(edaData.target_histogram_display.excluded_count)} extreme rows are excluded from this view.`;
        } else {
            noteEl.innerText = 'Raw target distribution. Extreme outliers may compress most values to the left of the chart.';
        }
    }
    
    if (!histData || !histData.bins) return;
    
    const bins = histData.bins.slice(0, -1).map((bin, index) => {
        const nextBin = histData.bins[index + 1];
        return `${formatCompactNumber(bin, 2)}-${formatCompactNumber(nextBin, 2)}`;
    });
    const counts = histData.counts;
    setChartBodyMinHeight('targetChart', bins.length > 8 ? 340 : 315);
    
    charts.target = new Chart(ctxTarget, {
        type: 'bar',
        data: {
            labels: bins,
            datasets: [{
                label: 'Frequency',
                data: counts,
                backgroundColor: barColor.bg,
                borderColor: barColor.border,
                borderWidth: 1,
                borderRadius: 2
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            layout: { padding: { top: 8, right: 10, bottom: 18, left: 6 } },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => items[0].label,
                        label: (ctx) => `Count: ${formatFullNumber(ctx.raw)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 8 },
                    title: { display: true, text: xLabel, font: { size: 11 } }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: theme.chartGrid }
                }
            }
        }
    });
}

function setHistToggleState(active) {
    const rawBtn = document.getElementById('btn-hist-raw');
    const logBtn = document.getElementById('btn-hist-log');
    if (active === 'log') {
        logBtn.style.background = 'var(--surface-primary-soft)';
        logBtn.style.borderColor = 'var(--surface-primary-border)';
        logBtn.style.color = 'var(--primary-strong)';
        rawBtn.style.background = 'var(--surface-soft)';
        rawBtn.style.borderColor = 'var(--border-strong)';
        rawBtn.style.color = 'var(--text-secondary)';
    } else {
        rawBtn.style.background = 'var(--surface-success-soft)';
        rawBtn.style.borderColor = 'var(--surface-success-border)';
        rawBtn.style.color = 'var(--success)';
        logBtn.style.background = 'var(--surface-soft)';
        logBtn.style.borderColor = 'var(--border-strong)';
        logBtn.style.color = 'var(--text-secondary)';
    }
}

// 3. Train Models
document.getElementById('btn-train').addEventListener('click', async () => {
    const useLogTransform = Boolean(document.getElementById('use-log-target')?.checked);
    const defaultTrainingModels = taskType === 'classification'
        ? ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']
        : ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost'];
    const rotationModels = (candidateModelNames.length ? candidateModelNames : defaultTrainingModels)
        .filter((name, index, arr) => arr.indexOf(name) === index)
        .slice(0, 8);
    const trainingCycle = [
        'Preparing folds and validation splits',
        ...rotationModels.map(name => `Benchmarking ${name}`),
        'Scoring validation metrics and ranking candidates'
    ];

    showLoader(taskType === 'classification' ? 'Training Classification Models' : 'Training Regression Models', {
        detail: useLogTransform && taskType === 'regression'
            ? 'Launching the training pipeline with log1p(target) enabled.'
            : 'Launching the training pipeline.',
        cycle: trainingCycle
    });
    
    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId, use_log_transform: useLogTransform })
        });
        
        let data;
        try {
            data = await res.json();
        } catch (parseErr) {
            throw new Error(`Server returned an invalid response (Status: ${res.status}). The dataset might have been too large to process, causing a backend timeout.`);
        }
        
        if (data.error) throw new Error(data.error);
        
        hideLoader();
        showSection('results');
        renderResultsList(data);
        
    } catch (e) {
        hideLoader();
        alert(e.message);
    }
});

function formatMetric(value, digits = 3) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(digits);
}

function formatTableMetric(value, digits = 3) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toLocaleString('en-US', { maximumFractionDigits: digits });
}

function formatMetricCompact(value, digits = 2) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: digits
    }).format(Number(value));
}

function abbreviateAxisLabel(label, maxLength = 16) {
    const text = String(label ?? '');
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function findSourceColumnForEncodedFeature(featureName) {
    const sourceCols = [...(datasetCols || [])].sort((a, b) => b.length - a.length);
    const exactMatch = sourceCols.find(col => col === featureName);
    if (exactMatch) return exactMatch;
    return sourceCols.find(col => featureName.startsWith(`${col}_`)) || null;
}

function buildImportanceDisplayData(featureImportance) {
    const grouped = new Map();
    let groupedFeatureCount = 0;

    Object.entries(featureImportance || {}).forEach(([featureName, rawValue]) => {
        const value = Number(rawValue) || 0;
        const sourceColumn = findSourceColumnForEncodedFeature(featureName);
        const isEncodedLevel = Boolean(sourceColumn && sourceColumn !== featureName);
        const displayKey = isEncodedLevel ? sourceColumn : featureName;

        if (!grouped.has(displayKey)) {
            grouped.set(displayKey, {
                label: displayKey,
                value: 0,
                members: [],
                containsEncodedLevels: false,
            });
        }

        const bucket = grouped.get(displayKey);
        bucket.value += value;
        bucket.members.push({ name: featureName, value });
        bucket.containsEncodedLevels = bucket.containsEncodedLevels || isEncodedLevel;

        if (isEncodedLevel) {
            groupedFeatureCount += 1;
        }
    });

    const entries = Array.from(grouped.values())
        .map(item => ({
            ...item,
            members: item.members.sort((a, b) => b.value - a.value),
        }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10);

    return {
        entries,
        usedGrouping: groupedFeatureCount > 0,
    };
}

function formatDuration(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return '—';
    const s = Number(seconds);
    if (s < 1) return `${Math.round(s * 1000)} ms`;
    if (s < 60) return `${s.toFixed(2)} s`;
    const mins = Math.floor(s / 60);
    const rem = s % 60;
    return `${mins}m ${rem.toFixed(1)}s`;
}

function computeRegressionBaselineImprovement(bestRow, baselineRow) {
    if (!bestRow || !baselineRow) return { text: '—', detail: 'Baseline not available' };
    const bestRmse = bestRow.holdout_rmse ?? bestRow.cv_rmse_mean;
    const baseRmse = baselineRow.holdout_rmse ?? baselineRow.cv_rmse_mean;
    const bestR2 = bestRow.holdout_r2 ?? bestRow.cv_r2_mean;
    const baseR2 = baselineRow.holdout_r2 ?? baselineRow.cv_r2_mean;

    if (Number.isFinite(bestRmse) && Number.isFinite(baseRmse) && baseRmse !== 0) {
        const reduction = ((baseRmse - bestRmse) / baseRmse) * 100;
        const sign = reduction >= 0 ? '+' : '';
        return {
            text: `${sign}${reduction.toFixed(1)}%`,
            detail: `RMSE vs baseline (${formatMetricCompact(baseRmse)} -> ${formatMetricCompact(bestRmse)})`
        };
    }

    if (Number.isFinite(bestR2) && Number.isFinite(baseR2)) {
        const lift = bestR2 - baseR2;
        const sign = lift >= 0 ? '+' : '';
        return {
            text: `${sign}${lift.toFixed(3)}`,
            detail: `R² lift vs baseline`
        };
    }

    return { text: '—', detail: 'Baseline comparison unavailable' };
}

function buildRangeErrorBands(scatterData, bandCount = 5) {
    if (!scatterData?.true || !scatterData?.pred || scatterData.true.length === 0) return [];
    const pairs = scatterData.true.map((actual, i) => ({
        actual: Number(actual),
        pred: Number(scatterData.pred[i]),
        absError: Math.abs(Number(actual) - Number(scatterData.pred[i]))
    })).filter(item => Number.isFinite(item.actual) && Number.isFinite(item.pred));

    if (!pairs.length) return [];

    pairs.sort((a, b) => a.actual - b.actual);
    const size = Math.max(1, Math.ceil(pairs.length / bandCount));
    const bands = [];

    for (let i = 0; i < pairs.length; i += size) {
        const slice = pairs.slice(i, i + size);
        if (!slice.length) continue;
        const minActual = slice[0].actual;
        const maxActual = slice[slice.length - 1].actual;
        const mae = slice.reduce((sum, item) => sum + item.absError, 0) / slice.length;
        bands.push({
            label: `${formatMetricCompact(minActual)}-${formatMetricCompact(maxActual)}`,
            mae
        });
    }

    return bands.slice(0, bandCount);
}

function buildSummaryPoints(bestName, improvement, featureImportance, bestModelInfo) {
    const points = [];
    points.push(`<div class="summary-point"><strong>${bestName}</strong> delivered the strongest cross-validated performance in the benchmark set and was selected as the production candidate.</div>`);
    if (bestModelInfo?.target_transform === 'log1p') {
        points.push('<div class="summary-point"><strong>Target transform:</strong> training used log1p(target) to stabilize a skewed positive target, while evaluation metrics were converted back to the original scale.</div>');
    }
    if (improvement?.text && improvement.text !== '—') {
        points.push(`<div class="summary-point"><strong>Baseline Lift:</strong> ${improvement.text} improvement. <span>${improvement.detail}</span></div>`);
    }
    const topDrivers = Object.entries(featureImportance || {}).slice(0, 3).map(([name]) => name);
    if (topDrivers.length > 0) {
        points.push(`<div class="summary-point"><strong>Top drivers:</strong> ${topDrivers.join(', ')}.</div>`);
    }
    if (cachedTargetSummary?.log_transform_suggested || (cachedTargetSummary?.outlier_count ?? 0) > 0) {
        points.push(`<div class="summary-point"><strong>Caution:</strong> The target remains skewed with notable outliers, so monitor extreme-value predictions and consider target transforms or robust tuning in the next iteration.</div>`);
    }
    if ((bestModelInfo?.recommendations || []).length > 0) {
        points.push(`<div class="summary-point"><strong>Next step:</strong> ${bestModelInfo.recommendations[0]}</div>`);
    }
    return points.join('');
}

function buildInterpretationItems(bestName, bestRow, baselineRow, featureImportance) {
    const items = [];
    const baselineR2 = baselineRow?.holdout_r2 ?? baselineRow?.cv_r2_mean;
    const bestR2 = bestRow?.holdout_r2 ?? bestRow?.cv_r2_mean;
    if (Number.isFinite(bestR2)) {
        items.push(`<div class="interpretation-item"><strong>${bestName}</strong> explains ${Math.max(0, bestR2 * 100).toFixed(1)}% of the variance on the holdout view used in this dashboard.</div>`);
    }
    if (Number.isFinite(bestR2) && Number.isFinite(baselineR2)) {
        const lift = bestR2 - baselineR2;
        items.push(`<div class="interpretation-item"><strong>Baseline comparison:</strong> the selected model improved R² by ${lift.toFixed(3)} over the mean baseline.</div>`);
    }
    if (lastTrainResultsData?.best_model?.target_transform === 'log1p') {
        items.push('<div class="interpretation-item"><strong>Training setup:</strong> the model was fit on log1p(target), which is often advisable for positive, right-skewed regression targets with large outliers.</div>');
    }
    const topFeatures = Object.entries(featureImportance || {}).slice(0, 5);
    if (topFeatures.length > 0) {
        items.push(`<div class="interpretation-item"><strong>Prediction drivers:</strong> ${topFeatures.map(([name]) => name).join(', ')} are the strongest model signals in this run.</div>`);
    } else {
        items.push(`<div class="interpretation-item"><strong>Interpretability note:</strong> this estimator does not expose native feature importances, so rely more on error diagnostics and benchmark metrics.</div>`);
    }
    if ((cachedTargetSummary?.skewness ?? 0) > 2) {
        items.push(`<div class="interpretation-item"><strong>Data caution:</strong> target skewness is ${cachedTargetSummary.skewness.toFixed(2)}, so extreme target values may dominate error behavior.</div>`);
    }
    return items.join('');
}

function renderResultsList(data) {
    lastTrainResultsData = data;
    predictionSchema = data.prediction_schema || null;
    const theme = getThemeTokens();

    const bestModelInfo = data.best_model || {};
    const bestName = bestModelInfo.best_model || 'None';
    const targetName = document.getElementById('analysis-target-name')?.innerText || cachedTargetCol || document.getElementById('target-select').value;
    const validModels = (data.models || []).filter(m => !m.error);
    if (validModels.length === 0) {
        document.getElementById('best-model-name').innerText = 'Unavailable';
        document.getElementById('best-model-description').innerText = 'Model training completed without any successful benchmark results.';
        return;
    }
    const bestRow = validModels.find(m => m.model === bestName) || validModels[0];
    const baselineRow = validModels.find(m => /baseline/i.test(m.model));
    const featureImportance = data.feature_importance || {};
    const evaluationCharts = data.evaluation_charts || {};
    const scatterData = evaluationCharts.predicted_vs_actual || evaluationCharts.scatter || null;
    const residualData = evaluationCharts.residuals || null;
    const rangeBands = buildRangeErrorBands(scatterData);
    const improvement = taskType === 'regression'
        ? computeRegressionBaselineImprovement(bestRow, baselineRow)
        : { text: '—', detail: 'Baseline comparison available in leaderboard' };
    const selectionMetricLabel = taskType === 'regression' ? 'CV R²' : 'CV Accuracy';
    const holdoutMetricLabel = taskType === 'regression' ? 'Holdout R²' : 'Holdout Accuracy';
    const heroSelectionNote = taskType === 'regression'
        ? 'Best model selected by cross-validated R², with CV RMSE and holdout R² used as tie-breakers.'
        : 'Best model selected by cross-validated accuracy, with ROC AUC and holdout accuracy used as tie-breakers when available.';
    const fallbackReason = taskType === 'regression'
        ? `${bestName} was selected using cross-validated R² as the primary ranking metric. Holdout metrics are shown separately for context.`
        : `${bestName} was selected using cross-validated accuracy as the primary ranking metric. Holdout metrics are shown separately for context.`;

    ['leaderboard', 'comparison', 'importance', 'scatter', 'residual', 'range', 'evalChart'].forEach(key => {
        if (charts[key]) {
            charts[key].destroy();
            charts[key] = null;
        }
    });

    document.getElementById('best-model-name').innerText = bestName;
    document.getElementById('predict-model-name').innerText = bestName;
    document.getElementById('results-target-name').innerText = targetName || '-';
    document.getElementById('results-status-text').innerText = taskType === 'regression' ? 'Regression Training Complete' : 'Classification Training Complete';
    document.getElementById('results-hero-title').innerText = taskType === 'regression' ? 'Regression Results Dashboard' : 'Classification Results Dashboard';
    document.getElementById('results-hero-subtitle').innerText = heroSelectionNote;
    document.getElementById('best-model-description').innerText = bestModelInfo.reason || fallbackReason;

    const primaryMetricLabel = selectionMetricLabel;
    const primaryHeroLabel = selectionMetricLabel;
    const primaryValue = taskType === 'regression'
        ? (bestRow?.cv_r2_mean ?? bestModelInfo.cv_metrics?.cv_r2_mean ?? bestRow?.holdout_r2)
        : (bestRow?.cv_accuracy_mean ?? bestModelInfo.cv_metrics?.cv_accuracy_mean ?? bestRow?.holdout_accuracy);
    const secondaryValue = taskType === 'regression'
        ? (bestRow?.holdout_rmse ?? bestModelInfo.holdout_metrics?.rmse)
        : (bestRow?.holdout_precision ?? bestModelInfo.holdout_metrics?.precision);
    const tertiaryValue = taskType === 'regression'
        ? (bestRow?.holdout_mae ?? bestModelInfo.holdout_metrics?.mae)
        : (bestRow?.holdout_recall ?? bestModelInfo.holdout_metrics?.recall);
    const cvMean = taskType === 'regression' ? bestRow?.cv_r2_mean : bestRow?.cv_accuracy_mean;
    const cvStd = taskType === 'regression' ? bestRow?.cv_r2_std : bestRow?.cv_accuracy_std;

    document.getElementById('results-primary-metric-label').innerText = primaryMetricLabel;
    document.getElementById('hero-metric-primary-label').innerText = primaryHeroLabel;
    document.getElementById('hero-metric-primary-value').innerText = formatMetric(primaryValue, 3);
    document.getElementById('hero-metric-primary-subtext').innerText = taskType === 'regression'
        ? `Selected on cross-validation; holdout R² ${formatMetric(bestRow?.holdout_r2, 3)}`
        : `Selected on cross-validation; holdout accuracy ${formatMetric(bestRow?.holdout_accuracy, 3)}`;
    document.getElementById('hero-metric-rmse').innerText = taskType === 'regression' ? formatMetricCompact(secondaryValue) : formatMetric(secondaryValue, 3);
    document.getElementById('hero-metric-rmse-subtext').innerText = taskType === 'regression' ? 'Holdout RMSE' : 'Holdout precision';
    document.getElementById('hero-metric-mae').innerText = taskType === 'regression' ? formatMetricCompact(tertiaryValue) : formatMetric(tertiaryValue, 3);
    document.getElementById('hero-metric-mae-subtext').innerText = taskType === 'regression' ? 'Holdout MAE' : 'Holdout recall';
    document.getElementById('hero-metric-improvement').innerText = improvement.text;
    document.getElementById('hero-metric-improvement').className = `result-stat-value ${improvement.text.startsWith('-') ? 'metric-negative' : 'metric-positive'}`;
    document.getElementById('hero-metric-improvement-subtext').innerText = improvement.detail;
    document.getElementById('hero-metric-stability').innerText = `${formatMetric(cvMean, 3)} ± ${formatMetric(cvStd, 3)}`;
    document.getElementById('hero-metric-stability-subtext').innerText = taskType === 'regression' ? 'Cross-validated R² stability' : 'Cross-validated accuracy stability';

    document.getElementById('leaderboard-primary-metric-header').innerText = holdoutMetricLabel;
    document.getElementById('leaderboard-secondary-metric-header').innerText = taskType === 'regression' ? 'RMSE' : 'Precision';
    document.getElementById('leaderboard-tertiary-metric-header').innerText = taskType === 'regression' ? 'MAE' : 'Recall';
    document.getElementById('leaderboard-cv-header').innerText = taskType === 'regression' ? 'CV R² / Std' : 'CV Accuracy / Std';
    document.getElementById('leaderboard-note').innerText = taskType === 'regression'
        ? 'Models are ranked by cross-validated R². Holdout metrics are shown separately for context.'
        : 'Models are ranked by cross-validated accuracy. Holdout metrics are shown separately for context.';
    document.getElementById('comparison-chart-title').innerText = taskType === 'regression' ? 'Model Comparison by CV R²' : 'Model Comparison by CV Accuracy';
    document.getElementById('comparison-chart-note').innerText = taskType === 'regression'
        ? 'Winner selection is based on cross-validated R², not the single holdout split.'
        : 'Winner selection is based on cross-validated accuracy, not the single holdout split.';
    document.getElementById('importance-chart-title').innerText = taskType === 'regression' ? 'Top Drivers of Prediction' : 'Top Model Drivers';
    document.getElementById('scatter-chart-title').innerText = taskType === 'regression' ? 'Actual vs Predicted' : 'ROC Curve';
    document.getElementById('residual-chart-title').innerText = taskType === 'regression' ? 'Residual Diagnostics' : 'Classification Diagnostics';
    document.getElementById('range-chart-title').innerText = taskType === 'regression' ? 'Error by Target Range' : 'Model Error Distribution';

    const tbody = document.getElementById('results-leaderboard-body');
    tbody.innerHTML = validModels.map((model, idx) => {
        const isBest = model.model === bestName;
        const primary = taskType === 'regression' ? model.holdout_r2 : model.holdout_accuracy;
        const secondary = taskType === 'regression' ? model.holdout_rmse : model.holdout_precision;
        const tertiary = taskType === 'regression' ? model.holdout_mae : model.holdout_recall;
        const cvMeanVal = taskType === 'regression' ? model.cv_r2_mean : model.cv_accuracy_mean;
        const cvStdVal = taskType === 'regression' ? model.cv_r2_std : model.cv_accuracy_std;
        const timeVal = model.cv_fit_time_mean;
        return `
            <tr class="${isBest ? 'best-row' : ''}">
                <td><span class="rank-pill">${idx + 1}</span></td>
                <td>
                    <div class="model-name">
                        <strong>${model.model}</strong>
                        ${isBest ? '<span class="winner-tag">Best</span>' : ''}
                    </div>
                </td>
                <td>${formatTableMetric(primary, 3)}</td>
                <td>${taskType === 'regression' ? formatTableMetric(secondary, 0) : formatTableMetric(secondary, 3)}</td>
                <td>${taskType === 'regression' ? formatTableMetric(tertiary, 0) : formatTableMetric(tertiary, 3)}</td>
                <td>${formatMetric(cvMeanVal, 3)} / ${formatMetric(cvStdVal, 3)}</td>
                <td>${formatDuration(timeVal)}</td>
            </tr>
        `;
    }).join('');

    document.getElementById('result-summary-points').innerHTML = buildSummaryPoints(bestName, improvement, featureImportance, bestModelInfo);
    document.getElementById('interpretation-content').innerHTML = buildInterpretationItems(bestName, bestRow, baselineRow, featureImportance);

    const comparisonCtx = document.getElementById('resultsComparisonChart').getContext('2d');
    const comparisonMetricKey = taskType === 'regression' ? 'cv_r2_mean' : 'cv_accuracy_mean';
    const comparisonValues = validModels.map(m => m[comparisonMetricKey] ?? 0);
    setChartBodyMinHeight('resultsComparisonChart', clamp(validModels.length * 34 + 120, 300, 420));
    charts.comparison = new Chart(comparisonCtx, {
        type: 'bar',
        data: {
            labels: validModels.map(m => m.model),
            datasets: [{
                label: taskType === 'regression' ? 'Cross-Validated R²' : 'Cross-Validated Accuracy',
                data: comparisonValues,
                backgroundColor: validModels.map(m => m.model === bestName ? theme.chartPrimaryStrong : theme.chartNeutralSoft),
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            layout: { padding: { top: 8, right: 16, bottom: 14, left: 8 } },
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: theme.chartGrid },
                    ticks: { callback: (value) => Number(value).toFixed(2) }
                },
                y: {
                    grid: { display: false },
                    ticks: { autoSkip: false }
                }
            }
        }
    });

    const importanceCtx = document.getElementById('importanceChart').getContext('2d');
    const importanceEmpty = document.getElementById('importance-empty');
    const importanceNote = document.getElementById('importance-chart-note');
    const importanceDisplay = buildImportanceDisplayData(featureImportance);
    const impEntries = importanceDisplay.entries;
    if (impEntries.length > 0) {
        importanceEmpty.classList.add('hidden');
        document.getElementById('importance-chart-title').innerText = importanceDisplay.usedGrouping
            ? 'Top Drivers of Prediction'
            : 'Encoded Feature Contributions';
        importanceNote.innerText = importanceDisplay.usedGrouping
            ? 'Grouped back to source features where possible. These values reflect model-specific influence from encoded training features.'
            : 'These values reflect model-specific influence from encoded training features.';
        setChartBodyMinHeight('importanceChart', clamp(impEntries.length * 34 + 115, 360, 520));
        charts.importance = new Chart(importanceCtx, {
            type: 'bar',
            data: {
                labels: impEntries.map(item => item.label),
                datasets: [{
                    label: 'Feature Influence',
                    data: impEntries.map(item => item.value),
                    backgroundColor: theme.chartPrimarySoft,
                    borderColor: theme.primaryStrong,
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                layout: { padding: { top: 6, right: 18, bottom: 16, left: 10 } },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => impEntries[items[0].dataIndex]?.label || items[0].label,
                            label: (ctx) => `Influence: ${Number(ctx.raw).toFixed(3)}`,
                            afterBody: (items) => {
                                const item = impEntries[items[0].dataIndex];
                                if (!item) return [];
                                if (!item.containsEncodedLevels) return [];
                                const members = item.members
                                    .slice(0, 3)
                                    .map(member => `Includes: ${member.name} (${member.value.toFixed(3)})`);
                                return members;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: theme.chartGrid },
                        ticks: { callback: (value) => Number(value).toFixed(2), padding: 10 },
                        title: { display: true, text: 'Relative Influence', padding: { top: 12 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            autoSkip: false,
                            padding: 10,
                            callback: (_, index) => abbreviateAxisLabel(impEntries[index]?.label || '', 24)
                        }
                    }
                }
            }
        });
    } else {
        importanceEmpty.classList.remove('hidden');
        document.getElementById('importance-chart-title').innerText = 'Top Drivers of Prediction';
        importanceNote.innerText = 'Model-specific feature influence for the winning estimator.';
    }

    if (taskType === 'regression' && scatterData) {
        const scatterCtx = document.getElementById('resultsScatterChart').getContext('2d');
        const residualCtx = document.getElementById('residualChart').getContext('2d');
        const rangeCtx = document.getElementById('rangeErrorChart').getContext('2d');
        const points = scatterData.true.map((actual, i) => ({ x: actual, y: scatterData.pred[i] }));
        const minAxis = Math.min(...scatterData.true, ...scatterData.pred);
        const maxAxis = Math.max(...scatterData.true, ...scatterData.pred);
        setChartBodyMinHeight('resultsScatterChart', 390);
        setChartBodyMinHeight('residualChart', 390);
        setChartBodyMinHeight('rangeErrorChart', rangeBands.length > 4 ? 360 : 340);

        charts.scatter = new Chart(scatterCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Predictions',
                        data: points,
                        backgroundColor: theme.chartPositiveSoft,
                        pointRadius: 4
                    },
                    {
                        label: 'Ideal',
                        type: 'line',
                        data: [{ x: minAxis, y: minAxis }, { x: maxAxis, y: maxAxis }],
                        borderColor: theme.chartLine,
                        borderDash: [6, 6],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                clip: false,
                layout: { padding: { top: 8, right: 18, bottom: 20, left: 10 } },
                plugins: { legend: { labels: { usePointStyle: true } } },
                scales: {
                    x: {
                        title: { display: true, text: 'Actual Value' },
                        grid: { color: theme.chartGrid },
                        ticks: { callback: (value) => formatMetricCompact(value), padding: 10 },
                        offset: true
                    },
                    y: {
                        title: { display: true, text: 'Predicted Value' },
                        grid: { color: theme.chartGrid },
                        ticks: { callback: (value) => formatMetricCompact(value), padding: 10 }
                    }
                }
            }
        });

        const residualPoints = residualData?.pred?.map((pred, i) => ({ x: pred, y: residualData.residual[i] })) || [];
        charts.residual = new Chart(residualCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Residuals',
                        data: residualPoints,
                        backgroundColor: theme.chartInfoSoft,
                        pointRadius: 4
                    },
                    {
                        label: 'Zero Error',
                        type: 'line',
                        data: residualPoints.length ? [
                            { x: Math.min(...residualData.pred), y: 0 },
                            { x: Math.max(...residualData.pred), y: 0 }
                        ] : [],
                        borderColor: theme.chartLine,
                        borderDash: [6, 6],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                clip: false,
                layout: { padding: { top: 8, right: 18, bottom: 20, left: 10 } },
                plugins: { legend: { labels: { usePointStyle: true } } },
                scales: {
                    x: {
                        title: { display: true, text: 'Predicted Value' },
                        grid: { color: theme.chartGrid },
                        ticks: { callback: (value) => formatMetricCompact(value), padding: 10 },
                        offset: true
                    },
                    y: {
                        title: { display: true, text: 'Residual' },
                        grid: { color: theme.chartGrid },
                        ticks: { callback: (value) => formatMetricCompact(value), padding: 10 }
                    }
                }
            }
        });

        charts.range = new Chart(rangeCtx, {
            type: 'bar',
            data: {
                labels: rangeBands.map(b => b.label),
                datasets: [{
                    label: 'Band MAE',
                    data: rangeBands.map(b => b.mae),
                    backgroundColor: theme.chartAmberSoft,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { top: 10, right: 18, bottom: 24, left: 10 } },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => rangeBands[items[0].dataIndex]?.label || items[0].label,
                            label: (ctx) => `MAE: ${formatMetricCompact(ctx.raw)}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            maxRotation: 24,
                            minRotation: 24,
                            padding: 12,
                            callback: (_, index) => abbreviateAxisLabel(rangeBands[index]?.label || '', 18)
                        }
                    },
                    y: {
                        title: { display: true, text: 'Mean Absolute Error' },
                        grid: { color: theme.chartGrid },
                        ticks: { callback: (value) => formatMetricCompact(value), padding: 10 }
                    }
                }
            }
        });
    }
}

document.getElementById('btn-view-predictions').addEventListener('click', () => {
    const panel = document.getElementById('predictions-panel');
    if (panel) {
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});

document.getElementById('btn-download-results').addEventListener('click', () => {
    if (!lastTrainResultsData) return;
    const payload = {
        generated_at: new Date().toISOString(),
        target: document.getElementById('analysis-target-name')?.innerText || cachedTargetCol || null,
        task_type: taskType,
        results: lastTrainResultsData
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `nexusml-results-${sessionId || 'session'}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
});

// 4. Manual Prediction Setup
document.getElementById('btn-go-predict').addEventListener('click', () => {
    renderPredictionForm(predictionSchema);
    showSection('predict');
});

// Predict Submit
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const features = {};

    const fields = predictionSchema?.fields || [];
    fields.forEach(field => {
        const el = document.getElementById(`input-${field.name}`);
        if (!el) return;
        const coerced = coercePredictionFieldValue(field, el.value);
        if (coerced !== undefined) {
            features[field.name] = coerced;
        }
    });

    if (Object.keys(features).length === 0) {
        alert('Enter at least one value or choose a categorical option. Blank fields already fall back to defaults.');
        return;
    }
    
    document.getElementById('prediction-output').innerHTML = '<span class="spinner" style="width:30px;height:30px;display:inline-block;border-width:3px;margin:0;"></span>';
    
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId, features: features })
        });
        const data = await res.json();
        
        if (data.error) throw new Error(data.error);
        
        let outVal = data.prediction;
        if (typeof outVal === 'number' && !Number.isInteger(outVal)) outVal = outVal.toFixed(3);
        
        document.getElementById('prediction-output').innerText = outVal;
        
    } catch (err) {
        document.getElementById('prediction-output').innerText = "Error";
        alert(err.message);
    }
});
