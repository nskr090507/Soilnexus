const API_BASE = 'http://127.0.0.1:5000';

// ─── DATA STORAGE FOR CHART ───
const chartLabels = [];
const moistureData = [];
const temperatureData = [];
const phData = [];

// ─── CREATE THE CHART ───
const ctx = document.getElementById('telemetryChart').getContext('2d');
const telemetryChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: chartLabels,
        datasets: [
            {
                label: 'Moisture %',
                data: moistureData,
                borderColor: '#e7c59a',
                backgroundColor: 'rgba(231,197,154,0.12)',
                tension: 0.4,
                fill: true
            },
            {
                label: 'Temperature °C',
                data: temperatureData,
                borderColor: '#e39776',
                backgroundColor: 'rgba(227,151,118,0.10)',
                tension: 0.4,
                fill: true
            },
            {
                label: 'pH Level',
                data: phData,
                borderColor: '#b8c786',
                backgroundColor: 'rgba(184,199,134,0.10)',
                tension: 0.4,
                fill: true
            }
        ]
    },
    options: {
        responsive: true,
        animation: { duration: 500 },
        scales: {
            y: { beginAtZero: false, ticks: { color: '#c9d9cd' }, grid: { color: 'rgba(255,255,255,0.08)' } },
            x: { display: true, ticks: { color: '#c9d9cd' }, grid: { display: false } }
        },
        plugins: {
            legend: { display: false }
        }
    }
});
// ─── OFFLINE DEMO MODE ───
let demoMode = false;

function loadDemoData() {
    fetch(`${API_BASE}/get_sensor_data?t=${Date.now()}`)
        .then(response => response.json())
        .then(data => {
            if (data.is_online === false) {
                showNetworkAlert("Hardware Offline - Check Sensors");
            } else {
                hideNetworkAlert();
            }
            applySensorData(data);
            return fetch(`${API_BASE}/predict_irrigation`);
        })
        .then(response => response.json())
        .then(data => {
            showAIAdvice(data);
            updateAdvisoryMessage(data);
            checkDiseaseWarning(data);
            const pct = ((data.confidence_score || 0) * 100).toFixed(1);
            document.getElementById('confidence-badge').innerText = pct + '%';
            document.getElementById('confidence-score-display').innerText = pct + '%';
            return fetch(`${API_BASE}/recommend_crop`);
        })
        .then(response => response.json())
        .then(data => renderRecommendations(data.recommendations || []))
        .catch(error => handleNetworkTimeout());
}

// ─── UPDATE CHART ───
function updateChartInstance(newData) {
    const now = new Date().toLocaleTimeString();
    chartLabels.push(now);
    moistureData.push(newData.moisture);
    temperatureData.push(newData.temperature);
    phData.push(newData.pH);

    if (chartLabels.length > 30) {
        chartLabels.shift();
        moistureData.shift();
        temperatureData.shift();
        phData.shift();
    }
    telemetryChart.update();
}

function fetchHistoricalData() {
    fetch(`${API_BASE}/sensor_history`)
        .then(response => response.json())
        .then(history => {
            if (history && history.length > 0) {
                chartLabels.length = 0;
                moistureData.length = 0;
                temperatureData.length = 0;
                phData.length = 0;
                history.forEach(data => {
                    const ts = new Date(data.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    chartLabels.push(ts);
                    moistureData.push(data.moisture);
                    temperatureData.push(data.temperature);
                    phData.push(data.ph);
                });
                telemetryChart.update();
            }
        })
        .catch(err => { /* silent fallback for demo */ });
}

// ─── FETCH LIVE DATA WITH SPEED TRACKING ───
function fetchLiveMetrics() {
    const startTime = Date.now();

    fetch(`${API_BASE}/get_sensor_data?t=${Date.now()}`)
        .then(response => {
            if (!response.ok) throw new Error("Network error");
            return response.json();
        })
        .then(data => {
            if (data.is_online === false) {
                showNetworkAlert("Hardware Offline - Check Sensors");
            } else {
                hideNetworkAlert();
            }
            const responseTime = Date.now() - startTime;
            trackNetworkSpeed(responseTime);

            applySensorData(data);
            return fetch(`${API_BASE}/predict_irrigation?lat=${currentLat}&lon=${currentLon}`);
        })
        .then(response => {
            if (!response.ok) throw new Error("Network error");
            return response.json();
        })
        .then(data => {
            showAIAdvice(data);
            updateAdvisoryMessage(data);
            checkDiseaseWarning(data);

            if (data.confidence_score !== undefined) {
                const pct = (data.confidence_score * 100).toFixed(1);
                document.getElementById('confidence-badge').innerText = pct + '%';
                document.getElementById('confidence-score-display').innerText = pct + '%';
            }
            const currentSoil = localStorage.getItem('farmsense-soil-type');
            if (!currentSoil) {
                return Promise.resolve({ gatekeeperBlocked: true });
            }
            return fetch(`${API_BASE}/recommend_crop?lat=${currentLat}&lon=${currentLon}&soil=${currentSoil}`);
        })
        .then(response => {
            if (response.gatekeeperBlocked) return response;
            if (!response.ok) throw new Error("Network error");
            return response.json();
        })
        .then(data => {
            if (data.gatekeeperBlocked) {
                const grid = document.querySelector('.crop-grid');
                const count = document.querySelector('.crop-count');
                if (grid) grid.innerHTML = '<article class="crop-item"><h3>Awaiting Selection</h3><p>Please select a soil type near the Field rhythm section to view recommendations.</p></article>';
                if (count) count.innerText = '0 crops';
                return;
            }
            const recs = data.recommendations || [];
            localStorage.setItem('farmsense-crops', JSON.stringify(recs));
            renderRecommendations(recs);
        })
        .catch(error => handleNetworkTimeout());
}
// ─── APPLY SENSOR DATA & CACHING ───
let cachedSensors = JSON.parse(localStorage.getItem('farmsense-sensors')) || null;

function applySensorData(data) {
    cachedSensors = data;
    localStorage.setItem('farmsense-sensors', JSON.stringify(data));
    localStorage.setItem('farmsense-last-data-time', Date.now().toString());
    updateDashboard(data);
    updateChartInstance(data);
    document.getElementById('ph-reading').innerText = data.pH || '--';
    document.getElementById('humidity-reading').innerText = data.humidity || '--';

    const sandbox = document.getElementById('judge-sandbox');
    if (sandbox) {
        sandbox.style.display = (data.demo_mode || demoMode) ? 'block' : 'none';
    }
    
    // NPK Tracking
    if (document.getElementById('n-reading')) {
        document.getElementById('n-reading').innerText = data.nitrogen || '--';
        document.getElementById('p-reading').innerText = data.phosphorus || '--';
        document.getElementById('k-reading').innerText = data.potassium || '--';
    }
    
    const badge = document.getElementById('advisory-urgency-badge');
    const statusLight = document.querySelector('.status-light');
    const liveText = document.getElementById('live-reading-text');
    const liveDot = document.getElementById('live-reading-dot');
    
    if (data.is_online === false) {
        badge.innerText = `⚪ Hardware Offline (Last online: ${data.last_seen_time_str || 'Unknown'})`;
        badge.className = "urgency-offline";
        badge.style.backgroundColor = "rgba(71, 46, 36, 0.12)";
        badge.style.color = "#624433";
        if (statusLight) statusLight.style.backgroundColor = '#95a5a6';
        if (liveText) liveText.innerText = `Offline (Last: ${data.last_seen_time_str || 'Unknown'})`;
        if (liveDot) liveDot.style.backgroundColor = '#95a5a6';
        if (liveDot) liveDot.style.boxShadow = '0 0 0 3px rgba(149,165,166,0.13)';
    } else {
        if (statusLight) statusLight.style.backgroundColor = '#2ecc71';
        if (liveText) liveText.innerText = 'Live readings';
        if (liveDot) liveDot.style.backgroundColor = '#a9cf7c';
        if (liveDot) liveDot.style.boxShadow = '0 0 0 3px rgba(169,207,124,0.13)';
    }
}

function restoreCachedSensors() {
    if (cachedSensors) {
        applySensorData(cachedSensors);
    }
}

// ─── SOFT VALUE TRANSITIONS ───
function animateMetric(id, value, decimals = 0) {
    const element = document.getElementById(id);
    const target = Number(value);
    if (!element || Number.isNaN(target)) return;

    const start = Number(element.dataset.metricValue ?? target);
    const startedAt = performance.now();
    const duration = 450;

    function frame(now) {
        const progress = Math.min((now - startedAt) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        element.innerText = (start + (target - start) * eased).toFixed(decimals);
        if (progress < 1) requestAnimationFrame(frame);
    }

    element.dataset.metricValue = target;
    requestAnimationFrame(frame);
}

// ─── UPDATE SENSOR VALUES ───
function updateDashboard(data) {
    animateMetric('soil-moisture', data.moisture);
    animateMetric('air-temp', data.temperature);
    animateMetric('humidity', data.humidity);
    animateMetric('ph-level', data.pH, 1);
    const moistureMeter = document.getElementById('moisture-meter');
    if (moistureMeter) moistureMeter.style.width = Math.min(100, Math.max(4, Number(data.moisture) || 0)) + '%';
    checkMoistureAlert(data.moisture);
}

// ─── MOISTURE ALERT ───
function checkMoistureAlert(moisture) {
    const alertBox = document.getElementById('moisture-alert');
    if (moisture < 20) {
        alertBox.innerText = "🚨 CRITICAL: Activate Irrigation Systems Immediately!";
        alertBox.className = "alert-red";
        updateAdvisoryUrgency("high");
    } else {
        alertBox.innerText = "✅ Soil Moisture Levels are Stable";
        alertBox.className = "alert-green";
        updateAdvisoryUrgency("healthy");
    }
}

// ─── UPDATE ADVISORY URGENCY BADGE ───
function updateAdvisoryUrgency(level) {
    const badge = document.getElementById('advisory-urgency-badge');
    if (level === "high") {
        badge.innerText = "🔴 High Priority - Immediate Action Required";
        badge.className = "urgency-high";
    } else {
        badge.innerText = "🟢 Healthy Status - Crops Are Fine";
        badge.className = "urgency-healthy";
    }
}

// ─── AI ADVICE ───
function showAIAdvice(data) {
    const box = document.getElementById('ai-recommendation');
    if (data.irrigation_required === 1) {
        box.innerText = "⚠️ Water Needed: Scheduled Irrigation Recommended";
        box.style.backgroundColor = "#d6eaf8";
        box.style.color = "#1a5276";
    } else {
        box.innerText = "✅ Soil Moisture Optimal: No Irrigation Required";
        box.style.backgroundColor = "#d5f5e3";
        box.style.color = "#1e8449";
    }
}
// ─── DYNAMIC ADVISORY MESSAGE ───
function updateAdvisoryMessage(data) {
    const msg = document.getElementById('advisory-message-text');
    if (data.message) {
        msg.innerText = `Advisory: ${data.message}.`;
    } else if (data.irrigation_required === 1) {
        msg.innerText = "Advisory: Soil moisture levels dropping. Schedule irrigation within the next 2 hours.";
    } else {
        msg.innerText = "Advisory: Soil condition optimal. No watering needed.";
    }

    // Fertilizer status (if backend sends it)
    if (data.fertilizer_status !== undefined) {
        if (data.fertilizer_status === 1) {
            msg.innerText += " Fertilizer application recommended this week.";
        } else {
            msg.innerText += " Fertilizer levels are sufficient.";
        }
    }
}
// ─── DISEASE WARNING CHECK ───
function checkDiseaseWarning(data) {
    const tag = document.getElementById('disease-warning-tag');
    const statusLight = document.querySelector('.status-light');

    // Example threshold: high humidity + high temp = disease risk
    if (data.disease_risk === 1) {
        tag.classList.remove('hidden');
        if (statusLight) statusLight.style.backgroundColor = '#e74c3c'; // red alert
    } else {
        tag.classList.add('hidden');
    }
}
function deriveRiskFactors(item) {
    const factors = [];
    const suitability = item.suitability_score ?? 0;
    const profit = item.expected_profit ?? 0;
    const fertilizer = item.fertilizer_info?.type || '';
    if (suitability < 70) factors.push('Low suitability for current field conditions');
    if (profit < 1000) factors.push('Expected profit is below threshold');
    if (!fertilizer || fertilizer === 'N/A') factors.push('Fertilizer recommendation is incomplete');
    if (item.measures?.length) factors.push('Requires extra management measures');
    if (!factors.length) factors.push('No major risks detected.');
    return factors;
}

 function renderRecommendations(recommendations) {
    const container = document.querySelector('.crop-grid');
    if (!container) return;

    if (!recommendations.length) {
        container.innerHTML = '<article class="crop-item"><h3>No recommendations</h3><p>Unable to load crop suggestions right now.</p></article>';
        return;
    }

    container.innerHTML = recommendations.slice(0, 6).map((item, index) => {
        const rawSuitability = item.suitability_score ?? 0;
        const suitability = Math.max(0, rawSuitability - 15);
        const riskScore = Math.min(100, Math.max(0, 100 - rawSuitability));
        const profit = item.expected_profit ?? 0;
        const fertType = item.fertilizer_info?.type || 'N/A';
        const fertMethod = item.fertilizer_info?.method || 'N/A';
        const fertSchedule = item.fertilizer_info?.schedule || 'N/A';
        const measures = item.measures && item.measures.length > 0 ? item.measures.join(', ') : 'None';
        const sowingWindow = item.sowing_window || 'Open';
        const advice = item.current_lifecycle_advice?.advice || 'Monitor crop growth closely.';
        const factors = deriveRiskFactors(item);
        const factorItems = factors.map(f => `<li>${f}</li>`).join('');
        
        return `
            <article class="crop-item expanded-item">
                <div class="crop-card-header">
                    <div>
                        <h3>${item.crop}</h3>
                        <p>Suitability: <strong>${suitability.toFixed(1)}%</strong> · Risk: <strong>${riskScore.toFixed(0)}%</strong></p>
                    </div>
                    <button class="risk-button button-secondary" data-index="${index}">Risk ${riskScore.toFixed(0)}%</button>
                </div>
                <i><b style="width:${Math.min(100, suitability)}%"></b></i>
                <div class="recommendation-details" style="display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 14px; flex-grow: 1; align-content: space-between;">
                    <div style="display: flex; justify-content: space-between; gap: 12px; color: #d6e7cf; font-size: 11px;"><span>Expected profit</span><strong>Rs. ${Math.round(item.expected_profit ?? 0).toLocaleString()}</strong></div>
                    <div style="display: flex; justify-content: space-between; gap: 12px; color: #d6e7cf; font-size: 11px;"><span>Fertilizer Type</span><strong>${fertType}</strong></div>
                    <div style="display: flex; justify-content: space-between; gap: 12px; color: #d6e7cf; font-size: 11px;"><span>Fertilizer Method</span><strong>${fertMethod}</strong></div>
                    <div style="display: flex; justify-content: space-between; gap: 12px; color: #d6e7cf; font-size: 11px;"><span>Fertilizer Schedule</span><strong>${fertSchedule}</strong></div>
                    <div style="display: flex; justify-content: space-between; gap: 12px; color: #d6e7cf; font-size: 11px;"><span>Measures</span><strong>${measures}</strong></div>
                    <div style="display: flex; justify-content: space-between; gap: 12px; color: #d6e7cf; font-size: 11px;"><span>Sowing Window</span><strong style="color: ${sowingWindow.toLowerCase() === 'open' ? '#34d399' : '#fca5a5'}">${sowingWindow}</strong></div>
                    <div style="display: flex; justify-content: space-between; gap: 12px; color: #d6e7cf; font-size: 11px;"><span>Advice</span><strong style="text-align: right; max-width: 60%;">${advice}</strong></div>
                </div>
                <div class="risk-details hidden" id="home-risk-details-${index}">
                    <p>Risk factors</p>
                    <ul>${factorItems}</ul>
                </div>
            </article>
        `;
    }).join('');

    const countBadge = document.querySelector('.crop-count');
    if (countBadge) {
        countBadge.innerText = `${recommendations.length} crop options`;
    }

    document.querySelectorAll('.risk-button').forEach(button => {
        button.addEventListener('click', () => {
            const details = document.getElementById(`home-risk-details-${button.dataset.index}`);
            if (details) details.classList.toggle('hidden');
        });
    });
}

// ─── NETWORK SPEED TRACKER ───
function trackNetworkSpeed(responseTime) {
    const speedDisplay = document.getElementById('network-speed');
    speedDisplay.innerText = responseTime + 'ms';
    if (responseTime > 3000) {
        speedDisplay.className = "speed-slow";
        document.getElementById('network-speed').title = "⚠️ Slow connection detected";
    } else {
        speedDisplay.className = "speed-normal";
    }
    adjustPollingSpeed(responseTime);
}
 
 // ─── NETWORK BANNER / ERROR HANDLING ───
let lastOnlineTime = localStorage.getItem('farmsense-last-online') || null;

function showNetworkAlert(message) {
    const banner = document.getElementById('network-banner');
    const status = document.getElementById('system-status');
    const lastOnlineEl = document.getElementById('last-online-time');
    if (banner) banner.style.display = 'block';
    if (status) status.innerText = message || "Offline - Reconnecting to FarmSense Network...";
    if (lastOnlineEl && lastOnlineTime) {
        lastOnlineEl.innerText = `(Last online: ${lastOnlineTime})`;
    }
}

function hideNetworkAlert() {
    const banner = document.getElementById('network-banner');
    const lastOnlineEl = document.getElementById('last-online-time');
    if (banner) banner.style.display = 'none';
    if (lastOnlineEl) lastOnlineEl.innerText = "";
    
    // We are online, update the last online time
    const now = new Date();
    lastOnlineTime = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + " on " + now.toLocaleDateString();
    localStorage.setItem('farmsense-last-online', lastOnlineTime);
}

// ─── TIMEOUT HANDLER ───
function handleNetworkTimeout() {
    // Only show alert if we haven't received data recently
    const lastDataTime = localStorage.getItem('farmsense-last-data-time');
    const now = Date.now();
    if (!lastDataTime || (now - parseInt(lastDataTime)) > 10000) {
        showNetworkAlert("Offline - Reconnecting to FarmSense Network...");
    }
    restoreCachedSensors();
    const cachedCrops = JSON.parse(localStorage.getItem('farmsense-crops'));
    if (cachedCrops) {
        renderRecommendations(cachedCrops);
    }
}
// ─── TOGGLE CHART VISIBILITY ───
document.getElementById('toggle-chart-btn').addEventListener('click', function() {
    const canvas = document.getElementById('telemetryChart');
    if (canvas.style.display === 'none') {
        canvas.style.display = 'block';
        this.innerText = 'Hide Chart';
    } else {
        canvas.style.display = 'none';
        this.innerText = 'Show Chart';
    }
});
// ─── DEMO MODE TOGGLE ───
document.getElementById('demo-mode-btn').addEventListener('click', function() {
    demoMode = !demoMode;
    if (!demoMode) { fetch(`${API_BASE}/update_simulated_sensors`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})}); }
    const sandbox = document.getElementById('judge-sandbox');
    if (demoMode) {
        this.innerText = 'Disable Demo Mode';
        loadDemoData();
        if (sandbox) sandbox.style.display = 'block';
        localStorage.setItem('farmsense_demo_mode', 'true');
    } else {
        this.innerText = 'Enable Demo Mode';
        fetchLiveMetrics();
        if (sandbox) sandbox.style.display = 'none';
        localStorage.setItem('farmsense_demo_mode', 'false');
    }
});
document.getElementById('refresh-recommendations')?.addEventListener('click', () => {
    fetch(`${API_BASE}/recommend_crop`)
        .then(response => response.json())
        .then(data => renderRecommendations(data.recommendations || []))
        .catch(() => handleNetworkTimeout());
});

const leafUpload = document.getElementById('leaf-upload');
const leafPreview = document.getElementById('leaf-preview');
const leafPreviewPlaceholder = document.getElementById('leaf-preview-placeholder');
const analyzeButton = document.getElementById('analyze-btn');
const predictedCrop = document.getElementById('predicted-crop');
const predictedCondition = document.getElementById('predicted-condition');
const predictionConfidence = document.getElementById('prediction-confidence');
const predictionRemedy = document.getElementById('prediction-remedy');
const predictionPrecaution = document.getElementById('prediction-precaution');

leafUpload?.addEventListener('change', () => {
    const file = leafUpload.files?.[0];
    if (!file) {
        leafPreview.src = '';
        leafPreview.classList.add('hidden');
        leafPreviewPlaceholder.innerText = 'No image selected';
        return;
    }

    const url = URL.createObjectURL(file);
    leafPreview.src = url;
    leafPreview.classList.remove('hidden');
    leafPreviewPlaceholder.innerText = '';
});

analyzeButton?.addEventListener('click', () => {
    const file = leafUpload.files?.[0];
    if (!file) {
        alert('Please upload a leaf image before analyzing.');
        return;
    }

    const formData = new FormData();
    formData.append('leaf_image', file);
    analyzeButton.disabled = true;
    analyzeButton.innerText = 'Analyzing...';

    fetch(`${API_BASE}/predict_disease`, {
        method: 'POST',
        body: formData,
    })
        .then(async response => {
            analyzeButton.disabled = false;
            analyzeButton.innerText = 'Analyze photo';
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Disease analysis failed.');
            }
            updateDiseaseDetails(data);
        })
        .catch(error => {
            analyzeButton.disabled = false;
            analyzeButton.innerText = 'Analyze photo';
            alert(error.message || 'Unable to analyze the leaf image.');
        });
});

function updateDiseaseDetails(data) {
    predictedCrop.innerText = data.crop || 'Unknown';
    predictedCondition.innerText = data.condition || 'Unknown';
    predictionConfidence.innerText = data.confidence !== undefined ? `${(data.confidence * 100).toFixed(1)}%` : '--';
    predictionRemedy.innerText = data.remedy || 'No remedy available.';
    predictionPrecaution.innerText = data.precaution || 'No precaution available.';
}

// ─── WEBSOCKET & ADAPTIVE POLLING ───
let pollInterval = 3000; // Manual refresh or websocket pushes only for telemetry
let pollTimer = null; // Removed aggressive auto-refresh interval

const socket = typeof io !== 'undefined' ? io(API_BASE) : null;
if (socket) {
    socket.on('sensor_update', (data) => {
        if (data.is_online === false) {
            showNetworkAlert("Hardware Offline - Check Sensors");
        } else {
            hideNetworkAlert();
        }
        applySensorData(data);
        updateChartInstance(data);
    });
    socket.on('connect_error', () => {
        console.log('Socket connection error, starting fallback');
        startFallbackPolling();
    });
    socket.on('disconnect', () => {
        console.log('Socket disconnected, enabling fallback polling');
        startFallbackPolling();
    });
    socket.on('connect', () => {
        console.log('Socket connected');
        stopFallbackPolling();
        hideNetworkAlert();
    });
} else {
    console.log('Socket.io not available, using polling only');
    startFallbackPolling();
}

function adjustPollingSpeed(responseTime) {
    // Auto-polling disabled to respect manual refresh constraints
}

// ─── FALLBACK POLLING FOR SOCKET DISCONNECTS ───
let fallbackPollInterval = null;

function startFallbackPolling() {
    if (fallbackPollInterval) return;
    console.log('Starting fallback polling at 2 second interval');
    fallbackPollInterval = setInterval(() => {
        fetchLiveMetrics();
    }, 2000);
}

function stopFallbackPolling() {
    if (fallbackPollInterval) {
        clearInterval(fallbackPollInterval);
        fallbackPollInterval = null;
        console.log('Stopped fallback polling');
    }
}

fetchHistoricalData();
fetchLiveMetrics();

// ─── SIDEBAR MENU ───
let currentLat = localStorage.getItem('farm-lat') || 28.61;
let currentLon = localStorage.getItem('farm-lon') || 77.20;

document.addEventListener("DOMContentLoaded", () => {
    const latInput = document.getElementById('coord-lat');
    const lonInput = document.getElementById('coord-lon');
    if (latInput) latInput.value = currentLat;
    if (lonInput) lonInput.value = currentLon;
    
    function saveCoords() {
        if (latInput && lonInput) {
            currentLat = latInput.value;
            currentLon = lonInput.value;
            localStorage.setItem('farm-lat', currentLat);
            localStorage.setItem('farm-lon', currentLon);
            fetchLiveMetrics();
        }
    }
    if (latInput) latInput.addEventListener('change', saveCoords);
    if (lonInput) lonInput.addEventListener('change', saveCoords);

    const soilSelector = document.getElementById('dashboard-soil-selector');
    const submitSoilBtn = document.getElementById('submit-soil-btn');
    if (soilSelector) {
        const savedSoil = localStorage.getItem('farmsense-soil-type');
        if (savedSoil) soilSelector.value = savedSoil;
    }
    if (submitSoilBtn && soilSelector) {
        submitSoilBtn.addEventListener('click', () => {
            const val = soilSelector.value;
            if (val) {
                localStorage.setItem('farmsense-soil-type', val);
                fetchLiveMetrics();
            } else {
                alert("Please select a soil type first.");
            }
        });
    }
});

const sidebarToggle = document.getElementById('sidebar-toggle');
if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
        const shell = document.querySelector('.app-shell');
        const collapsed = shell.classList.toggle('sidebar-collapsed');
        sidebarToggle.setAttribute('aria-expanded', String(!collapsed));
        setTimeout(() => telemetryChart.resize(), 280);
    });
}

// ─── PREDICTOR SUBMENU ───
const predictorToggle = document.getElementById('predictor-toggle');
const predictorSubmenu = document.getElementById('predictor-submenu');
if (predictorToggle && predictorSubmenu) {
    predictorToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const isExpanded = predictorToggle.getAttribute('aria-expanded') === 'true';
        predictorToggle.setAttribute('aria-expanded', !isExpanded);
        predictorSubmenu.classList.toggle('hidden');
    });

    document.addEventListener('click', (e) => {
        if (!predictorToggle.contains(e.target) && !predictorSubmenu.contains(e.target)) {
            predictorToggle.setAttribute('aria-expanded', 'false');
            predictorSubmenu.classList.add('hidden');
        }
    });
}

// ─── REFRESH HARDWARE BTN ───
const refreshHardwareBtn = document.getElementById('refresh-hardware-btn');
if (refreshHardwareBtn) {
    refreshHardwareBtn.addEventListener('click', () => {
        fetchLiveMetrics();
    });
}

// ─── DISEASE ANALYZER ───
document.addEventListener('DOMContentLoaded', () => {
    const leafUpload = document.getElementById('leaf-upload');
    const leafPreview = document.getElementById('leaf-preview');
    const leafPreviewPlaceholder = document.getElementById('leaf-preview-placeholder');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    let selectedFile = null;

    function analyzeDisease(file) {
        if (analyzeBtn) analyzeBtn.innerText = "Analyzing...";
        
        const formData = new FormData();
        formData.append('leaf_image', file);
        
        fetch(`${API_BASE}/predict_disease`, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (analyzeBtn) analyzeBtn.innerText = "Analyze photo";
            if (data.error) {
                document.getElementById('prediction-remedy').innerText = data.error;
                return;
            }
            document.getElementById('predicted-crop').innerText = data.crop || "Unknown";
            
            const conditionEl = document.getElementById('predicted-condition');
            conditionEl.innerText = data.condition || "Unknown";
            conditionEl.style.color = data.is_healthy ? "#1e8449" : "#e74c3c";
            
            document.getElementById('prediction-confidence').innerText = data.confidence ? (data.confidence * 100).toFixed(1) + "%" : "--%";
            document.getElementById('prediction-remedy').innerText = data.remedy || "N/A";
            document.getElementById('prediction-precaution').innerText = data.precaution || "N/A";
        })
        .catch(err => {
            if (analyzeBtn) analyzeBtn.innerText = "Analyze photo";
            document.getElementById('prediction-remedy').innerText = "Network error. Make sure the backend is running.";
            // silent fallback for demo
        });
    }

    if (leafUpload && leafPreview && leafPreviewPlaceholder) {
        leafUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    leafPreview.src = e.target.result;
                    leafPreview.classList.remove('hidden');
                    leafPreviewPlaceholder.classList.add('hidden');
                };
                reader.readAsDataURL(file);
                
                // Auto-analyze for real-time speed
                analyzeDisease(file);
            }
        });
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            if (selectedFile) analyzeDisease(selectedFile);
        });
    }

    // ─── JUDGE SANDBOX SIMULATOR CONTROLS ───
    const toggleSandboxBtn = document.getElementById('toggle-sandbox-btn');
    const sandboxContent = document.getElementById('sandbox-content');
    let isCollapsed = false;

    if (toggleSandboxBtn && sandboxContent) {
        toggleSandboxBtn.addEventListener('click', () => {
            isCollapsed = !isCollapsed;
            sandboxContent.style.display = isCollapsed ? 'none' : 'flex';
            toggleSandboxBtn.innerText = isCollapsed ? '+' : '−';
            document.getElementById('judge-sandbox').style.width = isCollapsed ? '160px' : '280px';
        });
    }

    // Update range labels on slide
    const sliders = ['moisture', 'ph', 'temp', 'humidity'];
    sliders.forEach(id => {
        const slider = document.getElementById(`sim-${id}`);
        const display = document.getElementById(`sim-${id}-val`);
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                const suffix = id === 'temp' ? '°C' : (id === 'ph' ? '' : '%');
                display.innerText = e.target.value + suffix;
            });
        }
    });

    const sendSimBtn = document.getElementById('send-sim-btn');
    if (sendSimBtn) {
        sendSimBtn.addEventListener('click', async () => {
            sendSimBtn.innerText = "Injecting...";
            const payload = {
                ph: parseFloat(document.getElementById('sim-ph').value),
                moisture: parseFloat(document.getElementById('sim-moisture').value),
                temperature: parseFloat(document.getElementById('sim-temp').value),
                humidity: parseFloat(document.getElementById('sim-humidity').value)
            };

            try {
                const res = await fetch(`${API_BASE}/update_simulated_sensors`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    sendSimBtn.innerText = "Injected!";
                    setTimeout(() => { sendSimBtn.innerText = "Inject Telemetry"; }, 1500);
                    // Trigger live metrics and recommendations fetch
                    if (typeof fetchLiveMetrics === 'function') fetchLiveMetrics();
                    if (typeof fetchRecommendations === 'function') fetchRecommendations();
                }
            } catch (err) {
                // silent fallback for demo
                sendSimBtn.innerText = "Error";
                setTimeout(() => { sendSimBtn.innerText = "Inject Telemetry"; }, 1500);
            }
        });
    }

    // --- AI Chat Widget Logic ---
    const chatHeader = document.getElementById('ai-chat-header');
    const chatBody = document.getElementById('ai-chat-body');
    const chatToggleIcon = document.getElementById('ai-chat-toggle-icon');
    const chatInput = document.getElementById('ai-chat-input');
    const chatSendBtn = document.getElementById('ai-chat-send');
    const chatMessages = document.getElementById('ai-chat-messages');

    if (chatHeader) {
        chatHeader.addEventListener('click', () => {
            const isHidden = chatBody.style.display === 'none';
            chatBody.style.display = isHidden ? 'flex' : 'none';
            chatToggleIcon.innerText = isHidden ? '▼' : '▲';
            if (isHidden) chatInput.focus();
        });

        async function sendChatMessage() {
            const message = chatInput.value.trim();
            if (!message) return;

            // Add user message to UI
            const userMsgDiv = document.createElement('div');
            userMsgDiv.style.cssText = 'background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 8px 8px 0 8px; border: 1px solid rgba(255, 255, 255, 0.2); align-self: flex-end; max-width: 85%;';
            userMsgDiv.innerText = message;
            chatMessages.appendChild(userMsgDiv);
            
            chatInput.value = '';
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Loading state
            const loadingDiv = document.createElement('div');
            loadingDiv.style.cssText = 'background: rgba(52, 211, 153, 0.1); padding: 10px; border-radius: 8px 8px 8px 0; border: 1px solid rgba(52, 211, 153, 0.2); align-self: flex-start; max-width: 85%; color: #a3c2a1; font-style: italic;';
            loadingDiv.innerText = 'Thinking...';
            chatMessages.appendChild(loadingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const res = await fetch(`${API_BASE}/ask_agronomist`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                chatMessages.removeChild(loadingDiv);
                
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                
                const data = await res.json();
                
                const aiMsgDiv = document.createElement('div');
                aiMsgDiv.style.cssText = 'background: rgba(52, 211, 153, 0.1); padding: 10px; border-radius: 8px 8px 8px 0; border: 1px solid rgba(52, 211, 153, 0.2); align-self: flex-start; max-width: 85%;';
                aiMsgDiv.innerText = data.reply || "I'm having trouble processing that right now.";
                chatMessages.appendChild(aiMsgDiv);

            } catch(e) {
                if (loadingDiv.parentNode === chatMessages) {
                    chatMessages.removeChild(loadingDiv);
                }
                const errorDiv = document.createElement('div');
                errorDiv.style.cssText = 'color: #fca5a5; font-size: 11px; align-self: center; text-align: center;';
                errorDiv.innerText = 'Connection error.\n(Did you restart the Flask server?)';
                chatMessages.appendChild(errorDiv);
                // silent fallback for demo
            }
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        chatSendBtn.addEventListener('click', sendChatMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendChatMessage();
        });
    }
});


