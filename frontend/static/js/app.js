// Predictive Maintenance Frontend JavaScript
// Big Data & Machine Learning Strategies

const API_URL = window.location.origin;
let modelsData = [];
let featuresData = [];
let uploadedFile = null;
let selectedModel = 'gradient_boosting'; // Default to gradient_boosting (best balance)

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadSystemInfo();
    await loadModels();
    await loadFeatures();
    setupEventListeners();
});

// Load system info
async function loadSystemInfo() {
    try {
        console.log('Loading system info from:', `${API_URL}/health`);
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        console.log('System info loaded:', data);
        
        // Update all instances of models-count
        document.querySelectorAll('#models-count').forEach(el => {
            el.textContent = data.models_loaded;
        });
        // Update all instances of features-count
        document.querySelectorAll('#features-count').forEach(el => {
            el.textContent = data.features_count;
        });
        // Update all instances of best-accuracy
        document.querySelectorAll('#best-accuracy').forEach(el => {
            el.textContent = data.models_loaded > 0 ? '92.30%' : '0%';
        });
    } catch (error) {
        console.error('Error loading system info:', error);
        alert('Failed to connect to server. Please refresh the page.');
    }
}

// Load available models
async function loadModels() {
    try {
        console.log('Loading models from:', `${API_URL}/models`);
        const response = await fetch(`${API_URL}/models`);
        const data = await response.json();
        modelsData = data.models;
        console.log('Models loaded:', modelsData.length, 'models');
        
        // Update best accuracy
        if (modelsData.length > 0) {
            const bestAccuracy = modelsData[0].accuracy; // Already sorted by backend
            document.querySelectorAll('#best-accuracy').forEach(el => {
                el.textContent = (bestAccuracy * 100).toFixed(2) + '%';
            });
            
            // Update best model name
            const bestModelName = modelsData[0].name.replace('_', ' ').toUpperCase();
            const bestModelEl = document.getElementById('best-model-name');
            if (bestModelEl) {
                bestModelEl.textContent = bestModelName;
            }
        }
        
        // Render model cards
        console.log('Rendering model cards...');
        renderModelCards();
        
        // Render model selector
        console.log('Rendering model selector...');
        renderModelSelector();
        
        // Select best model by default
        if (modelsData.length > 0) {
            selectedModel = modelsData[0].name;
            console.log('Default model selected:', selectedModel);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        alert('Failed to load ML models. Please refresh the page.');
    }
}

// Render model cards with selection
function renderModelCards() {
    const container = document.getElementById('models-container');
    container.innerHTML = '';
    
    modelsData.forEach((model, index) => {
        const isBest = index === 0;
        const isSelected = selectedModel === model.name;
        const card = document.createElement('div');
        card.className = `model-card ${isSelected ? 'selected' : ''}`;
        card.onclick = () => selectModel(model.name);
        
        card.innerHTML = `
            <div class="model-header">
                <div class="model-name">
                    <i class="fas fa-brain"></i> ${model.name.replace('_', ' ').toUpperCase()}
                </div>
                ${isSelected ? '<span class="selected-badge">✓ Selected</span>' : ''}
            </div>
            <div class="model-metrics">
                <div class="metric">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value">${(model.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div class="accuracy-bar">
                    <div class="accuracy-fill" style="width: ${model.accuracy * 100}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">F1 Score</span>
                    <span class="metric-value">${model.f1_score.toFixed(2)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">AUROC</span>
                    <span class="metric-value">${model.auroc.toFixed(2)}</span>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Select model function
function selectModel(modelName) {
    selectedModel = modelName;
    console.log('✅ Model selected:', modelName);
    renderModelCards();
    
    // Update model selector if exists
    const modelOptions = document.querySelectorAll('.model-option');
    modelOptions.forEach(option => {
        if (option.dataset.model === modelName) {
            option.classList.add('selected');
        } else {
            option.classList.remove('selected');
        }
    });
    
    // Update selected model display
    const selectedDisplay = document.getElementById('selected-model-display');
    if (selectedDisplay) {
        selectedDisplay.textContent = modelName.replace('_', ' ').toUpperCase();
    }
    
    // Show notification
    const modelDisplayName = modelName.replace('_', ' ').toUpperCase();
    const accuracy = modelsData.find(m => m.name === modelName)?.accuracy || 0;
    showNotification(`🎯 ${modelDisplayName} Selected!`, `Accuracy: ${(accuracy * 100).toFixed(2)}% - This model will be used for predictions`);
}

// Render model selector
function renderModelSelector() {
    const container = document.getElementById('model-selector-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    const selector = document.createElement('div');
    selector.className = 'model-selector';
    
    selector.innerHTML = `
        <h4><i class="fas fa-robot"></i> Select Model for Prediction</h4>
        <p style="color: var(--text-gray); margin-bottom: 1rem; font-size: 0.95rem;">
            Click a model below - Your selection will be used for predictions
        </p>
        <div class="model-options">
            ${modelsData.map((model, index) => {
                // Default to gradient_boosting for best practical results
                const isSelected = selectedModel === model.name;
                return `
                    <div class="model-option ${isSelected ? 'selected' : ''}" 
                         data-model="${model.name}"
                         onclick="selectModel('${model.name}')">
                        <div class="model-option-name">
                            ${isSelected ? '✓ ' : ''}${model.name.replace('_', ' ').toUpperCase()}
                        </div>
                        <div class="model-option-accuracy">
                            ${(model.accuracy * 100).toFixed(2)}% accuracy
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
    
    container.appendChild(selector);
}

// Load features schema
async function loadFeatures() {
    try {
        const response = await fetch(`${API_URL}/schema`);
        const data = await response.json();
        featuresData = data.features;
        
        // Generate sensor input fields
        generateSensorInputs();
    } catch (error) {
        console.error('Error loading features:', error);
    }
}

// Generate sensor input fields with grouping
function generateSensorInputs() {
    const container = document.getElementById('sensor-inputs');
    container.innerHTML = '';
    
    // Define important features to show by default
    const importantFeatures = ['cycle', 'temp_avg', 'pressure_avg', 'vibration_avg', 'rpm_avg', 'setting1', 'setting2', 'setting3'];
    
    // Group features
    const important = [];
    const advanced = [];
    
    featuresData.forEach(feature => {
        if (importantFeatures.includes(feature.name)) {
            important.push(feature);
        } else {
            advanced.push(feature);
        }
    });
    
    // Add info banner
    const infoBanner = document.createElement('div');
    infoBanner.style.cssText = `
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(181, 55, 242, 0.1));
        border-left: 4px solid var(--neon-blue);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: var(--text-gray);
    `;
    infoBanner.innerHTML = `
        <strong style="color: var(--neon-blue);">💡 Quick Input:</strong> 
        Only 8 key sensor readings needed! Advanced sensors are optional.
        <br><small style="opacity: 0.8;">Tip: Click "Use Example" button above to load random realistic data</small>
    `;
    container.appendChild(infoBanner);
    
    // Render important features
    important.forEach(feature => {
        const div = document.createElement('div');
        div.className = 'sensor-input';
        
        let placeholder = "Enter value";
        let description = "";
        
        if (feature.name === 'cycle') {
            placeholder = "e.g., 150";
            description = "Operating cycle number";
        } else if (feature.name.includes('temp')) {
            placeholder = "e.g., 22.5";
            description = "Temperature in °C";
        } else if (feature.name.includes('pressure')) {
            placeholder = "e.g., 101.3";
            description = "Pressure in kPa";
        } else if (feature.name.includes('vibration')) {
            placeholder = "e.g., 0.05";
            description = "Vibration in mm/s";
        } else if (feature.name.includes('rpm')) {
            placeholder = "e.g., 1500";
            description = "Rotations per minute";
        } else if (feature.name === 'setting1') {
            placeholder = "e.g., 100";
            description = "Operational setting 1";
        } else if (feature.name === 'setting2') {
            placeholder = "e.g., 200";
            description = "Operational setting 2";
        } else if (feature.name === 'setting3') {
            placeholder = "e.g., 300";
            description = "Operational setting 3";
        }
        
        div.innerHTML = `
            <label>
                ${feature.name.replace(/_/g, ' ').toUpperCase()}
                ${description ? `<br><small style="font-weight: normal; opacity: 0.7; font-size: 0.85rem;">${description}</small>` : ''}
            </label>
            <input type="number" 
                   step="0.01" 
                   id="input-${feature.name}" 
                   placeholder="${placeholder}"
                   value="0">
        `;
        container.appendChild(div);
    });
    
    // Add collapsible advanced section
    if (advanced.length > 0) {
        const advancedSection = document.createElement('div');
        advancedSection.style.cssText = 'margin-top: 2rem;';
        
        const toggleButton = document.createElement('button');
        toggleButton.type = 'button';
        toggleButton.style.cssText = `
            width: 100%;
            padding: 1rem 1.5rem;
            background: rgba(0, 212, 255, 0.1);
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            color: var(--neon-blue);
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: space-between;
        `;
        toggleButton.innerHTML = `
            <span>⚙️ Advanced Sensor Readings (${advanced.length} optional)</span>
            <span id="toggle-icon">▼</span>
        `;
        
        const advancedGrid = document.createElement('div');
        advancedGrid.id = 'advanced-sensors';
        advancedGrid.className = 'sensor-grid';
        advancedGrid.style.display = 'none';
        advancedGrid.style.marginTop = '1rem';
        
        advanced.forEach(feature => {
            const div = document.createElement('div');
            div.className = 'sensor-input';
            div.innerHTML = `
                <label>${feature.name.toUpperCase()}</label>
                <input type="number" 
                       step="0.01" 
                       id="input-${feature.name}" 
                       placeholder="Optional"
                       value="0">
            `;
            advancedGrid.appendChild(div);
        });
        
        toggleButton.onclick = () => {
            const isHidden = advancedGrid.style.display === 'none';
            advancedGrid.style.display = isHidden ? 'grid' : 'none';
            document.getElementById('toggle-icon').textContent = isHidden ? '▲' : '▼';
            toggleButton.style.background = isHidden ? 
                'rgba(0, 212, 255, 0.2)' : 'rgba(0, 212, 255, 0.1)';
        };
        
        advancedSection.appendChild(toggleButton);
        advancedSection.appendChild(advancedGrid);
        container.appendChild(advancedSection);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Upload area
    const uploadArea = document.getElementById('upload-area');
    const fileUpload = document.getElementById('file-upload');
    
    uploadArea.addEventListener('click', () => fileUpload.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
        uploadArea.style.background = '#eff6ff';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.background = 'white';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.background = 'white';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// Handle file selection
function handleFileSelect(file) {
    if (file.type !== 'text/csv') {
        alert('Please select a CSV file');
        return;
    }
    
    uploadedFile = file;
    document.getElementById('upload-area').innerHTML = `
        <i class="fas fa-file-csv" style="color: var(--success-color);"></i>
        <p><strong>${file.name}</strong></p>
        <small>${(file.size / 1024).toFixed(2)} KB</small>
    `;
    document.getElementById('predict-file-btn').style.display = 'block';
}

// Show input method
function showMethod(method, event) {
    // Update buttons
    document.querySelectorAll('.method-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Find and activate the correct button
    if (event && event.target) {
        event.target.closest('.method-btn').classList.add('active');
    } else {
        // If no event (called programmatically), find button by method name
        document.querySelectorAll('.method-btn').forEach(btn => {
            const btnText = btn.textContent.toLowerCase();
            if ((method === 'manual' && btnText.includes('manual')) ||
                (method === 'file' && btnText.includes('upload')) ||
                (method === 'example' && btnText.includes('example'))) {
                btn.classList.add('active');
            }
        });
    }
    
    // Update content
    document.querySelectorAll('.input-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${method}-input`).classList.add('active');
}

// Scroll to section
function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
}

// Show loading overlay
function showLoading() {
    document.getElementById('loading-overlay').classList.add('active');
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

// Predict from manual input
async function predictManual() {
    showLoading();
    
    try {
        // Collect input values
        const inputs = {};
        let hasValues = false;
        
        featuresData.forEach(feature => {
            const input = document.getElementById(`input-${feature.name}`);
            const value = parseFloat(input.value);
            
            if (!isNaN(value)) {
                inputs[feature.name] = value;
                hasValues = true;
            } else {
                inputs[feature.name] = 0;
            }
        });
        
        if (!hasValues) {
            alert('Please enter at least some sensor values');
            hideLoading();
            return;
        }
        
        // Use selected model if specified - FIX: Use path parameter, not query parameter
        const endpoint = selectedModel ? 
            `${API_URL}/predict/${encodeURIComponent(selectedModel)}` : 
            `${API_URL}/predict`;
        
        console.log('🚀 Making prediction with endpoint:', endpoint);
        console.log('📊 Selected model:', selectedModel || 'Best Model');
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ inputs: [inputs] })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Show which model was actually used (from backend response)
        const actualModelUsed = data.model_used || selectedModel || 'Best Model';
        console.log('✅ Prediction complete!');
        console.log('📊 Model used:', actualModelUsed);
        console.log('🎯 Result:', data.predictions[0]);
        console.log('💯 Confidence:', (data.confidence[0] * 100).toFixed(1) + '%');
        
        showNotification(
            '✅ Prediction Complete!', 
            `Used: ${actualModelUsed.replace('_', ' ').toUpperCase()} | Result: ${data.predictions[0]}`
        );
        
        displayResults([data]);
        
        // Scroll to results
        scrollToSection('predict');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Show notification
function showNotification(title, message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 30px;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.95), rgba(181, 55, 242, 0.95));
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.5), 0 10px 40px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        min-width: 320px;
        animation: slideInRight 0.3s ease-out;
        border: 1px solid rgba(0, 212, 255, 0.5);
    `;
    
    notification.innerHTML = `
        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">${title}</div>
        <div style="font-size: 0.95rem; opacity: 0.95;">${message}</div>
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Add CSS animation for notification
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Predict from file
async function predictFile() {
    if (!uploadedFile) {
        alert('Please select a file first');
        return;
    }
    
    showLoading();
    
    try {
        // Parse CSV file
        const text = await uploadedFile.text();
        const inputs = parseCSV(text);
        
        // Use comparison endpoint for file uploads
        const response = await fetch(`${API_URL}/predict/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ inputs })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        displayComparisonResults(data);
        
        // Scroll to results
        scrollToSection('predict');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Parse CSV file
function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const inputs = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
            row[header] = parseFloat(values[index]) || 0;
        });
        inputs.push(row);
    }
    
    return inputs;
}

// Generate random value based on feature name and scenario
// Based on ACTUAL failure patterns from NASA C-MAPSS dataset
function generateRandomValue(featureName, scenario = 'normal') {
    const name = featureName.toLowerCase();
    
    // Key failure indicators from dataset analysis:
    // s9, s11, s12, s20 are MUCH HIGHER in failures (2-3x normal)
    // s7, s14, s21 are LOWER in failures
    // s1-s4 slightly higher in failures
    
    // Settings
    if (name === 'setting1') {
        return (0.2 + Math.random() * 0.2).toFixed(6);
    }
    if (name === 'setting2') {
        return (-0.06 + Math.random() * 0.06).toFixed(6);
    }
    if (name === 'setting3') {
        return (95 + Math.random() * 10).toFixed(2);
    }
    
    // Sensor-specific patterns (from dataset analysis)
    // s1-s4: slightly higher in failures
    if (['s1', 's2', 's3', 's4'].includes(name)) {
        if (scenario === 'normal') {
            return (95 + Math.random() * 15).toFixed(2);  // 95-110 normal
        } else if (scenario === 'high_risk') {
            return (100 + Math.random() * 15).toFixed(2);  // 100-115 elevated
        } else {
            return (105 + Math.random() * 20).toFixed(2);  // 105-125 high
        }
    }
    
    // s7: LOWER in failures
    if (name === 's7') {
        if (scenario === 'normal') {
            return (46 + Math.random() * 6).toFixed(2);  // 46-52 normal
        } else if (scenario === 'high_risk') {
            return (44 + Math.random() * 5).toFixed(2);  // 44-49 low
        } else {
            return (42 + Math.random() * 4).toFixed(2);  // 42-46 very low
        }
    }
    
    // s9, s11, s12: MUCH HIGHER in failures (KEY INDICATORS!)
    if (['s9', 's11', 's12'].includes(name)) {
        if (scenario === 'normal') {
            // Based on bigdata CSV: s9 mean=0.86, s11 mean=0.95, s12 mean=1.21
            // Normal range: mean ± 0.5*std (healthy operation)
            if (name === 's9') return (0.4 + Math.random() * 0.9).toFixed(2);  // 0.4-1.3 (around mean 0.86)
            if (name === 's11') return (0.4 + Math.random() * 1.1).toFixed(2);  // 0.4-1.5 (around mean 0.95)
            if (name === 's12') return (0.6 + Math.random() * 1.2).toFixed(2);  // 0.6-1.8 (around mean 1.21)
        } else if (scenario === 'high_risk') {
            return (1.5 + Math.random() * 1.5).toFixed(2);  // 1.5-3.0 high
        } else {
            return (2.5 + Math.random() * 2.0).toFixed(2);  // 2.5-4.5 critical
        }
    }
    
    // s14: Lower in failures
    if (name === 's14') {
        if (scenario === 'normal') {
            return (3000 + Math.random() * 400).toFixed(2);  // 3000-3400 normal
        } else if (scenario === 'high_risk') {
            return (2900 + Math.random() * 300).toFixed(2);  // 2900-3200 low
        } else {
            return (2700 + Math.random() * 300).toFixed(2);  // 2700-3000 very low
        }
    }
    
    // s20: HIGHER in failures (KEY INDICATOR!)
    if (name === 's20') {
        if (scenario === 'normal') {
            // Based on bigdata CSV: s20 mean=0.32, std=0.39
            // Normal range: mean ± 0.3*std (healthy operation)
            return (0.20 + Math.random() * 0.25).toFixed(2);  // 0.20-0.45 (around mean 0.32)
        } else if (scenario === 'high_risk') {
            return (0.5 + Math.random() * 0.4).toFixed(2);  // 0.5-0.9 high
        } else {
            return (0.8 + Math.random() * 0.5).toFixed(2);  // 0.8-1.3 critical
        }
    }
    
    // s21: LOWER in failures
    if (name === 's21') {
        if (scenario === 'normal') {
            return (95 + Math.random() * 15).toFixed(2);  // 95-110 normal
        } else if (scenario === 'high_risk') {
            return (90 + Math.random() * 12).toFixed(2);  // 90-102 low
        } else {
            return (85 + Math.random() * 10).toFixed(2);  // 85-95 very low
        }
    }
    
    // Other sensors (s5, s6, s8, s10, s13, s15-s19): moderate changes
    if (name.startsWith('s') && name.length <= 3) {
        if (scenario === 'normal') {
            return (85 + Math.random() * 30).toFixed(2);  // 85-115
        } else if (scenario === 'high_risk') {
            return (80 + Math.random() * 35).toFixed(2);  // 80-115
        } else {
            return (75 + Math.random() * 40).toFixed(2);  // 75-115
        }
    }
    
    // Generic fallback
    if (scenario === 'normal') {
        return (50 + Math.random() * 50).toFixed(2);
    } else if (scenario === 'high_risk') {
        return (40 + Math.random() * 60).toFixed(2);
    } else {
        return (30 + Math.random() * 70).toFixed(2);
    }
}

// Load example data with selected scenario
async function loadExample(scenario = null) {
    showLoading();
    
    try {
        // If no scenario provided, show selection dialog
        if (!scenario) {
            hideLoading();
            showScenarioDialog();
            return;
        }
        
        // Generate values based on scenario
        let fieldsPopulated = 0;
        featuresData.forEach(feature => {
            const input = document.getElementById(`input-${feature.name}`);
            if (input) {
                input.value = generateRandomValue(feature.name, scenario);
                fieldsPopulated++;
            }
        });
        
        // Switch to manual input tab
        showMethod('manual');
        
        // Scroll to form
        setTimeout(() => {
            document.getElementById('manual-input').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
        
        hideLoading();
        
        // Show notification
        const scenarioNames = {
            'normal': 'Normal Operating Conditions',
            'high_risk': 'High Risk - Abnormal Readings',
            'critical': 'Critical - Failure Imminent'
        };
        
        showNotification(
            '✅ Sample data generated!', 
            `${fieldsPopulated} sensor values populated with ${scenarioNames[scenario]}. Click "Predict Maintenance" to see results.`
        );
        
        console.log(`✅ Example data loaded: ${fieldsPopulated} fields populated (${scenario} scenario)`);
    } catch (error) {
        console.error('Error loading example:', error);
        showNotification('❌ Error', error.message);
        hideLoading();
    }
}

// Show scenario selection dialog
function showScenarioDialog() {
    const dialog = document.createElement('div');
    dialog.className = 'scenario-dialog-overlay';
    dialog.innerHTML = `
        <div class="scenario-dialog">
            <h3>Select Sample Data Scenario</h3>
            <p>Choose the type of sample data to generate:</p>
            <div class="scenario-options">
                <button class="scenario-btn normal" onclick="selectScenario('normal')">
                    <i class="fas fa-check-circle"></i>
                    <strong>Normal Operation</strong>
                    <span>Healthy machine, no maintenance needed</span>
                </button>
                <button class="scenario-btn high-risk" onclick="selectScenario('high_risk')">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>High Risk</strong>
                    <span>Abnormal readings, maintenance may be needed</span>
                </button>
                <button class="scenario-btn critical" onclick="selectScenario('critical')">
                    <i class="fas fa-times-circle"></i>
                    <strong>Critical Failure</strong>
                    <span>Extreme values, failure imminent</span>
                </button>
            </div>
            <button class="cancel-btn" onclick="closeScenarioDialog()">Cancel</button>
        </div>
    `;
    document.body.appendChild(dialog);
}

// Select scenario and close dialog
function selectScenario(scenario) {
    closeScenarioDialog();
    loadExample(scenario);
}

// Close scenario dialog
function closeScenarioDialog() {
    const dialog = document.querySelector('.scenario-dialog-overlay');
    if (dialog) {
        dialog.remove();
    }
}

// Display results
function displayResults(results) {
    const container = document.getElementById('results-content');
    container.innerHTML = '';
    
    results.forEach((result, index) => {
        const prediction = result.predictions[0];
        const probability = result.probability[0];
        const confidence = result.confidence[0];
        const isMaintenance = prediction.includes('Maintenance') || prediction.includes('Required');
        
        const resultDiv = document.createElement('div');
        resultDiv.className = `prediction-result ${isMaintenance ? 'maintenance' : 'normal'}`;
        
        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="result-status">
                    <i class="fas ${isMaintenance ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                    ${prediction}
                </div>
                <div class="confidence-badge">${(confidence * 100).toFixed(1)}% Confident</div>
            </div>
            <div class="result-details">
                <div class="detail-item">
                    <span class="detail-label">Model Used</span>
                    <span class="detail-value">${result.model_used || 'Best Model'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Probability</span>
                    <span class="detail-value">${(probability * 100).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Model Accuracy</span>
                    <span class="detail-value">${((result.model_metrics?.accuracy || 0) * 100).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">F1 Score</span>
                    <span class="detail-value">${(result.model_metrics?.f1 || 0).toFixed(4)}</span>
                </div>
            </div>
        `;
        
        container.appendChild(resultDiv);
    });
}

// Display comparison results
function displayComparisonResults(data) {
    const container = document.getElementById('results-content');
    container.innerHTML = '<h4 style="margin-bottom: 1rem;">Model Comparison Results</h4>';
    
    data.comparison.forEach(result => {
        const prediction = result.predictions[0];
        const probability = result.probability[0];
        const isMaintenance = prediction.includes('Maintenance') || prediction.includes('Required');
        
        const resultDiv = document.createElement('div');
        resultDiv.className = `prediction-result ${isMaintenance ? 'maintenance' : 'normal'}`;
        
        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="result-status">
                    <i class="fas fa-brain"></i>
                    ${result.model.toUpperCase()}
                </div>
                <div class="confidence-badge">${(result.confidence[0] * 100).toFixed(1)}%</div>
            </div>
            <div class="result-details">
                <div class="detail-item">
                    <span class="detail-label">Prediction</span>
                    <span class="detail-value">${prediction}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Probability</span>
                    <span class="detail-value">${(probability * 100).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Accuracy</span>
                    <span class="detail-value">${((result.metrics?.accuracy || 0) * 100).toFixed(2)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">AUROC</span>
                    <span class="detail-value">${(result.metrics?.auroc || 0).toFixed(4)}</span>
                </div>
            </div>
        `;
        
        container.appendChild(resultDiv);
    });
}
