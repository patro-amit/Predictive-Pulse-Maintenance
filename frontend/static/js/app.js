// Predictive Maintenance Frontend JavaScript
// Big Data & Machine Learning Strategies

const API_URL = window.location.origin;
let modelsData = [];
let featuresData = [];
let uploadedFile = null;
let selectedModel = null; // Track selected model

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
        card.className = `model-card ${isBest ? 'best' : ''} ${isSelected ? 'selected' : ''}`;
        card.onclick = () => selectModel(model.name);
        
        card.innerHTML = `
            <div class="model-header">
                <div class="model-name">
                    <i class="fas fa-brain"></i> ${model.name.replace('_', ' ').toUpperCase()}
                </div>
                ${isBest ? '<span class="best-badge">🏆 Best</span>' : ''}
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
                const isSelected = selectedModel === model.name || (index === 0 && !selectedModel);
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

// Generate random value based on feature name
function generateRandomValue(featureName) {
    const name = featureName.toLowerCase();
    
    // Temperature (15-40°C)
    if (name.includes('temp')) {
        return (15 + Math.random() * 25).toFixed(2);
    }
    // Pressure (95-110 kPa)
    else if (name.includes('pressure')) {
        return (95 + Math.random() * 15).toFixed(2);
    }
    // Vibration (0.01-0.15 mm/s)
    else if (name.includes('vibration')) {
        return (0.01 + Math.random() * 0.14).toFixed(3);
    }
    // RPM (1000-2000)
    else if (name.includes('rpm')) {
        return Math.floor(1000 + Math.random() * 1000);
    }
    // Cycle (50-300)
    else if (name.includes('cycle')) {
        return Math.floor(50 + Math.random() * 250);
    }
    // Operating hours (1000-8000)
    else if (name.includes('hour')) {
        return Math.floor(1000 + Math.random() * 7000);
    }
    // Age (1-48 months)
    else if (name.includes('age')) {
        return Math.floor(1 + Math.random() * 47);
    }
    // Load factor (0.5-1.0)
    else if (name.includes('load')) {
        return (0.5 + Math.random() * 0.5).toFixed(2);
    }
    // Generic (10-100)
    else {
        return (10 + Math.random() * 90).toFixed(2);
    }
}

// Load example data with RANDOM values
async function loadExample() {
    showLoading();
    
    try {
        // Generate random values for all features
        let fieldsPopulated = 0;
        featuresData.forEach(feature => {
            const input = document.getElementById(`input-${feature.name}`);
            if (input) {
                input.value = generateRandomValue(feature.name);
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
        
        // Show notification instead of alert
        showNotification(
            '✅ Random example data generated!', 
            `${fieldsPopulated} sensor values populated with realistic random data. Click "Predict Maintenance" to see results.`
        );
        
        console.log(`✅ Example data loaded: ${fieldsPopulated} fields populated`);
    } catch (error) {
        console.error('Error loading example:', error);
        showNotification('❌ Error', error.message);
        hideLoading();
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
