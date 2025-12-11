// DOM Elements
const canvas = document.getElementById('drawing-canvas');
const ctx = canvas?.getContext('2d');
const clearBtn = document.getElementById('clear-button');
const predictBtn = document.getElementById('predict-button');
const explicationDiv = document.getElementById('card-explication');
const chargementDiv = document.getElementById('card-chargement');

// Variables
let session = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Init
async function init() {
    setupCanvas();
    setupEventListeners();
    await loadModel();
}

// Canvas setup
function setupCanvas() {
    if (!ctx) return;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#FFF';
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

// Model loading
async function loadModel() {
    try {
        session = await ort.InferenceSession.create('model.onnx');
        chargementDiv.className = 'success';
        chargementDiv.innerHTML = '<p>Modèle chargé ✓</p>';
        explicationDiv.innerHTML = '<p>Dessinez un chiffre pour commencer</p>';
        clearBtn.disabled = false;
        predictBtn.disabled = false;
    } catch (e) {
        chargementDiv.className = 'error';
        chargementDiv.innerHTML = `<p>Erreur: ${e.message}</p>`;
    }
}

// Event listeners
function setupEventListeners() {
    canvas?.addEventListener('mousedown', startDrawing);
    canvas?.addEventListener('mousemove', draw);
    canvas?.addEventListener('mouseup', stopDrawing);
    canvas?.addEventListener('mouseout', stopDrawing);
    canvas?.addEventListener('touchstart', e => { e.preventDefault(); startDrawing(e); });
    canvas?.addEventListener('touchmove', e => { e.preventDefault(); draw(e); });
    canvas?.addEventListener('touchend', stopDrawing);
    clearBtn?.addEventListener('click', clearCanvas);
    predictBtn?.addEventListener('click', predict);
}

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getPosition(e);
}

function draw(e) {
    if (!isDrawing || !ctx) return;
    const [x, y] = getPosition(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

function getPosition(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX || e.touches?.[0]?.clientX || 0;
    const clientY = e.clientY || e.touches?.[0]?.clientY || 0;
    return [clientX - rect.left, clientY - rect.top];
}

// Clear canvas
function clearCanvas() {
    if (!ctx) return;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    explicationDiv.innerHTML = '<p>Dessinez un chiffre pour commencer</p>';
}

// Preprocessing
function preprocessCanvas() {
    const temp = document.createElement('canvas');
    temp.width = temp.height = 28;
    const tempCtx = temp.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    return Float32Array.from({ length: 784 }, (_, i) => imageData.data[i * 4] / 255);
}

// Softmax
function softMax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(l => Math.exp(l - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(e => e / sum);
}

// Prediction
async function predict() {
    if (!session) return;
    
    try {
        explicationDiv.innerHTML = '<p style="color: #007bff;">Prédiction en cours...</p>';
        
        const input = preprocessCanvas();
        const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
        const results = await session.run({ [session.inputNames[0]]: tensor });
        const probs = softMax(results[session.outputNames[0]].data);
        const predicted = probs.indexOf(Math.max(...probs));
        
        displayPrediction(predicted, probs);
    } catch (e) {
        explicationDiv.innerHTML = `<p style="color: red;">Erreur: ${e.message}</p>`;
    }
}

// Display results
function displayPrediction(predicted, probs) {
    const confidence = Math.max(...probs);
    const top3 = probs
        .map((prob, idx) => ({ class: idx, prob }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 3);
    
    let html = `<div class="prediction-result">Chiffre prédit : ${predicted}</div>
                <p>Confiance : ${(confidence * 100).toFixed(1)}%</p>
                <div class="confidence-bars">`;
    
    top3.forEach(item => {
        const percent = (item.prob * 100).toFixed(1);
        html += `<div class="confidence-bar">
                    <span class="confidence-label">${item.class}</span>
                    <div class="confidence-fill-container">
                        <div class="confidence-fill" style="width: ${percent}%"></div>
                    </div>
                    <span class="confidence-value">${percent}%</span>
                </div>`;
    });
    
    explicationDiv.innerHTML = html + '</div>';
}

// Launch
document.addEventListener('DOMContentLoaded', init);

// let session = null;
// let isDrawing = false;
// let lastX = 0;
// let lastY = 0;

// const canvas = document.getElementById('drawing-canvas');
// const ctx = canvas ? canvas.getContext('2d') : null; 
// const clearButton = document.getElementById('clear-button');
// const predictButton = document.getElementById('predict-button');
// const explicationDiv = document.getElementById('card-explication');
// const chargementDiv = document.getElementById('card-chargement');

// function log(message, type = 'info') {
//     const timestamp = new Date().toLocaleTimeString('fr-FR');
//     const prefix = `[${timestamp}]`;
//     if (type === 'error') {
//         console.error(prefix, message);
//     } else if (type === 'warn') {
//         console.warn(prefix, message);
//     } else {
//         console.log(prefix, message);
//     }
// }

// if (ctx) {
//     ctx.strokeStyle = '#FFFFFF'; 
//     ctx.lineWidth = 5;          // Épaisseur de trait 
//     ctx.lineJoin = 'round';
//     ctx.lineCap = 'round';
// }

// async function loadModel() {
//     if (!chargementDiv || !clearButton || !predictButton) {
//         log("Erreur: Éléments du DOM manquants.", 'error');
//         return; 
//     }

//     try {
//         log('=== DÉBUT DU CHARGEMENT DU MODÈLE ===');
//         chargementDiv.innerHTML = '<p>Chargement du modèle...</p>';
        
//         log('Vérification de ONNX Runtime...', 'info');
//         if (typeof ort === 'undefined') {
//             throw new Error('ONNX Runtime non chargé. Assurez-vous que la balise script CDN est présente.');
//         }
//         log('✓ ONNX Runtime détecté', 'info');
        
//         log('Tentative de chargement de model.onnx...', 'info');
//         session = await ort.InferenceSession.create('model.onnx'); 
        
//         log('✓ Modèle chargé avec succès!', 'info');
//         log('Inputs du modèle: ' + session.inputNames.join(', '), 'info');
//         log('Outputs du modèle: ' + session.outputNames.join(', '), 'info');
        
//         chargementDiv.className = 'success';
//         chargementDiv.innerHTML = '<p>Modèle chargé avec succès ✓</p>';
//         explicationDiv.innerHTML = '<p>Dessinez un chiffre pour commencer</p>';
        
//         clearButton.disabled = false;
//         predictButton.disabled = false;
        
//         log('=== CHARGEMENT TERMINÉ ===');
//     } catch (error) {
//         log('=== ERREUR DE CHARGEMENT ===', 'error');
//         chargementDiv.className = 'error';
//         chargementDiv.innerHTML = `
//             <p><strong>Erreur de chargement</strong></p>
//             <p style="font-size: 0.9em;">Message: ${error.message}</p>
//             <p style="font-size: 0.85em;">Vérifiez la présence de <strong>model.onnx</strong> et du serveur local.</p>
//         `;
//         explicationDiv.innerHTML = '<p>Le modèle n\'a pas pu être chargé</p>';
//     }
// }

// function clearCanvas() {
//     if (!ctx) return;
    
//     ctx.fillStyle = '#000000';
//     ctx.fillRect(0, 0, canvas.width, canvas.height);
    
//     if (explicationDiv) {
//         explicationDiv.innerHTML = '<p>Dessinez un chiffre pour commencer</p>';
//     }
// }

// function getMousePos(e) {
//     const rect = canvas.getBoundingClientRect();
//     const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
//     const clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
    
//     return {
//         x: clientX - rect.left,
//         y: clientY - rect.top
//     };
// }

// function startDrawing(e) {
//     if (!ctx) return;
//     isDrawing = true;
//     const pos = getMousePos(e);
//     lastX = pos.x;
//     lastY = pos.y;
    
//     ctx.beginPath();
//     ctx.moveTo(lastX, lastY);
    
//     ctx.lineTo(lastX + 0.01, lastY); 
//     ctx.stroke();
//     ctx.closePath(); 
    
//     ctx.beginPath();
//     ctx.moveTo(lastX, lastY);
// }

// function draw(e) {
//     if (!isDrawing || !ctx) return;
//     const pos = getMousePos(e);
    
//     ctx.lineTo(pos.x, pos.y);
//     ctx.stroke();
    
//     lastX = pos.x;
//     lastY = pos.y;
// }

// function stopDrawing() {
//     if (isDrawing && ctx) {
//         ctx.closePath();
//         isDrawing = false;
//     }
// }

// function preprocessCanvas() {
//     log('=== DÉBUT DU PRÉTRAITEMENT ===');
    
//     const tempCanvas = document.createElement('canvas');
//     tempCanvas.width = 28;
//     tempCanvas.height = 28;
//     const tempCtx = tempCanvas.getContext('2d');
    
//     tempCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);
    
//     const imageData = tempCtx.getImageData(0, 0, 28, 28);
//     const data = imageData.data;
    
//     const input = new Float32Array(28 * 28);
    
//     for (let i = 0; i < 28 * 28; i++) {
//         input[i] = data[i * 4] / 255.0; 
//     }
    
//     log('Normalisation terminée. Taille des données (784): ' + input.length, 'info');
//     return input;
// }

// async function predict() {
//     log('=== DÉBUT DE LA PRÉDICTION ===');
    
//     if (!session) {
//         log('Erreur: Session non initialisée', 'error');
//         explicationDiv.innerHTML = '<p style="color: red;">Le modèle n\'est pas chargé</p>';
//         return;
//     }

//     try {
//         explicationDiv.innerHTML = '<p style="color: #007bff;">Prédiction en cours...</p>';
        
//         const inputData = preprocessCanvas();
 
//         const inputShape = [1, 28, 28]; 
        
//         if (inputData.length !== 784) {
//              throw new Error(`Taille des données incorrecte: ${inputData.length} attendu 784.`);
//         }
        
//         const tensor = new ort.Tensor('float32', inputData, inputShape);
//         log(`Tensor créé - Type: ${tensor.type}, Forme: [${tensor.dims.join(', ')}]`, 'info');
        
//         const feeds = { [session.inputNames[0]]: tensor };
        
//         const startTime = performance.now();
//         const results = await session.run(feeds);
//         const endTime = performance.now();
        
//         log(`✓ Inférence terminée en ${(endTime - startTime).toFixed(2)}ms`, 'info');
        
//         const output = results[session.outputNames[0]];
//         const logits = output.data;
        
//         const exp = Array.from(logits).map(x => Math.exp(x));
//         const sumExp = exp.reduce((a, b) => a + b, 0);
//         const probabilities = exp.map(x => x / sumExp);
        
//         const predictedClass = probabilities.indexOf(Math.max(...probabilities));
//         const confidence = probabilities[predictedClass];
        
//         log(`✓ RÉSULTAT: Chiffre ${predictedClass} (confiance: ${(confidence * 100).toFixed(1)}%)`, 'info');
        
//         displayPrediction(predictedClass, probabilities);
        
//     } catch (error) {
//         log('=== ERREUR PENDANT LA PRÉDICTION ===', 'error');
//         log('Message: ' + error.message, 'error');
//         explicationDiv.innerHTML = `<p style="color: red;">Erreur lors de la prédiction: ${error.message}</p>`;
//     }
// }


// function displayPrediction(predictedClass, probabilities) {
//     const confidence = (probabilities[predictedClass] * 100).toFixed(1);
    
//     let html = `
//         <div class="prediction-result">Chiffre prédit : ${predictedClass}</div>
//         <p>Confiance : ${confidence}%</p>
//         <div class="confidence-bars">
//     `;
    
//     const sorted = probabilities
//         .map((prob, idx) => ({ class: idx, prob: prob }))
//         .sort((a, b) => b.prob - a.prob)
//         .slice(0, 3);
    
//     sorted.forEach(item => {
//         const percent = (item.prob * 100).toFixed(1);
//         html += `
//             <div class="confidence-bar">
//                 <span class="confidence-label">${item.class}</span>
//                 <div class="confidence-fill-container">
//                     <div class="confidence-fill" style="width: ${percent}%"></div>
//                 </div>
//                 <span class="confidence-value">${percent}%</span>
//             </div>
//         `;
//     });
    
//     html += '</div>';
//     explicationDiv.innerHTML = html;
// }

// // ------------------------------------------------------------------
// // Gestionnaires d'événements
// // ------------------------------------------------------------------

// if (canvas) {
//     canvas.addEventListener('mousedown', startDrawing);
//     canvas.addEventListener('mousemove', draw);
//     document.addEventListener('mouseup', stopDrawing); 
//     canvas.addEventListener('mouseout', stopDrawing);

//     canvas.addEventListener('touchstart', (e) => {
//         e.preventDefault();
//         const touch = e.touches[0];
//         const mouseEvent = new MouseEvent('mousedown', {
//             clientX: touch.clientX,
//             clientY: touch.clientY
//         });
//         canvas.dispatchEvent(mouseEvent);
//     });

//     canvas.addEventListener('touchmove', (e) => {
//         e.preventDefault();
//         const touch = e.touches[0];
//         const mouseEvent = new MouseEvent('mousemove', {
//             clientX: touch.clientX,
//             clientY: touch.clientY
//         });
//         canvas.dispatchEvent(mouseEvent);
//     });

//     canvas.addEventListener('touchend', (e) => {
//         e.preventDefault();
//         const mouseEvent = new MouseEvent('mouseup', {});
//         canvas.dispatchEvent(mouseEvent);
//     });
// }


// if (clearButton) {
//     clearButton.addEventListener('click', clearCanvas);
// }

// if (predictButton) {
//     predictButton.addEventListener('click', predict);
// }


// document.addEventListener('DOMContentLoaded', () => {
//     clearCanvas();
//     loadModel();
// });