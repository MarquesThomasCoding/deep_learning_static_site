const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predictButton');
const clearButton = document.getElementById('clearButton');
const resultElement = document.getElementById('result');

let isDrawing = false;

function resetCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
}

resetCanvas();

canvas.addEventListener('mousedown', () => {
    isDrawing = true;
    ctx.beginPath();
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

canvas.addEventListener('mouseleave', () => {
    isDrawing = false;
});

canvas.addEventListener('mousemove', (event) => {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
});

function preprocessCanvas() {
    const tmp = document.createElement('canvas');
    tmp.width = 28;
    tmp.height = 28;
    const tctx = tmp.getContext('2d');
    tctx.fillStyle = 'black';
    tctx.fillRect(0, 0, 28, 28);
    tctx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tctx.getImageData(0, 0, 28, 28).data;
    const pixels = new Float32Array(28 * 28);

    for (let i = 0, p = 0; i < imageData.length; i += 4, p++) {
        const r = imageData[i] / 255;
        pixels[p] = (r - 0.1307) / 0.3081;
    }

    return new ort.Tensor('float32', pixels, [1, 1, 28, 28]);
}

async function predict() {
    try {
        const session = await ort.InferenceSession.create('./model.onnx');
        const inputTensor = preprocessCanvas();
        const inputName = session.inputNames[0];
        const feeds = { [inputName]: inputTensor };
        const outputMap = await session.run(feeds);
        const outputTensor = Object.values(outputMap)[0];

        const expSum = Array.from(outputTensor.data).reduce((sum, val) => sum + Math.exp(val), 0);
        const confidence = (Math.exp(Math.max(...outputTensor.data)) / expSum * 100).toFixed(1);

        const predictedDigit = outputTensor.data.indexOf(Math.max(...outputTensor.data));

        console.log(`Predicted Digit: ${predictedDigit}, Confidence: ${confidence}%`);

        resultElement.textContent = `Résultat : ${predictedDigit} (Confiance : ${confidence}%)`;
    } catch (error) {
        console.error('Erreur lors de la prédiction :', error);
        resultElement.textContent = 'Erreur lors de la prédiction';
    }
}

predictButton.addEventListener('click', predict);
clearButton.addEventListener('click', resetCanvas);