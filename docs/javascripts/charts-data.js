const buildDefaultData = () => {
    const engines = ['Inductor', 'AITemplate', 'TensorRT', 'DeepSpeed', 'Baseline', 'Cuda graphs', 'Nvfuser', 'Kernl', 'ONNX Runtime'];
    const sequence = [16, 128, 256, 384, 512];
    const batch = [1, 8];
    const defaultData = [];
    engines.forEach(engine => {
        batch.forEach(batch => {
            sequence.forEach(sequence => {
                defaultData.push({
                    'engine': engine,
                    'batch': batch,
                    'sequence': sequence,
                    'speed': 0
                });
            });
        });
    });
    return defaultData;
}

async function getData() {
    if (window.fetch) {
        const response = await fetch(url)
        if (response.ok) {
            return await response.json();
        } else return undefined;
    }
    return undefined;
}

const url = 'https://api.github.com/repos/javascript-tutorial/en.javascript.info/commits';

export const data = getData(url) || buildDefaultData();


