export const buildDefaultData = () => {
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

export async function fetchData(url) {
    if (window.fetch) {
        const response = await fetch(url)
        if (response.ok && response.status === 200) {
            return await response.json();
        } else return undefined;
    }
    return undefined;
}


