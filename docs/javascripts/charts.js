import './chart.4.0.1.js';
import {data} from './charts-data.js';

const font = {
    family: 'Poppins',
}
const scaleTitle = {
    display: true,
    font
}

const options = {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    onResize: (chart, size) => {
        if (size && size.width < 530) {
            chart.options.plugins.legend.position = 'bottom';
        } else {
            chart.options.plugins.legend.position = 'right';
        }
    },
    plugins: {
        title: {
            display: true,
            padding: {top: 16, bottom: 16},
            text: ['Speed up over Hugging Face baseline', '(Bert base, AMP fp16)'],
            color: '#4351e8',
            font: {...font, weight: 'bold'}
        },
        subtitle: {
            display: true,
            text: 'batch size = 1',
            font
        },
        legend: {
            display: true,
            position: 'right',
            title: {
                display: true,
                padding: {top: 24},
                text: 'Inference engines',
                font: {...font, weight: 'bold'}
            },
            labels: {
                boxWidth: 20
            },
            font
        },
        tooltip: {
            enabled: true,
            position: 'nearest'
        },
    },
    scales: {
        x: {
            title: {
                ...scaleTitle,
                text: ['input shape', '(batch size, sequence length)'],
            }
        },
        y: {
            title: {
                ...scaleTitle,
                text: 'speed up (x time faster)',
            }
        }
    },
    parsing: {
        xAxisKey: 'sequence',
        yAxisKey: 'speed',
    },
}
const optionsBatch1 = {
    ...options,
    plugins: {...options.plugins, subtitle: {...options.plugins.subtitle, text: 'batch size = 1'}}
};
const optionsBatch8 = {
    ...options,
    plugins: {...options.plugins, subtitle: {...options.plugins.subtitle, text: 'batch size = 8'}}
};

const getLabels = batch => [`(${batch}, 16)`, `(${batch}, 128)`, `(${batch}, 256)`, `(${batch}, 384)`, `(${batch}, 512)`]

const filterBySequence = (d) => !(d.sequence === 32 || d.sequence === 64)
const filterByBaseline = (d) => d.engine === 'Baseline'
const sortBySequence = (a, b) => a.sequence - b.sequence

const batch1BaseLine = data.filter(d => filterByBaseline(d) && filterBySequence(d) && d.batch === 1).sort(sortBySequence);
const batch8BaseLine = data.filter(d => filterByBaseline(d) && filterBySequence(d) && d.batch === 8).sort(sortBySequence);
const avgBatch1BaseLine = batch1BaseLine.reduce((acc, {speed}) => acc + speed, 0) / batch1BaseLine.length;
const avgBatch8BaseLine = batch8BaseLine.reduce((acc, {speed}) => acc + speed, 0) / batch1BaseLine.length;

const avgBatch1 = d => {
    return {sequence: d.sequence, speed: (avgBatch1BaseLine / d.speed).toFixed(2)}
}
const avgBatch8 = d => {
    return {sequence: d.sequence, speed: (avgBatch8BaseLine / d.speed).toFixed(2)}
}

const datasets = [
    {
        label: 'ONNX Runtime',
    },
    {
        label: 'AITemplate',
    },
    {
        label: 'TensorRT',
    },
    {
        label: 'Inductor',
    },
    {
        label: 'DeepSpeed',
    },
    {
        label: 'Cuda graphs',
    },
    {
        label: 'Nvfuser',
    },
    {
        label: 'Kernl',
        borderWidth: 2,
    },
    {
        type: 'line',
        label: 'Baseline',
    },
]

const batch1Datasets = datasets.map(v => {
    return {
        ...v,
        label: v.label.startsWith('Kernl') ? v.label + ' ❤' : v.label,
        data: data
            .filter(d => d.engine === v.label && filterBySequence(d) && d.batch === 1)
            .sort(sortBySequence)
            .map(avgBatch1),
    }
})

const batch8Datasets = datasets.map(v => {
    return {
        ...v,
        label: v.label.startsWith('Kernl') ? v.label + ' ❤' : v.label,
        data: data
            .filter(d => d.engine === v.label && filterBySequence(d) && d.batch === 8)
            .sort(sortBySequence)
            .map(avgBatch8),
    }
})

new Chart(
    document.getElementById('performance-chart-1'),
    {
        type: 'bar',
        options: optionsBatch1,
        data: {
            labels: getLabels(1),
            datasets: batch1Datasets,
        }
    }
);

new Chart(
    document.getElementById('performance-chart-8'),
    {
        type: 'bar',
        options: optionsBatch8,
        data: {
            labels: getLabels(8),
            datasets: batch8Datasets,
        }
    }
);