import './chart.4.0.1.js';

const data = [
    {
        'engine': 'Inductor',
        'batch': 1,
        'sequence': 16,
        'speed': 0.0018
    },
    {
        'engine': 'Inductor',
        'batch': 1,
        'sequence': 32,
        'speed': 0.002
    },
    {
        'engine': 'Inductor',
        'batch': 1,
        'sequence': 64,
        'speed': 0.002
    },
    {
        'engine': 'Inductor',
        'batch': 1,
        'sequence': 128,
        'speed': 0.0025
    },
    {
        'engine': 'Inductor',
        'batch': 1,
        'sequence': 256,
        'speed': 0.003
    },
    {
        'engine': 'Inductor',
        'batch': 1,
        'sequence': 384,
        'speed': 0.0036
    },
    {
        'engine': 'Inductor',
        'batch': 1,
        'sequence': 512,
        'speed': 0.0048
    },
    {
        'engine': 'Inductor',
        'batch': 8,
        'sequence': 16,
        'speed': 0.0023
    },
    {
        'engine': 'Inductor',
        'batch': 8,
        'sequence': 32,
        'speed': 0.0027
    },
    {
        'engine': 'Inductor',
        'batch': 8,
        'sequence': 64,
        'speed': 0.0039
    },
    {
        'engine': 'Inductor',
        'batch': 8,
        'sequence': 128,
        'speed': 0.0065
    },
    {
        'engine': 'Inductor',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0117
    },
    {
        'engine': 'Inductor',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0156
    },
    {
        'engine': 'Inductor',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0212
    },
    {
        'engine': 'Inductor',
        'batch': 32,
        'sequence': 16,
        'speed': 0.0039
    },
    {
        'engine': 'Inductor',
        'batch': 32,
        'sequence': 32,
        'speed': 0.0062
    },
    {
        'engine': 'Inductor',
        'batch': 32,
        'sequence': 64,
        'speed': 0.0108
    },
    {
        'engine': 'Inductor',
        'batch': 32,
        'sequence': 128,
        'speed': 0.0177
    },
    {
        'engine': 'Inductor',
        'batch': 32,
        'sequence': 256,
        'speed': 0.0357
    },
    {
        'engine': 'AITemplate',
        'batch': 1,
        'sequence': 16,
        'speed': 0.0012
    },
    {
        'engine': 'AITemplate',
        'batch': 1,
        'sequence': 32,
        'speed': 0.0012
    },
    {
        'engine': 'AITemplate',
        'batch': 1,
        'sequence': 64,
        'speed': 0.0013
    },
    {
        'engine': 'AITemplate',
        'batch': 1,
        'sequence': 128,
        'speed': 0.0015
    },
    {
        'engine': 'AITemplate',
        'batch': 1,
        'sequence': 256,
        'speed': 0.002
    },
    {
        'engine': 'AITemplate',
        'batch': 1,
        'sequence': 384,
        'speed': 0.0022
    },
    {
        'engine': 'AITemplate',
        'batch': 1,
        'sequence': 512,
        'speed': 0.0031
    },
    {
        'engine': 'AITemplate',
        'batch': 8,
        'sequence': 16,
        'speed': 0.0013
    },
    {
        'engine': 'AITemplate',
        'batch': 8,
        'sequence': 32,
        'speed': 0.0017
    },
    {
        'engine': 'AITemplate',
        'batch': 8,
        'sequence': 64,
        'speed': 0.0026
    },
    {
        'engine': 'AITemplate',
        'batch': 8,
        'sequence': 128,
        'speed': 0.0043
    },
    {
        'engine': 'AITemplate',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0076
    },
    {
        'engine': 'AITemplate',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0115
    },
    {
        'engine': 'AITemplate',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0149
    },
    {
        'engine': 'AITemplate',
        'batch': 32,
        'sequence': 16,
        'speed': 0.0026
    },
    {
        'engine': 'AITemplate',
        'batch': 32,
        'sequence': 32,
        'speed': 0.0043
    },
    {
        'engine': 'AITemplate',
        'batch': 32,
        'sequence': 64,
        'speed': 0.0073
    },
    {
        'engine': 'AITemplate',
        'batch': 32,
        'sequence': 128,
        'speed': 0.0127
    },
    {
        'engine': 'AITemplate',
        'batch': 32,
        'sequence': 256,
        'speed': 0.0242
    },
    {
        'engine': 'TensorRT',
        'batch': 1,
        'sequence': 16,
        'speed': 0.001
    },
    {
        'engine': 'TensorRT',
        'batch': 1,
        'sequence': 32,
        'speed': 0.001
    },
    {
        'engine': 'TensorRT',
        'batch': 1,
        'sequence': 64,
        'speed': 0.0011
    },
    {
        'engine': 'TensorRT',
        'batch': 1,
        'sequence': 128,
        'speed': 0.0013
    },
    {
        'engine': 'TensorRT',
        'batch': 1,
        'sequence': 256,
        'speed': 0.0016
    },
    {
        'engine': 'TensorRT',
        'batch': 1,
        'sequence': 384,
        'speed': 0.0026
    },
    {
        'engine': 'TensorRT',
        'batch': 1,
        'sequence': 512,
        'speed': 0.0026
    },
    {
        'engine': 'TensorRT',
        'batch': 8,
        'sequence': 16,
        'speed': 0.0011
    },
    {
        'engine': 'TensorRT',
        'batch': 8,
        'sequence': 32,
        'speed': 0.0015
    },
    {
        'engine': 'TensorRT',
        'batch': 8,
        'sequence': 64,
        'speed': 0.0019
    },
    {
        'engine': 'TensorRT',
        'batch': 8,
        'sequence': 128,
        'speed': 0.0036
    },
    {
        'engine': 'TensorRT',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0064
    },
    {
        'engine': 'TensorRT',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0139
    },
    {
        'engine': 'TensorRT',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0139
    },
    {
        'engine': 'TensorRT',
        'batch': 32,
        'sequence': 16,
        'speed': 0.002
    },
    {
        'engine': 'TensorRT',
        'batch': 32,
        'sequence': 32,
        'speed': 0.0031
    },
    {
        'engine': 'TensorRT',
        'batch': 32,
        'sequence': 64,
        'speed': 0.0054
    },
    {
        'engine': 'TensorRT',
        'batch': 32,
        'sequence': 128,
        'speed': 0.0103
    },
    {
        'engine': 'TensorRT',
        'batch': 32,
        'sequence': 256,
        'speed': 0.021
    },
    {
        'engine': 'DeepSpeed',
        'batch': 1,
        'sequence': 16,
        'speed': 0.0009
    },
    {
        'engine': 'DeepSpeed',
        'batch': 1,
        'sequence': 32,
        'speed': 0.0008
    },
    {
        'engine': 'DeepSpeed',
        'batch': 1,
        'sequence': 64,
        'speed': 0.0009
    },
    {
        'engine': 'DeepSpeed',
        'batch': 1,
        'sequence': 128,
        'speed': 0.0012
    },
    {
        'engine': 'DeepSpeed',
        'batch': 1,
        'sequence': 256,
        'speed': 0.0019
    },
    {
        'engine': 'DeepSpeed',
        'batch': 1,
        'sequence': 384,
        'speed': 0.0023
    },
    {
        'engine': 'DeepSpeed',
        'batch': 1,
        'sequence': 512,
        'speed': 0.0032
    },
    {
        'engine': 'DeepSpeed',
        'batch': 8,
        'sequence': 16,
        'speed': 0.0011
    },
    {
        'engine': 'DeepSpeed',
        'batch': 8,
        'sequence': 32,
        'speed': 0.0016
    },
    {
        'engine': 'DeepSpeed',
        'batch': 8,
        'sequence': 64,
        'speed': 0.0025
    },
    {
        'engine': 'DeepSpeed',
        'batch': 8,
        'sequence': 128,
        'speed': 0.0051
    },
    {
        'engine': 'DeepSpeed',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0106
    },
    {
        'engine': 'DeepSpeed',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0161
    },
    {
        'engine': 'DeepSpeed',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0219
    },
    {
        'engine': 'DeepSpeed',
        'batch': 32,
        'sequence': 16,
        'speed': 0.0025
    },
    {
        'engine': 'DeepSpeed',
        'batch': 32,
        'sequence': 32,
        'speed': 0.005
    },
    {
        'engine': 'DeepSpeed',
        'batch': 32,
        'sequence': 64,
        'speed': 0.0097
    },
    {
        'engine': 'DeepSpeed',
        'batch': 32,
        'sequence': 128,
        'speed': 0.0176
    },
    {
        'engine': 'DeepSpeed',
        'batch': 32,
        'sequence': 256,
        'speed': 0.0374
    },
    {
        'engine': 'Baseline',
        'batch': 1,
        'sequence': 16,
        'speed': 0.00795173406600952
    },
    {
        'engine': 'Cuda graphs',
        'batch': 1,
        'sequence': 16,
        'speed': 0.00106519103050232
    },
    {
        'engine': 'Nvfuser',
        'batch': 1,
        'sequence': 16,
        'speed': 0.00355699706077576
    },
    {
        'engine': 'Kernl',
        'batch': 1,
        'sequence': 16,
        'speed': 0.000644994974136352
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 1,
        'sequence': 16,
        'speed': 0.00239814496040344
    },
    {
        'engine': 'Baseline',
        'batch': 1,
        'sequence': 128,
        'speed': 0.00808307266235352
    },
    {
        'engine': 'Cuda graphs',
        'batch': 1,
        'sequence': 128,
        'speed': 0.00161654901504517
    },
    {
        'engine': 'Nvfuser',
        'batch': 1,
        'sequence': 128,
        'speed': 0.00363480401039124
    },
    {
        'engine': 'Kernl',
        'batch': 1,
        'sequence': 128,
        'speed': 0.00145257699489594
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 1,
        'sequence': 128,
        'speed': 0.00245641589164734
    },
    {
        'engine': 'Baseline',
        'batch': 1,
        'sequence': 256,
        'speed': 0.00817633533477783
    },
    {
        'engine': 'Cuda graphs',
        'batch': 1,
        'sequence': 256,
        'speed': 0.00291786408424377
    },
    {
        'engine': 'Nvfuser',
        'batch': 1,
        'sequence': 256,
        'speed': 0.00357921099662781
    },
    {
        'engine': 'Kernl',
        'batch': 1,
        'sequence': 256,
        'speed': 0.00198752295970917
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 1,
        'sequence': 256,
        'speed': 0.00270905590057373
    },
    {
        'engine': 'Baseline',
        'batch': 1,
        'sequence': 384,
        'speed': 0.00817633533477783
    },
    {
        'engine': 'Cuda graphs',
        'batch': 1,
        'sequence': 384,
        'speed': 0.00291786408424377
    },
    {
        'engine': 'Nvfuser',
        'batch': 1,
        'sequence': 384,
        'speed': 0.00357921099662781
    },
    {
        'engine': 'Kernl',
        'batch': 1,
        'sequence': 384,
        'speed': 0.00198752295970917
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 1,
        'sequence': 384,
        'speed': 0.00270905590057373
    },
    {
        'engine': 'Baseline',
        'batch': 1,
        'sequence': 512,
        'speed': 0.00850767040252686
    },
    {
        'engine': 'Cuda graphs',
        'batch': 1,
        'sequence': 512,
        'speed': 0.00442398595809937
    },
    {
        'engine': 'Nvfuser',
        'batch': 1,
        'sequence': 512,
        'speed': 0.00375320100784302
    },
    {
        'engine': 'Kernl',
        'batch': 1,
        'sequence': 512,
        'speed': 0.00254801301956177
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 1,
        'sequence': 512,
        'speed': 0.003773845911026
    },
    {
        'engine': 'Baseline',
        'batch': 8,
        'sequence': 16,
        'speed': 0.00851109409332275
    },
    {
        'engine': 'Cuda graphs',
        'batch': 8,
        'sequence': 16,
        'speed': 0.00164371204376221
    },
    {
        'engine': 'Nvfuser',
        'batch': 8,
        'sequence': 16,
        'speed': 0.00403249311447144
    },
    {
        'engine': 'Kernl',
        'batch': 8,
        'sequence': 16,
        'speed': 0.00136490595340729
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 8,
        'sequence': 16,
        'speed': 0.0024571259021759
    },
    {
        'engine': 'Baseline',
        'batch': 8,
        'sequence': 128,
        'speed': 0.00816122531890869
    },
    {
        'engine': 'Cuda graphs',
        'batch': 8,
        'sequence': 128,
        'speed': 0.00626455688476562
    },
    {
        'engine': 'Nvfuser',
        'batch': 8,
        'sequence': 128,
        'speed': 0.00544208192825317
    },
    {
        'engine': 'Kernl',
        'batch': 8,
        'sequence': 128,
        'speed': 0.00442465305328369
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 8,
        'sequence': 128,
        'speed': 0.00574171686172485
    },
    {
        'engine': 'Baseline',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0135426902770996
    },
    {
        'engine': 'Cuda graphs',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0131534404754639
    },
    {
        'engine': 'Nvfuser',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0103799543380737
    },
    {
        'engine': 'Kernl',
        'batch': 8,
        'sequence': 256,
        'speed': 0.00705874919891357
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 8,
        'sequence': 256,
        'speed': 0.0111610059738159
    },
    {
        'engine': 'Baseline',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0192422294616699
    },
    {
        'engine': 'Cuda graphs',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0189292831420898
    },
    {
        'engine': 'Nvfuser',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0148918361663818
    },
    {
        'engine': 'Kernl',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0105360803604126
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 8,
        'sequence': 384,
        'speed': 0.0155409851074219
    },
    {
        'engine': 'Baseline',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0277956104278564
    },
    {
        'engine': 'Cuda graphs',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0273300457000732
    },
    {
        'engine': 'Nvfuser',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0203310241699219
    },
    {
        'engine': 'Kernl',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0137086231231689
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 8,
        'sequence': 512,
        'speed': 0.0207632541656494
    },
    {
        'engine': 'Baseline',
        'batch': 32,
        'sequence': 16,
        'speed': 0.00819225311279297
    },
    {
        'engine': 'Cuda graphs',
        'batch': 32,
        'sequence': 16,
        'speed': 0.00312871694564819
    },
    {
        'engine': 'Nvfuser',
        'batch': 32,
        'sequence': 16,
        'speed': 0.00410407495498657
    },
    {
        'engine': 'Kernl',
        'batch': 32,
        'sequence': 16,
        'speed': 0.00239948511123657
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 32,
        'sequence': 16,
        'speed': 0.00305436611175537
    },
    {
        'engine': 'Baseline',
        'batch': 32,
        'sequence': 128,
        'speed': 0.019238655090332
    },
    {
        'engine': 'Cuda graphs',
        'batch': 32,
        'sequence': 128,
        'speed': 0.0191719589233398
    },
    {
        'engine': 'Nvfuser',
        'batch': 32,
        'sequence': 128,
        'speed': 0.0172065410614014
    },
    {
        'engine': 'Kernl',
        'batch': 32,
        'sequence': 128,
        'speed': 0.0130825481414795
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 32,
        'sequence': 128,
        'speed': 0.017692741394043
    },
    {
        'engine': 'Baseline',
        'batch': 32,
        'sequence': 256,
        'speed': 0.0431598892211914
    },
    {
        'engine': 'Cuda graphs',
        'batch': 32,
        'sequence': 256,
        'speed': 0.042514949798584
    },
    {
        'engine': 'Nvfuser',
        'batch': 32,
        'sequence': 256,
        'speed': 0.0349777145385742
    },
    {
        'engine': 'Kernl',
        'batch': 32,
        'sequence': 256,
        'speed': 0.025843318939209
    },
    {
        'engine': 'ONNX Runtime',
        'batch': 32,
        'sequence': 256,
        'speed': 0.0348241157531738
    }
];

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