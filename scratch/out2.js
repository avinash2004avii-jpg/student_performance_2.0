
Chart.defaults.color = '#cbd5e1';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';

// 1. Score Distribution
const scoreCtx = document.getElementById('scoreHist').getContext('2d');
const gradientGreen = scoreCtx.createLinearGradient(0, 0, 0, 400);
gradientGreen.addColorStop(0, '#22c55e'); gradientGreen.addColorStop(1, '#15803d');
const gradientYellow = scoreCtx.createLinearGradient(0, 0, 0, 400);
gradientYellow.addColorStop(0, '#eab308'); gradientYellow.addColorStop(1, '#a16207');
const gradientOrange = scoreCtx.createLinearGradient(0, 0, 0, 400);
gradientOrange.addColorStop(0, '#f97316'); gradientOrange.addColorStop(1, '#c2410c');
const gradientRed = scoreCtx.createLinearGradient(0, 0, 0, 400);
gradientRed.addColorStop(0, '#ef4444'); gradientRed.addColorStop(1, '#b91c1c');

new Chart(scoreCtx, {
    type: 'bar',
    data: {
        labels: ["0-40", "41-60", "61-80", "81-100"],
        datasets: [{
            label: 'Number of Students',
            data: [0, 7, 598, 296],
            backgroundColor: [gradientRed, gradientOrange, gradientYellow, gradientGreen],
            borderRadius: 8,
            borderWidth: 0
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: 'rgba(0,0,0,0.8)', titleFont: { size: 14 }, bodyFont: { size: 14 }, padding: 10 }
        },
        scales: {
            x: { grid: { display: false } },
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
    }
});

// 2. Attendance vs Performance (Grouped Bar Chart)
new Chart(document.getElementById('attScatter'), {
    type: 'bar',
    data: {
        labels: ["Low (<70%)", "Medium (70-85%)", "High (>85%)"],
        datasets: [
            {
                label: 'Average Attendance %',
                data: [67.0, 77.3, 93.0],
                backgroundColor: 'rgba(56, 189, 248, 0.8)',
                borderRadius: 6
            },
            {
                label: 'Average Score',
                data: [76.7, 75.5, 78.2],
                backgroundColor: 'rgba(168, 85, 247, 0.8)',
                borderRadius: 6
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: true, position: 'top', labels: { color: '#cbd5e1' } },
            tooltip: { mode: 'index', intersect: false }
        },
        scales: {
            x: { grid: { display: false } },
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
    }
});

// 3. Component Comparison (Grouped Bar Chart instead of Radar)
new Chart(document.getElementById('compChart'), {
    type: 'bar',
    data: {
        labels: ["Assignments", "Exams", "Attendance", "Internal Marks"],
        datasets: [
            {
                label: 'Safe Students',
                data: [78.7, 71.8, 83.9, 72.9],
                backgroundColor: 'rgba(34, 197, 94, 0.8)',
                borderRadius: 6
            },
            {
                label: 'At-Risk Students',
                data: [56.0, 54.0, 82.1, 47.1],
                backgroundColor: 'rgba(239, 68, 68, 0.8)',
                borderRadius: 6
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: true, position: 'top', labels: { color: '#cbd5e1' } },
            tooltip: { mode: 'index', intersect: false }
        },
        scales: {
            x: { grid: { display: false } },
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Average Score / %' } }
        }
    }
});

// 4. Class-wise Performance
const classScores = [77.1, 76.6, 76.8];
const maxScore = Math.max(...classScores);
const classColors = classScores.map(score => score === maxScore ? 'rgba(236, 72, 153, 0.9)' : 'rgba(99, 102, 241, 0.6)');

new Chart(document.getElementById('classChart'), {
    type: 'bar',
    data: {
        labels: ["Class 10th", "Class 8th", "Class 9th"],
        datasets: [{
            label: 'Average Score',
            data: classScores,
            backgroundColor: classColors,
            borderRadius: 8,
            borderWidth: 0
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: 'rgba(0,0,0,0.8)' }
        },
        scales: {
            x: { grid: { display: false } },
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
    }
});
