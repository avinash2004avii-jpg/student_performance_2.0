
Chart.defaults.color = '#cbd5e1';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';

// Score Distribution
new Chart(document.getElementById('scoreHist'), {
    type: 'bar',
    data: {
        labels: ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"],
        datasets: [{
            label: 'Student Count',
            data: [0, 0, 0, 0, 0, 7, 164, 429, 253, 48],
            backgroundColor: 'rgba(124, 58, 237, 0.7)',
            borderColor: '#7c3aed',
            borderWidth: 2,
            borderRadius: 8
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } } }
    }
});

// Attendance vs Performance
new Chart(document.getElementById('attScatter'), {
    type: 'bar',
    data: {
        labels: ["<60%", "60-70%", "70-80%", "80-90%", "90-100%"],
        datasets: [{
            label: 'Avg Score by Attendance',
            data: [0, 76.7, 75.4, 76.9, 78.1],
            backgroundColor: 'rgba(16, 185, 129, 0.7)',
            borderColor: '#10b981',
            borderWidth: 2,
            borderRadius: 8
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
            x: { title: { display: true, text: 'Attendance Range' } },
            y: { title: { display: true, text: 'Average Final Score' }, beginAtZero: true }
        }
    }
});

// Component Comparison
new Chart(document.getElementById('compChart'), {
    type: 'radar',
    data: {
        labels: ["Internal 1", "Internal 2", "Assignment", "Previous"],
        datasets: [
            {
                label: 'Safe Students',
                data: [72.87403100775194, 72.90116279069767, 78.67441860465117, 71.81201550387597],
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderColor: '#10b981',
                borderWidth: 2
            },
            {
                label: 'At-Risk Students',
                data: [48.714285714285715, 45.42857142857143, 56.0, 54.0],
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                borderColor: '#ef4444',
                borderWidth: 2
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            r: {
                angleLines: { color: 'rgba(255,255,255,0.1)' },
                grid: { color: 'rgba(255,255,255,0.1)' },
                pointLabels: { font: { size: 12 } },
                suggestedMin: 0
            }
        }
    }
});
