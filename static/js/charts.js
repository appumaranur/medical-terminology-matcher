/**
 * Chart.js Visualizations for Medical Terminology Matcher
 */

let categoryChartInstance = null;
let relationshipChartInstance = null;
let modelComparisonChartInstance = null;

function renderCategoryChart(canvasId, categoriesData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (categoryChartInstance) {
        categoryChartInstance.destroy();
    }

    const labels = Object.keys(categoriesData);
    const dataValues = Object.values(categoriesData);

    categoryChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: dataValues,
                backgroundColor: [
                    '#0d9488', '#0284c7', '#3b82f6', '#8b5cf6',
                    '#ec4899', '#f43f5e', '#f97316', '#eab308',
                    '#10b981', '#06b6d4', '#64748b', '#6366f1'
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 12,
                        font: { size: 11, family: 'Inter' }
                    }
                }
            },
            cutout: '65%'
        }
    });
}

function renderRelationshipChart(canvasId, relationshipData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (relationshipChartInstance) {
        relationshipChartInstance.destroy();
    }

    const labels = Object.keys(relationshipData);
    const dataValues = Object.values(relationshipData);

    relationshipChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Comparisons Count',
                data: dataValues,
                backgroundColor: [
                    '#10b981', // Equivalent
                    '#0284c7', // Related
                    '#f59e0b', // Moderate
                    '#94a3b8'  // Unrelated
                ],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

function renderModelComparisonChart(canvasId, comparisonTable) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (modelComparisonChartInstance) {
        modelComparisonChartInstance.destroy();
    }

    const labels = comparisonTable.map(item => item.method_name);
    const scores = comparisonTable.map(item => Math.round(item.score * 100));

    modelComparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Semantic Similarity Score (%)',
                data: scores,
                backgroundColor: [
                    '#94a3b8', // Levenshtein
                    '#cbd5e1', // Jaccard
                    '#64748b', // TF-IDF
                    '#0d9488', // Sentence-BERT
                    '#0284c7'  // BioBERT
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Similarity: ${context.parsed.x}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: function(value) { return value + '%'; }
                    }
                }
            }
        }
    });
}

