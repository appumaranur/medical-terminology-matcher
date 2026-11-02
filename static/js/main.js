/**
 * Medical Terminology Matcher - Core Frontend Utilities
 */

// Toast Notifications
function showNotification(message, type = 'info') {
    const container = document.getElementById('notification-container') || createNotificationContainer();
    const alertDiv = document.createElement('div');
    const bgClass = type === 'success' ? 'alert-success' : type === 'error' ? 'alert-danger' : 'alert-info';
    
    alertDiv.className = `alert ${bgClass} alert-dismissible fade show shadow-sm mb-2`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="bi ${type === 'success' ? 'bi-check-circle-fill' : type === 'error' ? 'bi-exclamation-triangle-fill' : 'bi-info-circle-fill'} me-2"></i>
            <div>${message}</div>
        </div>
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    container.appendChild(alertDiv);
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 250);
    }, 4000);
}

function createNotificationContainer() {
    const c = document.createElement('div');
    c.id = 'notification-container';
    c.style.position = 'fixed';
    c.style.top = '20px';
    c.style.right = '20px';
    c.style.zIndex = '9999';
    c.style.maxWidth = '360px';
    document.body.appendChild(c);
    return c;
}

// Circular Score Gauge Animation
function updateScoreGauge(score, elementId = 'circular-gauge') {
    const gauge = document.getElementById(elementId);
    if (!gauge) return;

    const progressCircle = gauge.querySelector('.gauge-progress');
    const textElem = gauge.querySelector('.gauge-percentage');
    
    const percentage = Math.max(0, Math.min(100, Math.round(score * 100)));
    const radius = 70;
    const circumference = 2 * Math.PI * radius; // ~439.8
    
    const offset = circumference - (percentage / 100) * circumference;
    
    if (progressCircle) {
        progressCircle.style.strokeDasharray = `${circumference}`;
        progressCircle.style.strokeDashoffset = `${offset}`;
        
        // Color coding according to score
        if (score >= 0.80) {
            progressCircle.style.stroke = '#10b981'; // Green
        } else if (score >= 0.60) {
            progressCircle.style.stroke = '#0284c7'; // Blue
        } else if (score >= 0.40) {
            progressCircle.style.stroke = '#f59e0b'; // Amber
        } else {
            progressCircle.style.stroke = '#ef4444'; // Red
        }
    }
    
    if (textElem) {
        textElem.textContent = `${percentage}%`;
    }
}

// Format Float
function formatScore(score) {
    if (score === null || score === undefined) return 'N/A';
    return (score * 100).toFixed(1) + '%';
}

// Global Document Ready
document.addEventListener('DOMContentLoaded', () => {
    // Mobile sidebar toggle
    const toggleBtn = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.app-sidebar');
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('show');
        });
    }
});

