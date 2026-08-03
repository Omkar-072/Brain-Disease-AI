/**
 * Brain Disease AI - Main JavaScript
 * Common functionality across the application
 */

// ============== Configuration ==============
const API_BASE_URL = '/api/v1';

// ============== Disease Enum Validation ==============

/** Valid DiseaseType enum values — must stay in sync with backend DiseaseType enum */
const VALID_DISEASE_TYPES = ['GLIOMA', 'MENINGIOMA', 'PITUITARY', 'NO_TUMOR', 'VERY_MILD_DEMENTED', 'MILD_DEMENTED', 'MODERATE_DEMENTED', 'NON_DEMENTED'];

/**
 * Map any raw AI prediction string into a valid backend DiseaseType enum key.
 * Mirrors the backend map_prediction() function in scan_routes.py.
 * Falls back to 'NO_TUMOR' for anything unrecognised.
 * @param {string} raw - The raw predicted_disease string from the API
 * @returns {string} A value from VALID_DISEASE_TYPES
 */
function normalizeDiseaseType(raw) {
    if (!raw) return 'INCONCLUSIVE';

    // Normalise: lower, strip, spaces → underscores
    const normalised = raw.toLowerCase().trim().replace(/[\s/]+/g, '_');

    const mapping = {
        // MRI tumor model outputs
        'glioma': 'GLIOMA',
        'meningioma': 'MENINGIOMA',
        'no_tumor': 'NO_TUMOR',
        'no_tumor_detected': 'NO_TUMOR',
        'pituitary': 'PITUITARY',
        'pituitary_tumor': 'PITUITARY',
        // Alzheimer model outputs
        'very_mild_demented': 'ALZHEIMER',
        'mild_demented': 'ALZHEIMER',
        'moderate_demented': 'ALZHEIMER',
        'non_demented': 'ALZHEIMER',
        // Catch-all healthy phrases
        'healthy_no_significant_findings': 'HEALTHY',
        'healthy': 'HEALTHY',
        'inconclusive': 'INCONCLUSIVE',
        'low_confidence': 'HEALTHY',
    };

    // Also accept already-uppercase enum values directly
    const upperRaw = raw.trim().toUpperCase();
    if (VALID_DISEASE_TYPES.includes(upperRaw)) return upperRaw;

    return mapping.get(label, label.upper())
}

// ============== Authentication Utilities ==============

/**
 * Get the stored authentication token
 */
function getAuthToken() {
    return localStorage.getItem('access_token');
}

/**
 * Check if user is authenticated
 */
function isAuthenticated() {
    const token = getAuthToken();
    if (!token) return false;

    // Check if token is expired (basic check)
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        return payload.exp * 1000 > Date.now();
    } catch (e) {
        return false;
    }
}

/**
 * Set authentication token
 */
function setAuthToken(token) {
    localStorage.setItem('access_token', token);
}

/**
 * Clear authentication token
 */
function clearAuthToken() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
}

/**
 * Logout user
 */
function logout() {
    clearAuthToken();
    window.location.href = '/login';
}

/**
 * Redirect to login if not authenticated
 */
function requireAuth() {
    if (!isAuthenticated()) {
        window.location.href = '/login?redirect=' + encodeURIComponent(window.location.pathname);
        return false;
    }
    return true;
}

// ============== API Utilities ==============

/**
 * Make authenticated API request
 */
async function apiRequest(endpoint, options = {}) {
    const token = getAuthToken();

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` })
        }
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };

    const response = await fetch(`${API_BASE_URL}${endpoint}`, mergedOptions);

    // Handle 401 Unauthorized
    if (response.status === 401) {
        clearAuthToken();
        window.location.href = '/login?session_expired=true';
        return null;
    }

    return response;
}

/**
 * GET request
 */
async function apiGet(endpoint) {
    return apiRequest(endpoint, { method: 'GET' });
}

/**
 * POST request
 */
async function apiPost(endpoint, data) {
    return apiRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

/**
 * PUT request
 */
async function apiPut(endpoint, data) {
    return apiRequest(endpoint, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
}

/**
 * DELETE request
 */
async function apiDelete(endpoint) {
    return apiRequest(endpoint, { method: 'DELETE' });
}

// ============== UI Utilities ==============

/**
 * Show loading spinner
 */
function showLoading(element) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    if (element) {
        element.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Remove existing toasts
    const existingToast = document.querySelector('.toast-container');
    if (existingToast) {
        existingToast.remove();
    }

    const toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
    toastContainer.style.zIndex = '9999';

    const bgClass = {
        'success': 'bg-success',
        'error': 'bg-danger',
        'warning': 'bg-warning',
        'info': 'bg-info'
    }[type] || 'bg-info';

    toastContainer.innerHTML = `
        <div class="toast show" role="alert">
            <div class="toast-header ${bgClass} text-white">
                <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">${message}</div>
        </div>
    `;

    document.body.appendChild(toastContainer);

    // Auto remove after 5 seconds
    setTimeout(() => {
        toastContainer.remove();
    }, 5000);
}

/**
 * Format date for display
 */
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('en-US', options);
}

/**
 * Format date with time
 */
function formatDateTime(dateString) {
    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString('en-US', options);
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Truncate text
 */
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============== Form Utilities ==============

/**
 * Serialize form data to object
 */
function serializeForm(form) {
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    return data;
}

/**
 * Validate email format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Validate password strength
 */
function isValidPassword(password) {
    // At least 8 characters, one uppercase, one lowercase, one number
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(password);
}

// ============== Scroll Utilities ==============

/**
 * Smooth scroll to element
 */
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

/**
 * Scroll to top button functionality
 */
function initScrollToTop() {
    const scrollBtn = document.createElement('button');
    scrollBtn.className = 'scroll-top';
    scrollBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    scrollBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.body.appendChild(scrollBtn);

    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            scrollBtn.classList.add('show');
        } else {
            scrollBtn.classList.remove('show');
        }
    });
}

// ============== Navbar Utilities ==============

/**
 * Update navbar based on auth status
 */
function updateNavbar() {
    // Rely completely on base.html's inline script to render Auth Links properly.
}

// ============== Disease Information ==============

/** Disease display information keyed by VALID_DISEASE_TYPES enum values */
const DISEASE_INFO = {
    'GLIOMA': {
        name: 'Glioma',
        icon: 'fa-circle',
        color: '#fd7e14',
        description: 'A type of tumor that occurs in the brain and spinal cord, arising from glial cells.'
    },
    'MENINGIOMA': {
        name: 'Meningioma',
        icon: 'fa-brain',
        color: '#6f42c1',
        description: 'A tumor that arises from the meninges — the membranes surrounding the brain and spinal cord.'
    },
    'PITUITARY': {
        name: 'Pituitary Tumor',
        icon: 'fa-dot-circle',
        color: '#17a2b8',
        description: 'An abnormal growth in the pituitary gland that can affect hormone production.'
    },
    'NO_TUMOR': {
        name: 'No Tumor Detected',
        icon: 'fa-check-circle',
        color: '#28a745',
        description: 'No abnormal tumor growth was detected in the brain scan.'
    }
};

/**
 * Get disease display info by raw or normalised disease type string.
 * Always normalises through normalizeDiseaseType() first.
 * @param {string} type - Raw or enum disease type string
 * @returns {object} Display info object
 */
function getDiseaseInfo(type) {
    const key = normalizeDiseaseType(type);
    return DISEASE_INFO[key] || {
        name: type,
        icon: 'fa-question-circle',
        color: '#6c757d',
        description: 'Unknown disease type'
    };
}

// ============== Initialize ==============

document.addEventListener('DOMContentLoaded', function () {
    // Update navbar
    updateNavbar();

    // Initialize scroll to top
    initScrollToTop();

    // Handle logout links
    document.querySelectorAll('[data-action="logout"]').forEach(el => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            logout();
        });
    });

    // Check for session expiry message
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('session_expired') === 'true') {
        showToast('Your session has expired. Please log in again.', 'warning');
    }
});

// Export for use in other scripts
window.BrainAI = {
    apiRequest,
    apiGet,
    apiPost,
    apiPut,
    apiDelete,
    isAuthenticated,
    requireAuth,
    logout,
    showToast,
    showLoading,
    formatDate,
    formatDateTime,
    formatFileSize,
    getDiseaseInfo,
    normalizeDiseaseType,
    VALID_DISEASE_TYPES,
    debounce
};
