// Modern Professional Geophysical Software Landing Page - JavaScript
document.addEventListener('DOMContentLoaded', function() {
    
    // Navigation functionality
    const navbar = document.querySelector('.navbar');
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Mobile menu toggle
    hamburger.addEventListener('click', function() {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });
    
    // Close mobile menu when clicking on a link
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });
    
    // Navbar scroll effect
    let lastScrollTop = 0;
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScrollTop = scrollTop;
    });
    
    // Smooth scrolling for navigation links
    function scrollToSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            const navbarHeight = navbar.offsetHeight;
            const sectionTop = section.offsetTop - navbarHeight;
            
            window.scrollTo({
                top: sectionTop,
                behavior: 'smooth'
            });
        }
    }
    
    // Make scrollToSection available globally
    window.scrollToSection = scrollToSection;
    
    // Handle navigation link clicks
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            if (href.startsWith('#')) {
                const sectionId = href.substring(1);
                scrollToSection(sectionId);
            }
        });
    });
    
    // Active navigation link highlighting
    const sections = document.querySelectorAll('section[id]');
    
    function updateActiveNavLink() {
        const scrollPos = window.scrollY + navbar.offsetHeight + 50;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPos >= sectionTop && scrollPos <= sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }
    
    window.addEventListener('scroll', updateActiveNavLink);
    
    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.program-card, .feature-card, .contact-form, .contact-info');
    animateElements.forEach(el => {
        observer.observe(el);
    });
    
    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
        .program-card,
        .feature-card,
        .contact-form,
        .contact-info {
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .program-card.animate,
        .feature-card.animate,
        .contact-form.animate,
        .contact-info.animate {
            opacity: 1;
            transform: translateY(0);
        }
        
        .nav-link.active {
            color: var(--primary-color);
        }
        
        .nav-link.active::after {
            width: 100%;
        }
    `;
    document.head.appendChild(style);
    
    // Demo form functionality
    const demoForm = document.getElementById('demoForm');
    
    if (demoForm) {
        demoForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            const data = {
                name: formData.get('name'),
                email: formData.get('email'),
                company: formData.get('company'),
                program: formData.get('program'),
                message: formData.get('message')
            };
            
            // Validate required fields
            if (!data.name || !data.email || !data.program) {
                showNotification('Please fill in all required fields.', 'error');
                return;
            }
            
            // Validate email
            if (!isValidEmail(data.email)) {
                showNotification('Please enter a valid email address.', 'error');
                return;
            }
            
            // Simulate form submission
            submitDemoRequest(data);
        });
    }
    
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    function submitDemoRequest(data) {
        // Show loading state
        const submitButton = demoForm.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scheduling...';
        submitButton.disabled = true;
        
        // Simulate API call
        setTimeout(() => {
            // Reset button
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
            
            // Show success message
            showNotification(
                `Thank you, ${data.name}! Your demo request has been received. We'll contact you within 24 hours to schedule your personalized demonstration.`,
                'success'
            );
            
            // Reset form
            demoForm.reset();
            
            // Log the request (in a real app, this would be sent to a server)
            console.log('Demo request submitted:', data);
            
        }, 2000);
    }
    
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
                <span>${message}</span>
                <button class="notification-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Add notification styles if not already added
        if (!document.querySelector('#notification-styles')) {
            const notificationStyles = document.createElement('style');
            notificationStyles.id = 'notification-styles';
            notificationStyles.textContent = `
                .notification {
                    position: fixed;
                    top: 100px;
                    right: 20px;
                    max-width: 400px;
                    padding: 1rem;
                    border-radius: var(--border-radius);
                    color: var(--light-color);
                    box-shadow: var(--shadow-lg);
                    z-index: 1001;
                    transform: translateX(100%);
                    transition: transform 0.3s ease-out;
                }
                
                .notification.show {
                    transform: translateX(0);
                }
                
                .notification-success {
                    background: var(--success-color);
                }
                
                .notification-error {
                    background: var(--error-color);
                }
                
                .notification-info {
                    background: var(--primary-color);
                }
                
                .notification-content {
                    display: flex;
                    align-items: flex-start;
                    gap: 0.75rem;
                }
                
                .notification-content i {
                    margin-top: 0.125rem;
                }
                
                .notification-close {
                    background: none;
                    border: none;
                    color: var(--light-color);
                    cursor: pointer;
                    padding: 0;
                    margin-left: auto;
                    opacity: 0.7;
                    transition: opacity 0.2s ease;
                }
                
                .notification-close:hover {
                    opacity: 1;
                }
            `;
            document.head.appendChild(notificationStyles);
        }
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Show notification
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Auto remove after 5 seconds
        const autoRemoveTimer = setTimeout(() => {
            removeNotification(notification);
        }, 5000);
        
        // Handle close button
        const closeButton = notification.querySelector('.notification-close');
        closeButton.addEventListener('click', () => {
            clearTimeout(autoRemoveTimer);
            removeNotification(notification);
        });
    }
    
    function removeNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
    
    // Parallax effect for hero section
    const hero = document.querySelector('.hero');
    const floatingCards = document.querySelectorAll('.floating-card');
    
    function updateParallax() {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        floatingCards.forEach((card, index) => {
            const speed = 0.5 + (index * 0.1);
            card.style.transform = `translateY(${rate * speed}px)`;
        });
    }
    
    // Throttle parallax updates for performance
    let ticking = false;
    function requestParallaxUpdate() {
        if (!ticking) {
            requestAnimationFrame(updateParallax);
            ticking = true;
            setTimeout(() => {
                ticking = false;
            }, 16); // ~60fps
        }
    }
    
    window.addEventListener('scroll', requestParallaxUpdate);
    
    // Progressive enhancement: Add advanced features if supported
    if ('IntersectionObserver' in window) {
        // Advanced scroll animations
        const advancedObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.classList.add('animate-advanced');
                    }, index * 100);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });
        
        document.querySelectorAll('.feature-item').forEach(item => {
            advancedObserver.observe(item);
        });
        
        // Add advanced animation styles
        const advancedStyle = document.createElement('style');
        advancedStyle.textContent = `
            .feature-item {
                opacity: 0;
                transform: translateX(-20px);
                transition: all 0.4s ease-out;
            }
            
            .feature-item.animate-advanced {
                opacity: 1;
                transform: translateX(0);
            }
        `;
        document.head.appendChild(advancedStyle);
    }
    
    // Performance monitoring
    function logPerformanceMetrics() {
        if ('performance' in window && 'getEntriesByType' in performance) {
            const navigation = performance.getEntriesByType('navigation')[0];
            console.log('Page Load Performance:', {
                domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                totalTime: navigation.loadEventEnd - navigation.navigationStart
            });
        }
    }
    
    // Log performance after page load
    window.addEventListener('load', () => {
        setTimeout(logPerformanceMetrics, 0);
    });
    
    // Keyboard navigation support
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            // Close mobile menu
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
            
            // Remove any active notifications
            const notifications = document.querySelectorAll('.notification');
            notifications.forEach(removeNotification);
        }
    });
    
    // Print-friendly styles
    const printStyles = document.createElement('style');
    printStyles.media = 'print';
    printStyles.textContent = `
        @media print {
            .navbar,
            .hamburger,
            .hero-visual,
            .floating-card,
            .btn,
            .contact-form,
            .footer {
                display: none !important;
            }
            
            body {
                font-size: 12pt;
                line-height: 1.4;
                color: #000;
                background: #fff;
            }
            
            .hero {
                background: #fff;
                color: #000;
                min-height: auto;
                padding: 2rem 0;
            }
            
            .section-header h2 {
                color: #000;
                font-size: 18pt;
            }
            
            .program-card {
                border: 1px solid #ccc;
                margin-bottom: 1rem;
                break-inside: avoid;
            }
        }
    `;
    document.head.appendChild(printStyles);
    
    console.log('🚀 Geophysical Software Landing Page initialized successfully!');
    console.log('Features loaded: Navigation, Animations, Forms, Performance Monitoring');
});