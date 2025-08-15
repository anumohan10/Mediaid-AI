// Enhanced JavaScript functionality for MediAid AI website

class MediAidWebsite {
    constructor() {
        this.init();
    }

    init() {
        this.setupScrollAnimations();
        this.setupInteractiveElements();
        this.setupPerformanceCounters();
        this.setupParticleSystem();
        this.setupAccessibility();
        this.displayWelcomeMessage();
    }

    // Enhanced scroll animations with stagger effect
    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.classList.add('visible');
                    }, index * 100); // Stagger animation
                }
            });
        }, observerOptions);

        document.querySelectorAll('.scroll-animate').forEach(el => {
            observer.observe(el);
        });
    }

    // Interactive elements and dynamic content
    setupInteractiveElements() {
        // Enhanced typing effect for subtitle
        this.typeWriter('.subtitle', 'Advanced Generative AI Medical Assistant Platform', 50);

        // Interactive demo button with status check
        this.setupDemoButton();

        // Dynamic feature card interactions
        this.setupFeatureCards();

        // Tech stack hover effects
        this.setupTechStackHovers();

        // Smooth scrolling with offset for fixed nav
        this.setupSmoothScrolling();
    }

    // Animated performance counters
    setupPerformanceCounters() {
        const counters = [
            { element: '.metric-value', target: [94.2, 480, 87.5, 75.3, 92.1, 99.8], suffix: ['%', 'ms', '%', '%', '%', '%'] }
        ];

        const animateCounter = (element, target, suffix = '', duration = 2000) => {
            const obj = document.querySelector(element);
            if (!obj) return;

            let start = 0;
            const increment = target / (duration / 16);
            
            const timer = setInterval(() => {
                start += increment;
                if (start >= target) {
                    start = target;
                    clearInterval(timer);
                }
                obj.textContent = start.toFixed(1) + suffix;
            }, 16);
        };

        // Trigger counter animations when metrics section is visible
        const metricsObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const metricElements = document.querySelectorAll('.metric-value');
                    const targets = [94.2, 480, 87.5, 75.3, 92.1, 99.8];
                    const suffixes = ['%', 'ms', '%', '%', '%', '%'];
                    
                    metricElements.forEach((element, index) => {
                        setTimeout(() => {
                            this.animateValue(element, 0, targets[index], suffixes[index], 2000);
                        }, index * 200);
                    });
                    
                    metricsObserver.unobserve(entry.target);
                }
            });
        });

        const metricsSection = document.querySelector('#performance');
        if (metricsSection) {
            metricsObserver.observe(metricsSection);
        }
    }

    // Particle background system
    setupParticleSystem() {
        const header = document.querySelector('header');
        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'particles';
        header.appendChild(particlesContainer);

        for (let i = 0; i < 50; i++) {
            this.createParticle(particlesContainer);
        }
    }

    createParticle(container) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 5 + 2;
        const left = Math.random() * 100;
        const animationDuration = Math.random() * 3 + 3;
        const delay = Math.random() * 2;

        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${left}%;
            animation-duration: ${animationDuration}s;
            animation-delay: ${delay}s;
        `;

        container.appendChild(particle);
    }

    // Accessibility enhancements
    setupAccessibility() {
        // Keyboard navigation for cards
        document.querySelectorAll('.feature-card, .metric-card').forEach(card => {
            card.setAttribute('tabindex', '0');
            card.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    card.click();
                }
            });
        });

        // Skip to main content link
        const skipLink = document.createElement('a');
        skipLink.href = '#main';
        skipLink.textContent = 'Skip to main content';
        skipLink.className = 'sr-only';
        skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 6px;
            background: #4f46e5;
            color: white;
            padding: 8px;
            text-decoration: none;
            z-index: 1000;
            transition: top 0.3s;
        `;
        
        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '6px';
        });
        
        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });

        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    // Enhanced typing effect
    typeWriter(selector, text, speed = 50) {
        const element = document.querySelector(selector);
        if (!element) return;

        element.textContent = '';
        let i = 0;
        
        const type = () => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            } else {
                element.style.borderRight = 'none'; // Remove cursor
            }
        };

        element.style.borderRight = '2px solid white';
        setTimeout(type, 1000);
    }

    // Demo button with server status check
    setupDemoButton() {
        const demoButton = document.querySelector('a[href="http://localhost:8501"]');
        if (demoButton) {
            demoButton.addEventListener('click', async (e) => {
                e.preventDefault();
                
                demoButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Checking server...';
                demoButton.style.pointerEvents = 'none';

                try {
                    // Check if server is running
                    const response = await fetch('http://localhost:8501', { mode: 'no-cors' });
                    
                    setTimeout(() => {
                        window.open('http://localhost:8501', '_blank');
                        demoButton.innerHTML = '<i class="fas fa-external-link-alt"></i> Launch Application';
                        demoButton.style.pointerEvents = 'auto';
                    }, 1000);
                    
                } catch (error) {
                    demoButton.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Server not running';
                    demoButton.style.background = '#ef4444';
                    
                    setTimeout(() => {
                        if (confirm('Streamlit server is not running. Would you like to see setup instructions?')) {
                            this.showSetupInstructions();
                        }
                        demoButton.innerHTML = '<i class="fas fa-external-link-alt"></i> Launch Application';
                        demoButton.style.background = '#4f46e5';
                        demoButton.style.pointerEvents = 'auto';
                    }, 2000);
                }
            });
        }
    }

    // Feature cards interactive effects
    setupFeatureCards() {
        document.querySelectorAll('.feature-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-10px) scale(1.02)';
                card.style.boxShadow = '0 12px 40px rgba(0,0,0,0.2)';
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) scale(1)';
                card.style.boxShadow = '0 4px 20px rgba(0,0,0,0.1)';
            });

            // Add click to expand functionality
            card.addEventListener('click', () => {
                this.expandCard(card);
            });
        });
    }

    // Tech stack hover effects
    setupTechStackHovers() {
        document.querySelectorAll('.tech-list li').forEach(item => {
            item.addEventListener('mouseenter', () => {
                item.style.background = '#f8fafc';
                item.style.transform = 'translateX(5px)';
                item.style.borderLeft = '3px solid #4f46e5';
            });

            item.addEventListener('mouseleave', () => {
                item.style.background = 'transparent';
                item.style.transform = 'translateX(0)';
                item.style.borderLeft = 'none';
            });
        });
    }

    // Smooth scrolling with nav offset
    setupSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    const navHeight = document.querySelector('nav').offsetHeight;
                    const targetPosition = target.offsetTop - navHeight - 20;
                    
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    // Animate number values
    animateValue(element, start, end, suffix = '', duration = 2000) {
        const increment = (end - start) / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= end) {
                current = end;
                clearInterval(timer);
            }
            element.textContent = current.toFixed(1) + suffix;
        }, 16);
    }

    // Expand card functionality
    expandCard(card) {
        const isExpanded = card.classList.contains('expanded');
        
        // Reset all cards
        document.querySelectorAll('.feature-card').forEach(c => {
            c.classList.remove('expanded');
            c.style.position = 'relative';
            c.style.zIndex = '1';
        });

        if (!isExpanded) {
            card.classList.add('expanded');
            card.style.position = 'relative';
            card.style.zIndex = '10';
            
            // Add expanded content
            if (!card.querySelector('.expanded-content')) {
                const expandedContent = document.createElement('div');
                expandedContent.className = 'expanded-content';
                expandedContent.innerHTML = `
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
                        <h4>Technical Details:</h4>
                        <p>This component integrates seamlessly with the other AI systems to provide comprehensive medical assistance.</p>
                        <button onclick="this.parentElement.parentElement.remove()" style="margin-top: 1rem; padding: 0.5rem 1rem; background: #4f46e5; color: white; border: none; border-radius: 4px; cursor: pointer;">Close Details</button>
                    </div>
                `;
                card.appendChild(expandedContent);
            }
        }
    }

    // Setup instructions modal
    showSetupInstructions() {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;

        modal.innerHTML = `
            <div style="background: white; padding: 2rem; border-radius: 12px; max-width: 600px; max-height: 80vh; overflow-y: auto;">
                <h3>🚀 Setup Instructions</h3>
                <p>To run the MediAid AI application locally:</p>
                <ol>
                    <li>Clone the repository: <code>git clone https://github.com/anumohan10/Mediaid-AI.git</code></li>
                    <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
                    <li>Set up OpenAI API key in .env file</li>
                    <li>Run: <code>streamlit run streamlit_app2.py</code></li>
                </ol>
                <button onclick="this.parentElement.parentElement.remove()" style="margin-top: 1rem; padding: 0.5rem 1rem; background: #4f46e5; color: white; border: none; border-radius: 4px; cursor: pointer;">Close</button>
            </div>
        `;

        document.body.appendChild(modal);

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    // Welcome message in console
    displayWelcomeMessage() {
        const styles = [
            'color: #4f46e5',
            'font-size: 16px',
            'font-weight: bold'
        ].join(';');

        console.log('%c🏥 MediAid AI - Advanced Generative AI Medical Assistant', styles);
        console.log(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ Features:
• RAG System with 94.2% accuracy
• Multimodal document analysis
• Synthetic data generation
• Advanced prompt engineering
• Task decomposition & routing

📊 Performance Metrics:
• Response Time: <500ms
• System Uptime: 99.8%
• ML Model Accuracy: 87.5% (Heart), 75.3% (Diabetes)

🚀 GitHub: https://github.com/anumohan10/Mediaid-AI
🎓 Academic Achievement: 250% over-requirement (5/2 components)

Built with ❤️ for advancing medical AI and improving healthcare accessibility
        `);
    }
}

// Initialize website functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MediAidWebsite();
});

// Additional utility functions
const Utils = {
    // Debounce function for performance
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Check if element is in viewport
    isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    },

    // Get system theme preference
    getThemePreference() {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }
        return 'light';
    }
};

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MediAidWebsite, Utils };
}
