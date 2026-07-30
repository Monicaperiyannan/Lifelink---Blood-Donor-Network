/* ============================================================
   LifeLink — main.js
   Navbar, scroll animations, counters, cross-page section scroll
   ============================================================ */

'use strict';

// ── 1. Navbar: add shadow on scroll ───────────────────────
(function initNavbar() {
  const navbar = document.querySelector('.ll-navbar');
  if (!navbar) return;

  const onScroll = () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

// ── 2. Scroll-reveal: .fade-up elements ───────────────────
(function initScrollReveal() {
  const targets = document.querySelectorAll('.fade-up');
  if (!targets.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  targets.forEach((el) => observer.observe(el));
})();

// ── 3. Animated stat counters ─────────────────────────────
(function initCounters() {
  const counters = document.querySelectorAll('[data-count]');
  if (!counters.length) return;

  const animateCounter = (el) => {
    const target   = parseInt(el.dataset.count, 10);
    const suffix   = el.dataset.suffix || '';
    const duration = 1800;
    const step     = 16;
    const increment = target / (duration / step);
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      el.textContent = Math.floor(current).toLocaleString() + suffix;
    }, step);
  };

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  counters.forEach((el) => observer.observe(el));
})();

// ── 4. Active nav link based on current path ──────────────
(function setActiveNav() {
  const links = document.querySelectorAll('.ll-nav-link');
  const path  = window.location.pathname;

  links.forEach((link) => {
    const href = link.getAttribute('href');
    if (!href) return;

    // Strip hash for path comparison
    const linkPath = href.split('#')[0];

    if (linkPath === path && !href.includes('#')) {
      link.classList.add('active');
    }
  });
})();

// ── 5. Cross-Page Section Scroll (Features / About / Contact) ─
// Handles two cases:
//   a) Link clicked while ALREADY on Home page -> smooth scroll, no reload
//   b) Link clicked from ANOTHER page -> browser navigates to
//      home#section, then on load we smooth-scroll with navbar offset

const NAVBAR_OFFSET = 90; // px — accounts for sticky navbar height

function scrollToSection(hash) {
  const target = document.querySelector(hash);
  if (!target) return;

  const top = target.getBoundingClientRect().top + window.pageYOffset - NAVBAR_OFFSET;
  window.scrollTo({ top, behavior: 'smooth' });
}

// (a) Intercept clicks on .js-scroll-link
document.addEventListener('click', (e) => {
  const link = e.target.closest('.js-scroll-link');
  if (!link) return;

  const url = new URL(link.href);
  const isSamePage = url.pathname === window.location.pathname;
  const hash = url.hash;

  if (isSamePage && hash) {
    // Already on the right page — smooth scroll, update URL hash
    e.preventDefault();
    scrollToSection(hash);
    history.pushState(null, '', hash);
  }
  // Otherwise: let the browser navigate normally (cross-page),
  // and the code below will handle scrolling after load.
});

// (b) On page load, if URL has a hash, smooth-scroll to it
window.addEventListener('load', () => {
  if (window.location.hash) {
    // Small delay lets layout/fonts/images settle before measuring offsets
    setTimeout(() => {
      scrollToSection(window.location.hash);
    }, 150);
  }
});

// ── 6. Toast utility (future use) ─────────────────────────
window.showToast = function (message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `ll-toast ll-toast--${type}`;
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 9999;
    background: ${type === 'success' ? '#1A7A4A' : '#C0152A'};
    color: #fff; padding: 0.75rem 1.4rem; border-radius: 0.5rem;
    font-size: 0.9rem; font-weight: 500; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
};