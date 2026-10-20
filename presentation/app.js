/* =========================================================
   Q-SPINN Presentation — Navigation & Animations
   ========================================================= */

(function () {
  'use strict';

  // ── State ──
  const slides = Array.from(document.querySelectorAll('.slide'));
  let current = 0;
  const total = slides.length;

  // ── DOM refs ──
  const progressBar = document.getElementById('progressBar');
  const slideCounter = document.getElementById('slideCounter');
  const navPrev = document.getElementById('navPrev');
  const navNext = document.getElementById('navNext');

  // ── Helpers ──
  function pad(n) {
    return String(n).padStart(2, '0');
  }

  function updateUI() {
    const pct = ((current + 1) / total) * 100;
    progressBar.style.width = pct + '%';
    slideCounter.textContent = pad(current + 1) + ' / ' + pad(total);

    navPrev.style.opacity = current === 0 ? '0.1' : '';
    navPrev.style.pointerEvents = current === 0 ? 'none' : '';
    navNext.style.opacity = current === total - 1 ? '0.1' : '';
    navNext.style.pointerEvents = current === total - 1 ? 'none' : '';
  }

  // ── Navigation ──
  function goTo(index) {
    if (index < 0 || index >= total || index === current) return;

    const oldSlide = slides[current];
    oldSlide.classList.remove('active');
    resetAnimations(oldSlide);

    current = index;

    const newSlide = slides[current];
    newSlide.classList.add('active');
    triggerSlideAnimations(newSlide);

    updateUI();
  }

  function next() { goTo(current + 1); }
  function prev() { goTo(current - 1); }

  // ── Tick Counter Animation (Slide 2) ──
  let tickCounterInterval;

  function startTickCounter(slide) {
    const counterEl = slide.querySelector('#tick-counter');
    if (!counterEl) return;
    
    let currentCount = 0;
    const targetBase = 2847563;
    let step = Math.floor(targetBase / 30); // Reach target in ~1 second (30 frames at 33ms)
    
    counterEl.textContent = currentCount.toLocaleString('en-IN');

    tickCounterInterval = setInterval(() => {
      if (currentCount < targetBase) {
        currentCount += step;
        if (currentCount > targetBase) currentCount = targetBase;
      } else {
        // Once reached, simulate roughly 100,000 ticks per second (approx 3300 per 33ms frame)
        currentCount += Math.floor(Math.random() * 4000) + 1000;
      }
      counterEl.textContent = currentCount.toLocaleString('en-IN');
    }, 33);
  }

  function stopTickCounter() {
    if (tickCounterInterval) {
      clearInterval(tickCounterInterval);
      tickCounterInterval = null;
    }
  }

  // ── Animation Engine ──
  //
  // All animations are CSS-driven via the .slide.active selector.
  // This list covers every animated class so we can force-reset
  // them when navigating away, and force-restart on entry.

  const ANIM_SELECTORS = [
    // Core entrance animations
    '.anim-in', '.draw-in', '.bar-grow',
    // Slide 1: wave float
    '.wave-float',
    // Slide 2: regime shift
    '.regime-bar', '.regime-flash-overlay',
    // Slide 5: scope target
    '.scope-ring-outer', '.scope-ring-mid', '.scope-ring-inner',
    '.scope-bullseye', '.scope-bullseye-glow',
    // Slide 7: pipeline particle + pillar icons
    '.pipeline-particle',
    '.pillar--snn .icon-svg--lg',
    '.pillar--quantum .icon-svg--lg',
    '.pillar--pinn .icon-svg--lg',
    // Slide 9: load bars
    '.load-bar-fill',
    // Slide 10: icon idle animations
    '.icon-bob', '.icon-pulse', '.icon-blink',
    // Slide 11: gantt bars
    '.gantt-fill',
    // Slide 13: typewriter
    '.typewriter'
  ];

  function resetAnimations(slide) {
    // Force-reset all CSS animations
    ANIM_SELECTORS.forEach(selector => {
      slide.querySelectorAll(selector).forEach(el => {
        el.style.animation = 'none';
        void el.offsetHeight;
        el.style.animation = '';
      });
    });

    // Reset stateful inline styles
    slide.querySelectorAll('.scope-bullseye').forEach(el => {
      el.style.transform = '';
    });
    slide.querySelectorAll('.gantt-fill').forEach(el => {
      el.style.width = '0%';
    });
    slide.querySelectorAll('.load-bar-fill').forEach(el => {
      el.style.width = '0%';
    });
    slide.querySelectorAll('.typewriter').forEach(el => {
      el.style.width = '0';
    });
    
    if (slide.id === 'slide-intro') {
      stopTickCounter();
    }
  }

  function triggerSlideAnimations(slide) {
    // Force reflow on all animated elements to restart CSS animations
    ANIM_SELECTORS.forEach(selector => {
      slide.querySelectorAll(selector).forEach(el => {
        el.style.animation = 'none';
        void el.offsetHeight;
        el.style.animation = '';
      });
    });
    
    if (slide.id === 'slide-intro') {
      startTickCounter(slide);
    }
  }

  // ── Input Handlers ──

  // Keyboard
  document.addEventListener('keydown', (e) => {
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case ' ':
        e.preventDefault();
        next();
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault();
        prev();
        break;
      case 'Home':
        e.preventDefault();
        goTo(0);
        break;
      case 'End':
        e.preventDefault();
        goTo(total - 1);
        break;
    }
  });

  // Click
  navPrev.addEventListener('click', prev);
  navNext.addEventListener('click', next);

  // Touch / Swipe
  let touchStartX = 0;
  let touchStartY = 0;

  document.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
  }, { passive: true });

  document.addEventListener('touchend', (e) => {
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;

    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 50) {
      if (dx < 0) next();
      else prev();
    }
  }, { passive: true });

  // Mouse Wheel (throttled)
  let wheelThrottle = false;
  document.addEventListener('wheel', (e) => {
    if (wheelThrottle) return;
    wheelThrottle = true;

    if (e.deltaY > 30) next();
    else if (e.deltaY < -30) prev();

    setTimeout(() => { wheelThrottle = false; }, 400);
  }, { passive: true });

  // ── Initialize ──
  updateUI();
  triggerSlideAnimations(slides[0]);

})();
