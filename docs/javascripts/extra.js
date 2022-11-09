// ----- ----- ----- ----- -----
// gsap scrollTrigger management on landing page
gsap.registerPlugin('scrollTrigger');

const slideUpElements = gsap.utils.toArray('.slide-up');

slideUpElements.forEach(function (element) {
    gsap.from(element, {
        scrollTrigger: {
            trigger: element,
            toggleActions: 'restart none none reset'
            // markers: true
        },
        y: 48,
        duration: 0.48
    });
})

// ----- ----- ----- ----- -----
// modal management on landing page
const modal = document.querySelector('.modal');

function toggleModal() {
    modal.classList.toggle('show-modal');
}

function windowOnClick(event) {
    if (event.target === modal) {
        toggleModal();
    }
}

window.addEventListener('click', windowOnClick);