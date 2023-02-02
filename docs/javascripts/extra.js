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
// charts helpers
// ----- ----- ----- ----- -----
// modal management on landing page
const chartContainer = document.querySelector('.chart-container');
const sliderActions = document.querySelector('.slider__actions');
const modalContent = document.querySelector('.modal-content');
const chartContent = document.querySelector('.chart-content');
const modal = document.querySelector('.modal');
let addedChartContainer;
let addedSliderActions;
let isModalOpen = false;

function toggleModal() {
    if (!isModalOpen) {
        addedChartContainer = modalContent.appendChild(chartContainer);
        addedSliderActions = modalContent.appendChild(sliderActions);
    } else {
        chartContent.appendChild(addedChartContainer);
        chartContent.appendChild(addedSliderActions);
    }
    modal.classList.toggle('show-modal');
    isModalOpen = !isModalOpen;
}

function windowOnClick(event) {
    if (event.target === modal) {
        toggleModal();
    }
}

window.addEventListener('click', windowOnClick);

// ----- ----- ----- ----- -----
// slider management on landing page
const sliderContents = document.querySelectorAll('.slider canvas');

let currentSlide = 0;

function nextSlide() {
    goToSlide(currentSlide + 1);
}

function previousSlide() {
    goToSlide(currentSlide - 1);
}

function goToSlide(n) {
    sliderContents[currentSlide].className = 'slide';
    currentSlide = (n + sliderContents.length) % sliderContents.length;
    sliderContents[currentSlide].className = 'slide showing';
}