document.addEventListener("DOMContentLoaded", function () {
    const textElement = document.getElementById("typing-text");
    const words = ["farm smarter not harder", "developed under Sarkar S. by Chatterjee A."];
    let wordIndex = 0;
    let charIndex = 0;
    let isDeleting = false;

    function type() {
        const currentWord = words[wordIndex];

        if (isDeleting) {
            charIndex--; // Deleting characters
        } else {
            charIndex++; // Typing characters
        }

        textElement.textContent = currentWord.substring(0, charIndex);

        let speed = isDeleting ? 50 : 100; // Speed for deleting & typing

        if (!isDeleting && charIndex === currentWord.length) {
            speed = 2000; // Pause after typing
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            wordIndex = (wordIndex + 1) % words.length; // Switch to next text
            speed = 500; // Pause before typing next
        }

        setTimeout(type, speed);
    }

    type();
});
