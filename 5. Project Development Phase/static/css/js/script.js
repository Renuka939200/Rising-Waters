// Rising Waters JavaScript

document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    if (form) {

        form.addEventListener("submit", function () {

            const button = document.querySelector("button");

            button.disabled = true;

            button.innerHTML =
                '<i class="fa fa-spinner fa-spin"></i> Predicting...';

        });

    }

    const inputs = document.querySelectorAll("input");

    inputs.forEach((input) => {

        input.addEventListener("focus", function () {

            this.style.border = "2px solid #00c6ff";

        });

        input.addEventListener("blur", function () {

            this.style.border = "none";

        });

    });

});