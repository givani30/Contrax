window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

function contraxTypesetMath() {
  if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") {
    return;
  }
  if (typeof window.MathJax.typesetClear === "function") {
    window.MathJax.typesetClear();
  }
  window.MathJax.typesetPromise();
}

window.addEventListener("load", () => {
  if (
    typeof document$ !== "undefined" &&
    document$ &&
    typeof document$.subscribe === "function"
  ) {
    document$.subscribe(() => {
      contraxTypesetMath();
    });
  }
});
