// Version drop-down for the docs sidebar.
//
// The docs site is laid out as one directory per release version, with the
// newest release mirrored at the site root:
//
//   https://<host>/<site>/            <- mirror of the newest release
//   https://<host>/<site>/0.6.0/
//   https://<host>/<site>/0.7.0/
//
// The deployment workflow maintains a versions.json manifest at the site
// root, e.g. {"latest": "0.7.0", "versions": ["0.7.0", "0.6.0"]}. This
// script fetches the manifest and inserts a version selector below the
// sidebar search box. When switching versions it tries to stay on the
// current page, falling back to the target version's landing page if the
// page does not exist there.
(function () {
  "use strict";

  var script = document.currentScript;
  if (!script || !window.fetch) return;

  // .../<docs-root>/_static/version-switcher.js -> <docs-root>/
  var docRoot = new URL("../", script.src).href;

  function relativePagePath() {
    var page = new URL(window.location.href);
    var root = new URL(docRoot);
    if (page.pathname.indexOf(root.pathname) === 0) {
      return page.pathname.slice(root.pathname.length) + page.hash;
    }
    return "";
  }

  function insertSelector(manifest, siteRoot) {
    var versions = manifest.versions || [];
    if (versions.length < 2) return;

    var sidebar = document.querySelector(".wy-side-nav-search");
    if (!sidebar) return;

    var current = null;
    for (var i = 0; i < versions.length; i++) {
      if (docRoot === new URL(versions[i] + "/", siteRoot).href) {
        current = versions[i];
      }
    }
    // Pages at the site root are a mirror of the newest release.
    if (current === null) current = manifest.latest;

    var select = document.createElement("select");
    select.setAttribute("aria-label", "Documentation version");
    select.style.cssText =
      "display:block;margin:0.5em auto 0;padding:0.2em 0.5em;" +
      "border-radius:3px;border:0;font-size:90%;";
    for (var j = 0; j < versions.length; j++) {
      var option = document.createElement("option");
      option.value = versions[j];
      option.textContent = "v" + versions[j];
      option.selected = versions[j] === current;
      select.appendChild(option);
    }

    select.addEventListener("change", function () {
      var targetRoot = new URL(select.value + "/", siteRoot).href;
      var samePage = targetRoot + relativePagePath();
      fetch(samePage, { method: "HEAD" })
        .then(function (response) {
          window.location.href = response.ok ? samePage : targetRoot;
        })
        .catch(function () {
          window.location.href = targetRoot;
        });
    });

    sidebar.appendChild(select);
  }

  // The manifest lives at the site root: one level up when this page is
  // inside a version directory, or alongside the page for the root mirror.
  fetch(new URL("../versions.json", docRoot).href)
    .then(function (response) {
      if (!response.ok) throw new Error("no parent manifest");
      return response.json().then(function (manifest) {
        insertSelector(manifest, new URL("../", docRoot).href);
      });
    })
    .catch(function () {
      return fetch(new URL("versions.json", docRoot).href)
        .then(function (response) {
          if (!response.ok) throw new Error("no manifest");
          return response.json();
        })
        .then(function (manifest) {
          insertSelector(manifest, docRoot);
        })
        .catch(function () {
          /* No manifest (e.g. local build): no selector. */
        });
    });
})();
