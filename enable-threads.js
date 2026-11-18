// NOTE: This file creates a service worker that cross-origin-isolates the page (read more here: https://web.dev/coop-coep/) which allows us to use wasm threads.
// Normally you would set the COOP and COEP headers on the server to do this, but Github Pages doesn't allow this, so this is a hack to do that.
//
// The embedder policy is `require-corp` rather than `credentialless`: Safari
// (and so every browser on iOS, which all run WebKit) does not implement
// `credentialless`, so with it the page never becomes cross-origin isolated
// and the threaded wasm module fails when it hands its shared memory to the
// worker threads ("DataCloneError: The object can not be cloned"). Every
// resource these pages load is same-origin, so `require-corp` costs nothing.
//
// While the page is not isolated (no service worker support, a private
// window, a non-secure context) the reason is written into the #loading
// element, if there is one, instead of leaving the page stuck loading.

/* Edited version of: coi-serviceworker v0.1.6 - Guido Zuidhof, licensed under MIT */
// From here: https://github.com/gzuidhof/coi-serviceworker
if(typeof window === 'undefined') {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));

  async function handleFetch(request) {
    if(request.cache === "only-if-cached" && request.mode !== "same-origin") {
      return;
    }
    
    if(request.mode === "no-cors") { // We need to set `credentials` to "omit" for no-cors requests, per this comment: https://bugs.chromium.org/p/chromium/issues/detail?id=1309901#c7
      request = new Request(request.url, {
        cache: request.cache,
        credentials: "omit",
        headers: request.headers,
        integrity: request.integrity,
        destination: request.destination,
        keepalive: request.keepalive,
        method: request.method,
        mode: request.mode,
        redirect: request.redirect,
        referrer: request.referrer,
        referrerPolicy: request.referrerPolicy,
        signal: request.signal,
      });
    }
    
    let r = await fetch(request).catch(e => console.error(e));
    
    if(!r || r.status === 0) {
      return r;
    }

    const headers = new Headers(r.headers);
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    
    return new Response(r.body, { status: r.status, statusText: r.statusText, headers });
  }

  self.addEventListener("fetch", function(e) {
    e.respondWith(handleFetch(e.request)); // respondWith must be executed synchonously (but can be passed a Promise)
  });
  
} else {
  (async function() {
    if(window.crossOriginIsolated !== false) return;

    const report = message => {
      console.error(message);
      const loading = document.getElementById("loading");
      if(loading) loading.textContent = message;
    };
    if(!window.isSecureContext) {
      report("This page needs a secure context (https, or localhost) to run its multithreaded wasm module.");
      return;
    }
    if(!navigator.serviceWorker) {
      report("This browser does not offer service workers here (private browsing?), so the page cannot enable the wasm threads it needs.");
      return;
    }

    let registration = await navigator.serviceWorker.register(window.document.currentScript.src).catch(e => report("COOP/COEP Service Worker failed to register: " + e));
    if(registration) {
      console.log("COOP/COEP Service Worker registered", registration.scope);

      registration.addEventListener("updatefound", () => {
        console.log("Reloading page to make use of updated COOP/COEP Service Worker.");
        window.location.reload();
      });

      // If the registration is active, but it's not controlling the page
      if(registration.active && !navigator.serviceWorker.controller) {
        console.log("Reloading page to make use of COOP/COEP Service Worker.");
        window.location.reload();
      }
    }
  })();
}

// Code to deregister:
// let registrations = await navigator.serviceWorker.getRegistrations();
// for(let registration of registrations) {
//   await registration.unregister();
// }
