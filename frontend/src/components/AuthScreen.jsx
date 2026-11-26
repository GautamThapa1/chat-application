import { useEffect, useRef, useState } from "react";
import { useStore } from "../StoreContext.js";

export default function AuthScreen() {
  const { state, actions } = useStore();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const usernameRef = useRef(null);
  const registering = state.authMode === "register";

  useEffect(() => {
    usernameRef.current?.focus();
  }, [state.authFocusSeq]);

  async function onSubmit(event) {
    event.preventDefault();
    if (await actions.submitAuth(username, password)) setPassword("");
  }

  return (
    <section id="auth" className="auth" aria-labelledby="auth-title">
      <div className="auth-card">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">✳</div>
          <span>Luma Chat</span>
        </div>
        <h2 id="auth-title">{registering ? "Create account" : "Log in"}</h2>
        <form id="auth-form" noValidate onSubmit={onSubmit}>
          <label>
            Username
            <input
              id="auth-username"
              ref={usernameRef}
              autoComplete="username"
              maxLength={32}
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label>
            Password
            <input
              id="auth-password"
              type="password"
              autoComplete={registering ? "new-password" : "current-password"}
              maxLength={128}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <p id="auth-error" className="auth-error" role="alert">{state.authError}</p>
          <button id="auth-submit" className="button primary" type="submit" disabled={state.authSubmitting}>
            {registering ? "Create account" : "Log in"}
          </button>
        </form>
        <p className="auth-switch">
          <span id="auth-switch-text">{registering ? "Already have an account?" : "No account yet?"}</span>{" "}
          <button id="auth-switch" className="link-button" type="button" onClick={actions.toggleAuthMode}>
            {registering ? "Log in" : "Create one"}
          </button>
        </p>
      </div>
    </section>
  );
}
