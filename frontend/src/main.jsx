import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { createChatStore } from "./chatStore.js";
import { StoreContext } from "./StoreContext.js";
import "./index.css";

// Created (and the session checked) once, outside React, so StrictMode's
// double-mount in development cannot open two sockets.
const store = createChatStore();
void store.actions.checkSession();

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <StoreContext.Provider value={store}>
      <App />
    </StoreContext.Provider>
  </StrictMode>,
);
