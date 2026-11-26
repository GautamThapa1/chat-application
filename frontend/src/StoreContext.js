import { createContext, useContext, useSyncExternalStore } from "react";

export const StoreContext = createContext(null);

/** Returns the store and re-renders the caller on every store change. */
export function useStore() {
  const store = useContext(StoreContext);
  useSyncExternalStore(store.subscribe, store.getVersion);
  return store;
}
