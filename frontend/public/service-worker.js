self.addEventListener("push", (event) => {
  if (!event.data) return;
  const payload = event.data.json();
  event.waitUntil(
    self.registration.showNotification(payload.title || "Task reminder", {
      body: payload.body || "A task deadline is coming up.",
      tag: `task-${payload.taskId}`,
      renotify: false,
      data: { url: "/" },
    }),
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then(
      (clients) => {
        const client = clients[0];
        if (client) return client.focus();
        return self.clients.openWindow("/");
      },
    ),
  );
});
