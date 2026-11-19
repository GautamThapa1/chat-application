"""Attach the user's session token to every gRPC call."""

SESSION_METADATA_KEY = "session-token"


class SessionStub:
    """Wrap a gRPC stub so each call carries the caller's session token."""

    def __init__(self, stub, token: str):
        self.stub = stub
        self._metadata = ((SESSION_METADATA_KEY, token),)

    def __getattr__(self, name):
        rpc = getattr(self.stub, name)

        def call(request, **kwargs):
            return rpc(request, metadata=self._metadata, **kwargs)

        return call
