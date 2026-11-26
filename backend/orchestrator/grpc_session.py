"""Attach the user's session token to every gRPC call."""

from shared_common.common.log import get_trace_id

SESSION_METADATA_KEY = "session-token"
TRACE_METADATA_KEY = "x-trace-id"


class SessionStub:
    """Wrap a gRPC stub so each call carries the caller's session token."""

    def __init__(self, stub, token: str):
        self.stub = stub
        self._metadata = ((SESSION_METADATA_KEY, token),)

    def __getattr__(self, name):
        rpc = getattr(self.stub, name)

        def call(request, **kwargs):
            metadata = self._metadata
            trace_id = get_trace_id()
            if trace_id != "-":
                metadata += ((TRACE_METADATA_KEY, trace_id),)
            return rpc(request, metadata=metadata, **kwargs)

        return call
