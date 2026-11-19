"""Authenticate every gRPC call using session-token metadata (default deny)."""

from contextvars import ContextVar

import grpc

from backend.service import db

# Must match the key the orchestrator sends (backend/orchestrator/grpc_session.py).
SESSION_METADATA_KEY = "session-token"

# RPCs callable without a session. Anything NOT listed here requires one, so a
# new RPC added later is protected automatically.
PUBLIC_METHODS = frozenset({
    "/chat.ChatService/Register",
    "/chat.ChatService/Login",
    "/chat.ChatService/Logout",
    "/chat.ChatService/ValidateSession",
})

_current_user: ContextVar[int | None] = ContextVar("current_user", default=None)


def current_user_id() -> int:
    """The authenticated user for the RPC being handled on this thread."""
    user_id = _current_user.get()
    if user_id is None:
        raise RuntimeError("current_user_id() called outside an authenticated RPC")
    return user_id


def _deny_handler(code: grpc.StatusCode, message: str):
    def deny(request, context):
        context.abort(code, message)

    return grpc.unary_unary_rpc_method_handler(deny)


class AuthInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)

        if handler is None or handler_call_details.method in PUBLIC_METHODS:
            return handler

        if handler.request_streaming or handler.response_streaming:
            # Fail closed: refuse what we have not taught the interceptor yet.
            return _deny_handler(
                grpc.StatusCode.UNIMPLEMENTED,
                "Streaming RPCs are not supported by the auth interceptor yet.",
            )

        inner = handler.unary_unary

        def authenticated(request, context):
            # Runs on the worker thread, so a slow DB lookup does not block
            # gRPC's dispatch loop.
            metadata = dict(context.invocation_metadata())
            user_id = db.get_session_user(metadata.get(SESSION_METADATA_KEY, ""))
            if user_id is None:
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid session."
                )
            reset = _current_user.set(user_id)
            try:
                return inner(request, context)
            finally:
                _current_user.reset(reset)

        return grpc.unary_unary_rpc_method_handler(
            authenticated,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )